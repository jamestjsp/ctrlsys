/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB03OT - Solve reduced Lyapunov equation for triangular factors
 *
 * Solves for X = op(U)'*op(U) either the stable continuous-time Lyapunov equation:
 *     op(S)'*X + X*op(S) = -scale^2*op(R)'*op(R)
 * or the convergent discrete-time Lyapunov equation:
 *     op(S)'*X*op(S) - X = -scale^2*op(R)'*op(R)
 *
 * where S is block upper triangular (real Schur form), R is upper triangular.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void sb03ot(
    const bool discr,
    const bool ltrans,
    const i32 n,
    f64* s,
    const i32 lds,
    f64* r,
    const i32 ldr,
    f64* scale,
    f64* dwork,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 two = 2.0;

    bool cont, tbyt;
    i32 infom, isgn, j, j1, j2, j3, k, k1, k2, k3, kount, ksize;
    f64 absskk, alpha, bignum, d1, d2, dr, eps, scaloc, smin;
    f64 smlnum, sum, t1, t2, t3, t4, tau1, tau2, temp, v1, v2, v3, v4;
    f64 a[4], b[4], u[4];

    *info = 0;

    if (n < 0) {
        *info = -3;
        return;
    }
    if (lds < (n > 1 ? n : 1)) {
        *info = -5;
        return;
    }
    if (ldr < (n > 1 ? n : 1)) {
        *info = -7;
        return;
    }

    *scale = one;

    if (n == 0) {
        return;
    }

    eps = SLC_DLAMCH("P");
    smlnum = SLC_DLAMCH("S");
    bignum = one / smlnum;
    SLC_DLABAD(&smlnum, &bignum);
    smlnum = smlnum * (f64)(n*n) / eps;
    bignum = one / smlnum;

    f64 snorm = SLC_DLANGE("M", &n, &n, s, &lds, dwork);
    smin = fmax(smlnum, eps * snorm);
    infom = 0;

    cont = !discr;
    isgn = 1;

    if (!ltrans) {
        kount = 0;

        while (kount < n) {
            k = kount;
            if (kount >= n - 1) {
                tbyt = false;
                kount++;
            } else if (s[(k+1) + k*lds] == zero) {
                tbyt = false;
                kount++;
            } else {
                tbyt = true;
                if ((k + 2) < n) {
                    if (s[(k+2) + (k+1)*lds] != zero) {
                        *info = 3;
                        return;
                    }
                }
                kount += 2;
            }

            if (tbyt) {
                b[0] = s[k + k*lds];
                b[1] = s[(k+1) + k*lds];
                b[2] = s[k + (k+1)*lds];
                b[3] = s[(k+1) + (k+1)*lds];
                u[0] = r[k + k*ldr];
                u[2] = r[k + (k+1)*ldr];
                u[3] = r[(k+1) + (k+1)*ldr];

                sb03oy(discr, ltrans, isgn, b, 2, u, 2, a, 2, &scaloc, info);
                if (*info > 1) return;
                if (*info > infom) infom = *info;

                if (scaloc != one) {
                    for (j = 0; j < n; j++) {
                        i32 jp1 = j + 1;
                        i32 inc = 1;
                        SLC_DSCAL(&jp1, &scaloc, &r[j*ldr], &inc);
                    }
                    *scale *= scaloc;
                }

                r[k + k*ldr] = u[0];
                r[k + (k+1)*ldr] = u[2];
                r[(k+1) + (k+1)*ldr] = u[3];

                if (kount < n) {
                    ksize = n - k - 2;
                    k1 = ksize;
                    k2 = ksize + k1;
                    k3 = ksize + k2;

                    for (j = 0; j < ksize; j++) {
                        dwork[j] = r[k + (k+2+j)*ldr];
                        dwork[k1+j] = r[(k+1) + (k+2+j)*ldr];
                    }

                    i32 two_i = 2;
                    f64 neg_one = -one;
                    SLC_DTRMM("R", "U", "N", "N", &ksize, &two_i, &neg_one, a, &two_i, dwork, &ksize);

                    if (cont) {
                        f64 coef = -r[k + k*ldr];
                        i32 inc = 1;
                        SLC_DAXPY(&ksize, &coef, &s[k + (k+2)*lds], &lds, dwork, &inc);
                        coef = -r[k + (k+1)*ldr];
                        SLC_DAXPY(&ksize, &coef, &s[(k+1) + (k+2)*lds], &lds, dwork, &inc);
                        coef = -r[(k+1) + (k+1)*ldr];
                        SLC_DAXPY(&ksize, &coef, &s[(k+1) + (k+2)*lds], &lds, &dwork[k1], &inc);
                    } else {
                        f64 coef = -r[k + k*ldr] * b[0];
                        i32 inc = 1;
                        SLC_DAXPY(&ksize, &coef, &s[k + (k+2)*lds], &lds, dwork, &inc);
                        coef = -(r[k + (k+1)*ldr]*b[0] + r[(k+1) + (k+1)*ldr]*b[1]);
                        SLC_DAXPY(&ksize, &coef, &s[(k+1) + (k+2)*lds], &lds, dwork, &inc);
                        coef = -r[k + k*ldr] * b[2];
                        SLC_DAXPY(&ksize, &coef, &s[k + (k+2)*lds], &lds, &dwork[k1], &inc);
                        coef = -(r[k + (k+1)*ldr]*b[2] + r[(k+1) + (k+1)*ldr]*b[3]);
                        SLC_DAXPY(&ksize, &coef, &s[(k+1) + (k+2)*lds], &lds, &dwork[k1], &inc);
                    }

                    i32 two_const = 2;
                    sb03or(discr, ltrans, ksize, two_const, &s[(k+2) + (k+2)*lds], lds,
                           b, 2, dwork, ksize, &scaloc, info);
                    if (*info > infom) infom = *info;

                    if (scaloc != one) {
                        for (j = 0; j < n; j++) {
                            i32 jp1 = j + 1;
                            i32 inc = 1;
                            SLC_DSCAL(&jp1, &scaloc, &r[j*ldr], &inc);
                        }
                        *scale *= scaloc;
                    }

                    i32 two_ksize = 2 * ksize;
                    i32 inc = 1;
                    SLC_DCOPY(&two_ksize, dwork, &inc, &dwork[k2], &inc);

                    if (cont) {
                        for (j = 0; j < ksize; j++) {
                            f64 tmp = dwork[j];
                            dwork[j] = r[k + (k+2+j)*ldr];
                            r[k + (k+2+j)*ldr] = tmp;
                            tmp = dwork[k1+j];
                            dwork[k1+j] = r[(k+1) + (k+2+j)*ldr];
                            r[(k+1) + (k+2+j)*ldr] = tmp;
                        }

                        f64 coef = -a[0];
                        SLC_DAXPY(&ksize, &coef, &dwork[k2], &inc, dwork, &inc);
                        coef = -a[2];
                        SLC_DAXPY(&ksize, &coef, &dwork[k3], &inc, dwork, &inc);
                        coef = -a[3];
                        SLC_DAXPY(&ksize, &coef, &dwork[k3], &inc, &dwork[k1], &inc);
                    } else {
                        SLC_DTRMM("L", "U", "T", "N", &ksize, &two_i, &one, &s[(k+2) + (k+2)*lds], &lds, dwork, &ksize);

                        j1 = k1;
                        j2 = k + 2;
                        for (j = 0; j < ksize - 1; j++) {
                            if (s[(j2+1) + j2*lds] != zero) {
                                dwork[j] += s[(j2+1) + j2*lds] * dwork[k2+j+1];
                                dwork[j1] += s[(j2+1) + j2*lds] * dwork[k3+j+1];
                            }
                            j1++;
                            j2++;
                        }

                        f64 coef = r[k + k*ldr];
                        SLC_DAXPY(&ksize, &coef, &s[k + (k+2)*lds], &lds, dwork, &inc);
                        coef = r[k + (k+1)*ldr];
                        SLC_DAXPY(&ksize, &coef, &s[(k+1) + (k+2)*lds], &lds, dwork, &inc);
                        coef = r[(k+1) + (k+1)*ldr];
                        SLC_DAXPY(&ksize, &coef, &s[(k+1) + (k+2)*lds], &lds, &dwork[k1], &inc);

                        for (j = 0; j < ksize; j++) {
                            f64 tmp = dwork[k2+j];
                            dwork[k2+j] = r[k + (k+2+j)*ldr];
                            r[k + (k+2+j)*ldr] = tmp;
                            tmp = dwork[k3+j];
                            dwork[k3+j] = r[(k+1) + (k+2+j)*ldr];
                            r[(k+1) + (k+2+j)*ldr] = tmp;
                        }

                        i32 three = 3;
                        SLC_DLARFG(&three, &a[0], &b[0], &inc, &tau1);
                        v1 = b[0];
                        t1 = tau1 * v1;
                        v2 = b[1];
                        t2 = tau1 * v2;
                        sum = a[2] + v1*b[2] + v2*b[3];
                        b[2] = b[2] - sum*t1;
                        b[3] = b[3] - sum*t2;
                        SLC_DLARFG(&three, &a[3], &b[2], &inc, &tau2);
                        v3 = b[2];
                        t3 = tau2 * v3;
                        v4 = b[3];
                        t4 = tau2 * v4;

                        j1 = k1;
                        j2 = k2;
                        j3 = k3;
                        for (j = 0; j < ksize; j++) {
                            sum = dwork[j2] + v1*dwork[j] + v2*dwork[j1];
                            d1 = dwork[j] - sum*t1;
                            d2 = dwork[j1] - sum*t2;
                            sum = dwork[j3] + v3*d1 + v4*d2;
                            dwork[j] = d1 - sum*t3;
                            dwork[j1] = d2 - sum*t4;
                            j1++;
                            j2++;
                            j3++;
                        }
                    }

                    SLC_DCOPY(&ksize, dwork, &inc, &dwork[k2], &inc);
                    SLC_DCOPY(&ksize, &dwork[k1], &inc, &dwork[k3], &inc);

                    for (j = 0; j < ksize; j++) {
                        dwork[2*j] = dwork[k2+j];
                        dwork[2*j+1] = dwork[k3+j];
                    }

                    mb04od("F", ksize, 0, 2, &r[(k+2) + (k+2)*ldr], ldr, dwork, 2,
                           dwork, 1, dwork, 1, &dwork[k2], &dwork[k3]);
                }
            } else {
                if (discr) {
                    absskk = fabs(s[k + k*lds]);
                    if ((absskk - one) >= zero) {
                        *info = 2;
                        return;
                    }
                    temp = sqrt((one - absskk) * (one + absskk));
                } else {
                    if (s[k + k*lds] >= zero) {
                        *info = 2;
                        return;
                    }
                    temp = sqrt(fabs(two * s[k + k*lds]));
                }

                scaloc = one;
                dr = fabs(r[k + k*ldr]);
                if (temp < one && dr > one) {
                    if (dr > bignum * temp)
                        scaloc = one / dr;
                }

                alpha = copysign(temp, r[k + k*ldr]);
                r[k + k*ldr] = r[k + k*ldr] / alpha;

                if (scaloc != one) {
                    for (j = 0; j < n; j++) {
                        i32 jp1 = j + 1;
                        i32 inc = 1;
                        SLC_DSCAL(&jp1, &scaloc, &r[j*ldr], &inc);
                    }
                    *scale *= scaloc;
                }

                if (kount < n) {
                    ksize = n - k - 1;
                    k1 = ksize;
                    k2 = ksize + k1;

                    i32 inc = 1;
                    SLC_DCOPY(&ksize, &r[k + (k+1)*ldr], &ldr, dwork, &inc);
                    f64 neg_alpha = -alpha;
                    SLC_DSCAL(&ksize, &neg_alpha, dwork, &inc);

                    if (cont) {
                        f64 coef = -r[k + k*ldr];
                        SLC_DAXPY(&ksize, &coef, &s[k + (k+1)*lds], &lds, dwork, &inc);
                    } else {
                        f64 coef = -s[k + k*lds] * r[k + k*ldr];
                        SLC_DAXPY(&ksize, &coef, &s[k + (k+1)*lds], &lds, dwork, &inc);
                    }

                    i32 one_const = 1;
                    sb03or(discr, ltrans, ksize, one_const, &s[(k+1) + (k+1)*lds], lds,
                           &s[k + k*lds], 1, dwork, ksize, &scaloc, info);
                    if (*info > infom) infom = *info;

                    if (scaloc != one) {
                        for (j = 0; j < n; j++) {
                            i32 jp1 = j + 1;
                            SLC_DSCAL(&jp1, &scaloc, &r[j*ldr], &inc);
                        }
                        *scale *= scaloc;
                    }

                    SLC_DCOPY(&ksize, dwork, &inc, &dwork[k1], &inc);

                    for (j = 0; j < ksize; j++) {
                        f64 tmp = dwork[j];
                        dwork[j] = r[k + (k+1+j)*ldr];
                        r[k + (k+1+j)*ldr] = tmp;
                    }

                    if (cont) {
                        f64 coef = -alpha;
                        SLC_DAXPY(&ksize, &coef, &dwork[k1], &inc, dwork, &inc);
                    } else {
                        f64 neg_skk = -s[k + k*lds];
                        SLC_DSCAL(&ksize, &neg_skk, dwork, &inc);
                        f64 coef = alpha * r[k + k*ldr];
                        SLC_DAXPY(&ksize, &coef, &s[k + (k+1)*lds], &lds, dwork, &inc);

                        j1 = k + 1;
                        for (j = 0; j < ksize - 1; j++) {
                            if (s[(j1+1) + j1*lds] != zero) {
                                dwork[j] += alpha * s[(j1+1) + j1*lds] * dwork[k1+j+1];
                            }
                            j1++;
                        }

                        SLC_DTRMV("U", "T", "N", &ksize, &s[(k+1) + (k+1)*lds], &lds, &dwork[k1], &inc);
                        SLC_DAXPY(&ksize, &alpha, &dwork[k1], &inc, dwork, &inc);
                    }

                    mb04od("F", ksize, 0, 1, &r[(k+1) + (k+1)*ldr], ldr, dwork, 1,
                           dwork, 1, dwork, 1, &dwork[k2], &dwork[k1]);
                }
            }
        }
    } else {
        kount = n - 1;

        while (kount >= 0) {
            k = kount;
            if (kount == 0) {
                tbyt = false;
                kount--;
            } else if (s[k + (k-1)*lds] == zero) {
                tbyt = false;
                kount--;
            } else {
                tbyt = true;
                k = k - 1;
                if (k > 0) {
                    if (s[k + (k-1)*lds] != zero) {
                        *info = 3;
                        return;
                    }
                }
                kount -= 2;
            }

            if (tbyt) {
                b[0] = s[k + k*lds];
                b[1] = s[(k+1) + k*lds];
                b[2] = s[k + (k+1)*lds];
                b[3] = s[(k+1) + (k+1)*lds];
                u[0] = r[k + k*ldr];
                u[2] = r[k + (k+1)*ldr];
                u[3] = r[(k+1) + (k+1)*ldr];

                sb03oy(discr, ltrans, isgn, b, 2, u, 2, a, 2, &scaloc, info);
                if (*info > 1) return;
                if (*info > infom) infom = *info;

                if (scaloc != one) {
                    for (j = 0; j < n; j++) {
                        i32 jp1 = j + 1;
                        i32 inc = 1;
                        SLC_DSCAL(&jp1, &scaloc, &r[j*ldr], &inc);
                    }
                    *scale *= scaloc;
                }

                r[k + k*ldr] = u[0];
                r[k + (k+1)*ldr] = u[2];
                r[(k+1) + (k+1)*ldr] = u[3];

                if (kount >= 0) {
                    ksize = k;
                    k1 = ksize;
                    k2 = ksize + k1;
                    k3 = ksize + k2;

                    i32 inc = 1;
                    SLC_DCOPY(&ksize, &r[k*ldr], &inc, dwork, &inc);
                    SLC_DCOPY(&ksize, &r[(k+1)*ldr], &inc, &dwork[k1], &inc);

                    i32 two_i = 2;
                    f64 neg_one = -one;
                    SLC_DTRMM("R", "U", "T", "N", &ksize, &two_i, &neg_one, a, &two_i, dwork, &ksize);

                    if (cont) {
                        f64 coef = -r[k + k*ldr];
                        SLC_DAXPY(&ksize, &coef, &s[k*lds], &inc, dwork, &inc);
                        coef = -r[k + (k+1)*ldr];
                        SLC_DAXPY(&ksize, &coef, &s[k*lds], &inc, &dwork[k1], &inc);
                        coef = -r[(k+1) + (k+1)*ldr];
                        SLC_DAXPY(&ksize, &coef, &s[(k+1)*lds], &inc, &dwork[k1], &inc);
                    } else {
                        f64 coef = -(r[k + k*ldr]*b[0] + r[k + (k+1)*ldr]*b[2]);
                        SLC_DAXPY(&ksize, &coef, &s[k*lds], &inc, dwork, &inc);
                        coef = -r[(k+1) + (k+1)*ldr] * b[2];
                        SLC_DAXPY(&ksize, &coef, &s[(k+1)*lds], &inc, dwork, &inc);
                        coef = -(r[k + k*ldr]*b[1] + r[k + (k+1)*ldr]*b[3]);
                        SLC_DAXPY(&ksize, &coef, &s[k*lds], &inc, &dwork[k1], &inc);
                        coef = -r[(k+1) + (k+1)*ldr] * b[3];
                        SLC_DAXPY(&ksize, &coef, &s[(k+1)*lds], &inc, &dwork[k1], &inc);
                    }

                    i32 two_const = 2;
                    sb03or(discr, ltrans, ksize, two_const, s, lds, b, 2, dwork, ksize, &scaloc, info);
                    if (*info > infom) infom = *info;

                    if (scaloc != one) {
                        for (j = 0; j < n; j++) {
                            i32 jp1 = j + 1;
                            SLC_DSCAL(&jp1, &scaloc, &r[j*ldr], &inc);
                        }
                        *scale *= scaloc;
                    }

                    i32 two_ksize = 2 * ksize;
                    SLC_DCOPY(&two_ksize, dwork, &inc, &dwork[k2], &inc);

                    if (cont) {
                        for (j = 0; j < ksize; j++) {
                            f64 tmp = dwork[j];
                            dwork[j] = r[j + k*ldr];
                            r[j + k*ldr] = tmp;
                            tmp = dwork[k1+j];
                            dwork[k1+j] = r[j + (k+1)*ldr];
                            r[j + (k+1)*ldr] = tmp;
                        }

                        f64 coef = -a[0];
                        SLC_DAXPY(&ksize, &coef, &dwork[k2], &inc, dwork, &inc);
                        coef = -a[2];
                        SLC_DAXPY(&ksize, &coef, &dwork[k2], &inc, &dwork[k1], &inc);
                        coef = -a[3];
                        SLC_DAXPY(&ksize, &coef, &dwork[k3], &inc, &dwork[k1], &inc);
                    } else {
                        SLC_DTRMM("L", "U", "N", "N", &ksize, &two_i, &one, s, &lds, dwork, &ksize);

                        j1 = k1;
                        for (j = 1; j < ksize; j++) {
                            j1++;
                            if (s[j + (j-1)*lds] != zero) {
                                dwork[j] += s[j + (j-1)*lds] * dwork[k2+j-1];
                                dwork[j1] += s[j + (j-1)*lds] * dwork[k3+j-1];
                            }
                        }

                        f64 coef = r[k + k*ldr];
                        SLC_DAXPY(&ksize, &coef, &s[k*lds], &inc, dwork, &inc);
                        coef = r[k + (k+1)*ldr];
                        SLC_DAXPY(&ksize, &coef, &s[k*lds], &inc, &dwork[k1], &inc);
                        coef = r[(k+1) + (k+1)*ldr];
                        SLC_DAXPY(&ksize, &coef, &s[(k+1)*lds], &inc, &dwork[k1], &inc);

                        for (j = 0; j < ksize; j++) {
                            f64 tmp = dwork[k2+j];
                            dwork[k2+j] = r[j + k*ldr];
                            r[j + k*ldr] = tmp;
                            tmp = dwork[k3+j];
                            dwork[k3+j] = r[j + (k+1)*ldr];
                            r[j + (k+1)*ldr] = tmp;
                        }

                        i32 three = 3;
                        i32 two_inc = 2;
                        SLC_DLARFG(&three, &a[3], &b[1], &two_inc, &tau1);
                        v1 = b[1];
                        t1 = tau1 * v1;
                        v2 = b[3];
                        t2 = tau1 * v2;
                        sum = a[2] + v1*b[0] + v2*b[2];
                        b[0] = b[0] - sum*t1;
                        b[2] = b[2] - sum*t2;
                        SLC_DLARFG(&three, &a[0], &b[0], &two_inc, &tau2);
                        v3 = b[0];
                        t3 = tau2 * v3;
                        v4 = b[2];
                        t4 = tau2 * v4;

                        j1 = k1;
                        j2 = k2;
                        j3 = k3;
                        for (j = 0; j < ksize; j++) {
                            sum = dwork[j3] + v1*dwork[j] + v2*dwork[j1];
                            d1 = dwork[j] - sum*t1;
                            d2 = dwork[j1] - sum*t2;
                            sum = dwork[j2] + v3*d1 + v4*d2;
                            dwork[j] = d1 - sum*t3;
                            dwork[j1] = d2 - sum*t4;
                            j1++;
                            j2++;
                            j3++;
                        }
                    }

                    SLC_MB04ND("F", ksize, 0, 2, r, ldr, dwork, ksize, dwork, 1, dwork, 1, &dwork[k2], &dwork[k3]);
                }
            } else {
                if (discr) {
                    absskk = fabs(s[k + k*lds]);
                    if ((absskk - one) >= zero) {
                        *info = 2;
                        return;
                    }
                    temp = sqrt((one - absskk) * (one + absskk));
                } else {
                    if (s[k + k*lds] >= zero) {
                        *info = 2;
                        return;
                    }
                    temp = sqrt(fabs(two * s[k + k*lds]));
                }

                scaloc = one;
                if (temp < smin) {
                    temp = smin;
                    infom = 1;
                }
                dr = fabs(r[k + k*ldr]);
                if (temp < one && dr > one) {
                    if (dr > bignum * temp)
                        scaloc = one / dr;
                }

                alpha = copysign(temp, r[k + k*ldr]);
                r[k + k*ldr] = r[k + k*ldr] / alpha;

                if (scaloc != one) {
                    for (j = 0; j < n; j++) {
                        i32 jp1 = j + 1;
                        i32 inc = 1;
                        SLC_DSCAL(&jp1, &scaloc, &r[j*ldr], &inc);
                    }
                    *scale *= scaloc;
                }

                if (kount >= 0) {
                    ksize = k;
                    k1 = ksize;
                    k2 = ksize + k1;

                    i32 inc = 1;
                    SLC_DCOPY(&ksize, &r[k*ldr], &inc, dwork, &inc);
                    f64 neg_alpha = -alpha;
                    SLC_DSCAL(&ksize, &neg_alpha, dwork, &inc);

                    if (cont) {
                        f64 coef = -r[k + k*ldr];
                        SLC_DAXPY(&ksize, &coef, &s[k*lds], &inc, dwork, &inc);
                    } else {
                        f64 coef = -s[k + k*lds] * r[k + k*ldr];
                        SLC_DAXPY(&ksize, &coef, &s[k*lds], &inc, dwork, &inc);
                    }

                    i32 one_const = 1;
                    sb03or(discr, ltrans, ksize, one_const, s, lds, &s[k + k*lds], 1, dwork, ksize, &scaloc, info);
                    if (*info > infom) infom = *info;

                    if (scaloc != one) {
                        for (j = 0; j < n; j++) {
                            i32 jp1 = j + 1;
                            SLC_DSCAL(&jp1, &scaloc, &r[j*ldr], &inc);
                        }
                        *scale *= scaloc;
                    }

                    SLC_DCOPY(&ksize, dwork, &inc, &dwork[k1], &inc);

                    for (j = 0; j < ksize; j++) {
                        f64 tmp = dwork[j];
                        dwork[j] = r[j + k*ldr];
                        r[j + k*ldr] = tmp;
                    }

                    if (cont) {
                        f64 coef = -alpha;
                        SLC_DAXPY(&ksize, &coef, &dwork[k1], &inc, dwork, &inc);
                    } else {
                        f64 neg_skk = -s[k + k*lds];
                        SLC_DSCAL(&ksize, &neg_skk, dwork, &inc);
                        f64 coef = alpha * r[k + k*ldr];
                        SLC_DAXPY(&ksize, &coef, &s[k*lds], &inc, dwork, &inc);

                        for (j = 1; j < ksize; j++) {
                            if (s[j + (j-1)*lds] != zero) {
                                dwork[j] += alpha * s[j + (j-1)*lds] * dwork[k1+j-1];
                            }
                        }

                        SLC_DTRMV("U", "N", "N", &ksize, s, &lds, &dwork[k1], &inc);
                        SLC_DAXPY(&ksize, &alpha, &dwork[k1], &inc, dwork, &inc);
                    }

                    SLC_MB04ND("F", ksize, 0, 1, r, ldr, dwork, ksize, dwork, 1, dwork, 1, &dwork[k2], &dwork[k1]);
                }
            }
        }
    }

    *info = infom;
}
