/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

void sg03bu(const char* trans, const i32 n, const f64* a, const i32 lda,
            const f64* e, const i32 lde, f64* b, const i32 ldb,
            f64* scale, f64* dwork, i32* info)
{
    const f64 half = 0.5;
    const f64 mone = -1.0;
    const f64 one = 1.0;
    const f64 two = 2.0;
    const f64 zero = 0.0;

    f64 m1[4] = {0}, m2[4] = {0}, m3[16] = {0}, m3c[16] = {0}, m3ew[4] = {0};
    f64 rw[32] = {0}, tm[4] = {0}, ui[4] = {0};
    i32 iw[24] = {0};

    f64 bignum, c, delta1, eps, r, s, scale1, smlnum, t, uflt, x;
    i32 i, info1, j, kb, kh, kl, l, ldws, m, uiipt, wpt, ypt;
    bool notrns;

    const i32 int0 = 0;
    const i32 int1 = 1;
    const i32 int2 = 2;
    const i32 int4 = 4;
    const i32 int32 = 32;

    notrns = (*trans == 'N' || *trans == 'n');

    *info = 0;
    if (!notrns && *trans != 'T' && *trans != 't') {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 0 ? n : 1)) {
        *info = -4;
    } else if (lde < (n > 0 ? n : 1)) {
        *info = -6;
    } else if (ldb < (n > 0 ? n : 1)) {
        *info = -8;
    }

    if (*info != 0) {
        return;
    }

    *scale = one;

    if (n == 0) {
        return;
    }

    eps = SLC_DLAMCH("P");
    uflt = SLC_DLAMCH("S");
    smlnum = uflt / eps;
    bignum = one / smlnum;
    SLC_DLABAD(&smlnum, &bignum);

    uiipt = 0;
    wpt = 2 * n - 2;
    ypt = 4 * n - 4;
    ldws = n - 1;

    if (notrns) {
        kh = -1;

        while (kh < n - 1) {
            kl = kh + 1;
            if (kl == n - 1) {
                kh = n - 1;
                kb = 1;
            } else {
                if (kl + 1 < n && a[(kl + 1) + kl * lda] == zero) {
                    kh = kl;
                    kb = 1;
                } else {
                    kh = kl + 1;
                    kb = 2;
                }
            }

            if (kb == 1) {
                delta1 = e[kl * lde + kl];
                t = fabs(a[kl * lda + kl]);
                x = (delta1 > t) ? delta1 : t;
                delta1 = delta1 / x;
                t = t / x;
                if (delta1 <= t) {
                    *info = 3;
                    return;
                }
                delta1 = sqrt(one - t) * sqrt(one + t) * x;
                t = b[kl * ldb + kl] * smlnum;
                if (t > delta1) {
                    scale1 = delta1 / t;
                    *scale = scale1 * (*scale);
                    SLC_DLASCL("Upper", &int0, &int0, &one, &scale1, &n, &n,
                              b, &ldb, &info1);
                }

                ui[0] = b[kl * ldb + kl] / delta1;
                m1[0] = a[kl * lda + kl] / e[kl * lde + kl];
                m2[0] = delta1 / e[kl * lde + kl];

            } else {
                sg03bx("D", "N", &a[kl * lda + kl], lda, &e[kl * lde + kl],
                      lde, &b[kl * ldb + kl], ldb, ui, int2, &scale1,
                      m1, int2, m2, int2, info);

                if (*info != 0) {
                    return;
                }

                if (scale1 != one) {
                    *scale = scale1 * (*scale);
                    SLC_DLASCL("Upper", &int0, &int0, &one, &scale1, &n, &n,
                              b, &ldb, &info1);
                }
            }

            if (kh < n - 1) {
                i32 nkh = n - kh - 1;

                SLC_DGEMM("T", "N", &nkh, &kb, &kb, &mone, &b[kl + (kh + 1) * ldb],
                         &ldb, m2, &int2, &zero, &dwork[uiipt], &ldws);

                SLC_DGEMM("T", "T", &nkh, &kb, &kb, &one, &e[kl + (kh + 1) * lde],
                         &lde, ui, &int2, &one, &dwork[uiipt], &ldws);

                SLC_DGEMM("T", "N", &kb, &kb, &kb, &one, ui, &int2, m1, &int2,
                         &zero, tm, &int2);

                SLC_DGEMM("T", "N", &nkh, &kb, &kb, &mone, &a[kl + (kh + 1) * lda],
                         &lda, tm, &int2, &one, &dwork[uiipt], &ldws);

                SLC_DLASET("A", &kb, &kb, &zero, &mone, tm, &int2);

                sg03bw("N", nkh, kb, &a[(kh + 1) * lda + kh + 1], lda,
                      m1, int2, &e[(kh + 1) * lde + kh + 1], lde, tm, int2,
                      &dwork[uiipt], ldws, &scale1, info);

                if (scale1 != one) {
                    *scale = scale1 * (*scale);
                    SLC_DLASCL("Upper", &int0, &int0, &one, &scale1, &n, &n,
                              b, &ldb, &info1);
                    SLC_DSCAL(&int4, &scale1, ui, &int1);
                }

                i32 kb2 = 2 * kb;

                SLC_DLASET("U", &kb2, &kb2, &zero, &one, m3, &int4);
                SLC_DSYRK("U", "N", &kb, &kb, &mone, m2, &int2, &one, m3, &int4);

                SLC_DGEMM("N", "T", &kb, &kb, &kb, &mone, m2, &int2, m1, &int2,
                         &zero, &m3[kb * int4], &int4);

                SLC_DSYRK("U", "N", &kb, &kb, &mone, m1, &int2, &one,
                         &m3[kb + kb * int4], &int4);
                f64 abstol = two * uflt;
                SLC_DSYEVX("V", "V", "U", &kb2, m3, &int4, &half, &two, &int1,
                          &int4, &abstol, &m, m3ew, m3c, &int4, rw, &int32,
                          &iw[4], iw, &info1);

                if (info1 != 0) {
                    *info = 4;
                    return;
                }

                SLC_DGEMM("T", "N", &nkh, &kb, &kb, &one, &b[kl + (kh + 1) * ldb],
                         &ldb, m3c, &int4, &zero, &dwork[ypt], &ldws);

                SLC_DGEMM("T", "T", &nkh, &kb, &kb, &one, &a[kl + (kh + 1) * lda],
                         &lda, ui, &int2, &zero, &dwork[wpt], &ldws);

                for (i = 0; i < nkh; i++) {
                    i32 mini = (i + 2 < nkh) ? i + 2 : nkh;
                    SLC_DGEMV("T", &mini, &kb, &one, &dwork[uiipt], &ldws,
                             &a[(kh + i + 1) * lda + kh + 1], &int1, &one,
                             &dwork[wpt + i], &ldws);
                }

                SLC_DGEMM("N", "N", &nkh, &kb, &kb, &one, &dwork[wpt], &ldws,
                         &m3c[kb + 0], &int4, &one, &dwork[ypt], &ldws);

                l = ypt;
                for (j = 0; j < kb; j++) {
                    for (i = 0; i < nkh; i++) {
                        x = b[(kh + i + 1) * ldb + kh + i + 1];
                        t = dwork[l + i];
                        SLC_DLARTG(&x, &t, &c, &s, &r);
                        b[(kh + i + 1) * ldb + kh + i + 1] = r;
                        if (i < nkh - 1) {
                            i32 nmihm1 = nkh - i - 1;
                            SLC_DROT(&nmihm1, &b[(kh + i + 2) * ldb + kh + i + 1],
                                    &ldb, &dwork[l + i + 1], &int1, &c, &s);
                        }
                    }
                    l += ldws;
                }

                for (i = kh + 1; i < n; i++) {
                    if (b[i * ldb + i] < zero) {
                        i32 nmi = n - i;
                        SLC_DSCAL(&nmi, &mone, &b[i * ldb + i], &ldb);
                    }
                }

                SLC_DCOPY(&nkh, &dwork[uiipt], &int1, &b[(kh + 1) * ldb + kl],
                         &ldb);
                if (kh > kl) {
                    SLC_DCOPY(&nkh, &dwork[uiipt + ldws], &int1,
                             &b[(kh + 1) * ldb + kh], &ldb);
                }
            }

            SLC_DLACPY("U", &kb, &kb, ui, &int2, &b[kl * ldb + kl], &ldb);
        }

    } else {
        kl = n;

        while (kl > 0) {
            kh = kl - 1;
            if (kh == 0) {
                kl = 0;
                kb = 1;
            } else {
                if (kh > 0 && a[kh + (kh - 1) * lda] == zero) {
                    kl = kh;
                    kb = 1;
                } else {
                    kl = kh - 1;
                    kb = 2;
                }
            }

            if (kb == 1) {
                delta1 = e[kl * lde + kl];
                t = fabs(a[kl * lda + kl]);
                x = (delta1 > t) ? delta1 : t;
                delta1 = delta1 / x;
                t = t / x;
                if (delta1 <= t) {
                    *info = 3;
                    return;
                }
                delta1 = sqrt(one - t) * sqrt(one + t) * x;
                t = b[kl * ldb + kl] * smlnum;
                if (t > delta1) {
                    scale1 = delta1 / t;
                    *scale = scale1 * (*scale);
                    SLC_DLASCL("Upper", &int0, &int0, &one, &scale1, &n, &n,
                              b, &ldb, &info1);
                }

                ui[0] = b[kl * ldb + kl] / delta1;
                m1[0] = a[kl * lda + kl] / e[kl * lde + kl];
                m2[0] = delta1 / e[kl * lde + kl];

            } else {
                sg03bx("D", "T", &a[kl * lda + kl], lda, &e[kl * lde + kl],
                      lde, &b[kl * ldb + kl], ldb, ui, int2, &scale1,
                      m1, int2, m2, int2, info);

                if (*info != 0) {
                    return;
                }

                if (scale1 != one) {
                    *scale = scale1 * (*scale);
                    SLC_DLASCL("Upper", &int0, &int0, &one, &scale1, &n, &n,
                              b, &ldb, &info1);
                }
            }

            if (kl > 0) {
                SLC_DGEMM("N", "T", &kl, &kb, &kb, &mone, &b[kl * ldb],
                         &ldb, m2, &int2, &zero, &dwork[uiipt], &ldws);

                SLC_DGEMM("N", "N", &kl, &kb, &kb, &one, &e[kl * lde],
                         &lde, ui, &int2, &one, &dwork[uiipt], &ldws);

                SLC_DGEMM("N", "T", &kb, &kb, &kb, &one, ui, &int2, m1, &int2,
                         &zero, tm, &int2);

                SLC_DGEMM("N", "N", &kl, &kb, &kb, &mone, &a[kl * lda],
                         &lda, tm, &int2, &one, &dwork[uiipt], &ldws);

                SLC_DLASET("A", &kb, &kb, &zero, &mone, tm, &int2);

                sg03bw("T", kl, kb, a, lda, m1, int2, e, lde, tm, int2,
                      &dwork[uiipt], ldws, &scale1, info);

                if (scale1 != one) {
                    *scale = scale1 * (*scale);
                    SLC_DLASCL("Upper", &int0, &int0, &one, &scale1, &n, &n,
                              b, &ldb, &info1);
                    SLC_DSCAL(&int4, &scale1, ui, &int1);
                }

                i32 kb2 = 2 * kb;

                SLC_DLASET("U", &kb2, &kb2, &zero, &one, m3, &int4);
                SLC_DSYRK("U", "T", &kb, &kb, &mone, m2, &int2, &one, m3, &int4);

                SLC_DGEMM("T", "N", &kb, &kb, &kb, &mone, m2, &int2, m1, &int2,
                         &zero, &m3[kb * int4], &int4);

                SLC_DSYRK("U", "T", &kb, &kb, &mone, m1, &int2, &one,
                         &m3[kb + kb * int4], &int4);
                f64 abstol = two * uflt;
                SLC_DSYEVX("V", "V", "U", &kb2, m3, &int4, &half, &two, &int1,
                          &int4, &abstol, &m, m3ew, m3c, &int4, rw, &int32,
                          &iw[4], iw, &info1);

                if (info1 != 0) {
                    *info = 4;
                    return;
                }

                SLC_DGEMM("N", "N", &kl, &kb, &kb, &one, &b[kl * ldb],
                         &ldb, m3c, &int4, &zero, &dwork[ypt], &ldws);

                SLC_DGEMM("N", "N", &kl, &kb, &kb, &one, &a[kl * lda],
                         &lda, ui, &int2, &zero, &dwork[wpt], &ldws);

                for (i = 0; i < kl; i++) {
                    i32 mini = (kl - i + 1 < kl) ? kl - i + 1 : kl;
                    i32 maxi = (i > 1) ? i - 1 : 0;
                    i32 uiipt_adj = (i >= 1) ? uiipt + i - 1 : uiipt;
                    SLC_DGEMV("T", &mini, &kb, &one, &dwork[uiipt_adj], &ldws,
                             &a[maxi * lda + i], &lda, &one,
                             &dwork[wpt + i], &ldws);
                }

                SLC_DGEMM("N", "N", &kl, &kb, &kb, &one, &dwork[wpt], &ldws,
                         &m3c[kb + 0], &int4, &one, &dwork[ypt], &ldws);

                l = ypt;
                for (j = 0; j < kb; j++) {
                    for (i = kl - 1; i >= 0; i--) {
                        x = b[i * ldb + i];
                        t = dwork[l + i];
                        SLC_DLARTG(&x, &t, &c, &s, &r);
                        b[i * ldb + i] = r;
                        if (i > 0) {
                            SLC_DROT(&i, &b[i * ldb], &int1, &dwork[l], &int1, &c, &s);
                        }
                    }
                    l += ldws;
                }

                for (i = 0; i < kl; i++) {
                    if (b[i * ldb + i] < zero) {
                        i32 ip1 = i + 1;
                        SLC_DSCAL(&ip1, &mone, &b[i * ldb], &int1);
                    }
                }

                SLC_DLACPY("A", &kl, &kb, &dwork[uiipt], &ldws, &b[kl * ldb],
                          &ldb);
            }

            SLC_DLACPY("U", &kb, &kb, ui, &int2, &b[kl * ldb + kl], &ldb);
        }
    }
}
