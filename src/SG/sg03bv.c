/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

void sg03bv(const char* trans, const i32 n, const f64* a, const i32 lda,
            const f64* e, const i32 lde, f64* b, const i32 ldb,
            f64* scale, f64* dwork, i32* info)
{
    const f64 mone = -1.0;
    const f64 one = 1.0;
    const f64 two = 2.0;
    const f64 zero = 0.0;

    f64 m1[4], m2[4], tm[4], ui[4];

    f64 bignum, c, delta1, eps, r, s, scale1, smlnum, sqtwo, t, x;
    i32 i, info1, j, kb, kh, kl, kl1, l, ldws, uiipt, wpt, ypt;
    bool notrns;

    const i32 int0 = 0;
    const i32 int1 = 1;
    const i32 int2 = 2;

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

    sqtwo = sqrt(two);
    eps = SLC_DLAMCH("P");
    smlnum = SLC_DLAMCH("S") / eps;
    bignum = one / smlnum;
    SLC_DLABAD(&smlnum, &bignum);

    uiipt = 0;
    wpt = 2 * n - 2;
    ypt = 4 * n - 4;
    ldws = n - 1;

    if (notrns) {
        kh = 0;

        while (kh < n) {
            kl = kh;
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
                delta1 = -a[kl * lda + kl];
                if (delta1 <= zero) {
                    *info = 3;
                    return;
                }
                delta1 = sqrt(delta1) * sqrt(e[kl * lde + kl]);
                t = (b[kl * ldb + kl] * smlnum) / sqtwo;
                if (t > delta1) {
                    scale1 = delta1 / t;
                    *scale = scale1 * (*scale);
                    SLC_DLASCL("Upper", &int0, &int0, &one, &scale1, &n, &n,
                              b, &ldb, &info1);
                }
                ui[0] = b[kl * ldb + kl] / delta1 / sqtwo;
                m1[0] = a[kl * lda + kl] / e[kl * lde + kl];
                m2[0] = (delta1 / e[kl * lde + kl]) * sqtwo;

            } else {
                sg03bx("C", "N", &a[kl * lda + kl], lda, &e[kl * lde + kl],
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

                SLC_DGEMM("T", "T", &nkh, &kb, &kb, &mone, &a[kl + (kh + 1) * lda],
                         &lda, ui, &int2, &one, &dwork[uiipt], &ldws);

                SLC_DGEMM("T", "N", &kb, &kb, &kb, &one, ui, &int2, m1, &int2,
                         &zero, tm, &int2);

                SLC_DGEMM("T", "N", &nkh, &kb, &kb, &mone, &e[kl + (kh + 1) * lde],
                         &lde, tm, &int2, &one, &dwork[uiipt], &ldws);

                SLC_DLASET("A", &kb, &kb, &zero, &one, tm, &int2);

                sg03bw("N", nkh, kb, &a[(kh + 1) * lda + kh + 1], lda,
                      tm, int2, &e[(kh + 1) * lde + kh + 1], lde, m1, int2,
                      &dwork[uiipt], ldws, &scale1, info);

                if (scale1 != one) {
                    *scale = scale1 * (*scale);
                    SLC_DLASCL("Upper", &int0, &int0, &one, &scale1, &n, &n,
                              b, &ldb, &info1);
                    i32 int4 = 4;
                    SLC_DSCAL(&int4, &scale1, ui, &int1);
                }

                SLC_DLACPY("A", &nkh, &kb, &dwork[uiipt], &ldws,
                          &dwork[wpt], &ldws);

                SLC_DTRMM("L", "U", "T", "N", &nkh, &kb, &one,
                         &e[(kh + 1) * lde + kh + 1], &lde, &dwork[wpt], &ldws);

                SLC_DGEMM("T", "T", &nkh, &kb, &kb, &one, &e[kl + (kh + 1) * lde],
                         &lde, ui, &int2, &one, &dwork[wpt], &ldws);

                SLC_DCOPY(&nkh, &b[(kh + 1) * ldb + kl], &ldb, &dwork[ypt], &int1);
                if (kh > kl) {
                    SLC_DCOPY(&nkh, &b[(kh + 1) * ldb + kh], &ldb,
                             &dwork[ypt + ldws], &int1);
                }

                SLC_DGEMM("N", "T", &nkh, &kb, &kb, &mone, &dwork[wpt],
                         &ldws, m2, &int2, &one, &dwork[ypt], &ldws);

                l = ypt;
                for (j = 0; j < kb; j++) {
                    for (i = 0; i < nkh; i++) {
                        x = b[(kh + i + 1) + (kh + i + 1) * ldb];
                        t = dwork[l + i];
                        SLC_DLARTG(&x, &t, &c, &s, &r);
                        b[(kh + i + 1) + (kh + i + 1) * ldb] = r;
                        if (i < nkh - 1) {
                            i32 nmihm1 = nkh - i - 1;
                            SLC_DROT(&nmihm1, &b[(kh + i + 2) * ldb + kh + i + 1],
                                    &ldb, &dwork[l + i + 1], &int1, &c, &s);
                        }
                    }
                    l += ldws;
                }

                for (i = kh + 1; i < n; i++) {
                    if (b[i + i * ldb] < zero) {
                        i32 nmi = n - i;
                        SLC_DSCAL(&nmi, &mone, &b[i + i * ldb], &ldb);
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

            kh++;
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
            kl1 = kl - 1;

            if (kb == 1) {
                delta1 = -a[kl * lda + kl];
                if (delta1 <= zero) {
                    *info = 3;
                    return;
                }
                delta1 = sqrt(delta1) * sqrt(e[kl * lde + kl]);
                t = (b[kl * ldb + kl] * smlnum) / sqtwo;
                if (t > delta1) {
                    scale1 = delta1 / t;
                    *scale = scale1 * (*scale);
                    SLC_DLASCL("Upper", &int0, &int0, &one, &scale1, &n, &n,
                              b, &ldb, &info1);
                }
                ui[0] = b[kl * ldb + kl] / delta1 / sqtwo;
                m1[0] = a[kl * lda + kl] / e[kl * lde + kl];
                m2[0] = (delta1 / e[kl * lde + kl]) * sqtwo;

            } else {
                sg03bx("C", "T", &a[kl * lda + kl], lda, &e[kl * lde + kl],
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

                SLC_DGEMM("N", "N", &kl, &kb, &kb, &mone, &a[kl * lda],
                         &lda, ui, &int2, &one, &dwork[uiipt], &ldws);

                SLC_DGEMM("N", "T", &kb, &kb, &kb, &one, ui, &int2, m1, &int2,
                         &zero, tm, &int2);

                SLC_DGEMM("N", "N", &kl, &kb, &kb, &mone, &e[kl * lde],
                         &lde, tm, &int2, &one, &dwork[uiipt], &ldws);

                SLC_DLASET("A", &kb, &kb, &zero, &one, tm, &int2);

                sg03bw("T", kl, kb, a, lda, tm, int2, e, lde, m1, int2,
                      &dwork[uiipt], ldws, &scale1, info);

                if (scale1 != one) {
                    *scale = scale1 * (*scale);
                    SLC_DLASCL("Upper", &int0, &int0, &one, &scale1, &n, &n,
                              b, &ldb, &info1);
                    i32 int4 = 4;
                    SLC_DSCAL(&int4, &scale1, ui, &int1);
                }

                SLC_DLACPY("A", &kl, &kb, &dwork[uiipt], &ldws,
                          &dwork[wpt], &ldws);

                SLC_DTRMM("L", "U", "N", "N", &kl, &kb, &one, e, &lde,
                         &dwork[wpt], &ldws);

                SLC_DGEMM("N", "N", &kl, &kb, &kb, &one, &e[kl * lde],
                         &lde, ui, &int2, &one, &dwork[wpt], &ldws);

                SLC_DLACPY("A", &kl, &kb, &b[kl * ldb], &ldb, &dwork[ypt],
                          &ldws);

                SLC_DGEMM("N", "N", &kl, &kb, &kb, &mone, &dwork[wpt],
                         &ldws, m2, &int2, &one, &dwork[ypt], &ldws);

                l = ypt;
                for (j = 0; j < kb; j++) {
                    for (i = kl - 1; i >= 0; i--) {
                        x = b[i + i * ldb];
                        t = dwork[l + i];
                        SLC_DLARTG(&x, &t, &c, &s, &r);
                        b[i + i * ldb] = r;
                        if (i > 0) {
                            SLC_DROT(&i, &b[i * ldb], &int1, &dwork[l], &int1, &c, &s);
                        }
                    }
                    l += ldws;
                }

                for (i = 0; i < kl; i++) {
                    if (b[i + i * ldb] < zero) {
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
