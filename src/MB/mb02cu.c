/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>
#include <stdbool.h>

void mb02cu(const char* typeg, i32 k, i32 p, i32 q, i32 nb,
            f64* a1, i32 lda1, f64* a2, i32 lda2,
            f64* b, i32 ldb, i32* rnk, i32* ipvt, f64* cs,
            f64 tol, f64* dwork, i32 ldwork, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    char typeg_u = (char)toupper((unsigned char)typeg[0]);
    bool lrdef = (typeg_u == 'D');
    bool lcol = (typeg_u == 'C');

    *info = 0;
    i32 col2 = p - k;

    i32 wrkmin;
    if (lrdef) {
        wrkmin = (4 * k > 1) ? 4 * k : 1;
    } else {
        i32 nbk = nb * k;
        wrkmin = (nbk > k) ? nbk : k;
        wrkmin = (wrkmin > 1) ? wrkmin : 1;
    }

    if (!lcol && !lrdef && typeg_u != 'R') {
        *info = -1;
    } else if (k < 0) {
        *info = -2;
    } else if (p < k) {
        *info = -3;
    } else if (q < 0 || (lrdef && q < k)) {
        *info = -4;
    } else if (lda1 < (k > 1 ? k : 1)) {
        *info = -7;
    } else if ((p == k && lda2 < 1) ||
               (p > k && (lrdef || lcol) && lda2 < (k > 1 ? k : 1)) ||
               (p > k && !lrdef && !lcol && lda2 < (p - k))) {
        *info = -9;
    } else if ((q == 0 && ldb < 1) ||
               (q > 0 && (lrdef || lcol) && ldb < (k > 1 ? k : 1)) ||
               (q > 0 && !lrdef && !lcol && ldb < q)) {
        *info = -11;
    } else if (ldwork < wrkmin) {
        dwork[0] = (f64)wrkmin;
        *info = -17;
    }

    if (*info != 0) {
        return;
    }

    if (k == 0 || (!lrdef && q == 0 && p == k)) {
        if (lrdef) *rnk = 0;
        return;
    }

    f64 tolz = sqrt(SLC_DLAMCH("Epsilon"));

    i32 int1 = 1;
    i32 ierr = 0;

    if (lrdef) {
        i32 pst2;
        if (col2 == 0) {
            pst2 = 2 * k;
        } else {
            pst2 = 4 * k;
        }

        *rnk = 0;
        i32 phv = 3 * k;

        for (i32 i = 0; i < k; i++) {
            ipvt[i] = i + 1;
            dwork[i] = SLC_DNRM2(&k, &a1[i], &lda1);
        }

        for (i32 i = 0; i < k; i++) {
            f64 nrm_a2 = SLC_DNRM2(&col2, &a2[i], &lda2);
            dwork[i] = SLC_DLAPY2(&dwork[i], &nrm_a2);
            dwork[k + i] = dwork[i];
        }

        for (i32 i = 0; i < k; i++) {
            dwork[2 * k + i] = SLC_DNRM2(&q, &b[i], &ldb);
        }

        for (i32 i = 0; i < k; i++) {
            f64 alpha_abs = fabs(dwork[i]);
            f64 beta_abs = fabs(dwork[2 * k + i]);
            f64 dmax = copysign(sqrt(fabs(alpha_abs - beta_abs)) * sqrt(alpha_abs + beta_abs),
                                alpha_abs - beta_abs);
            i32 imax = i;

            for (i32 j = 1; j <= k - i - 1; j++) {
                f64 a_abs = fabs(dwork[i + j]);
                f64 b_abs = fabs(dwork[2 * k + i + j]);
                f64 temp = copysign(sqrt(fabs(a_abs - b_abs)) * sqrt(a_abs + b_abs),
                                    a_abs - b_abs);
                if (temp > dmax) {
                    imax = i + j;
                    dmax = temp;
                }
            }

            if (dmax > tol) {
                i32 pvt = imax;
                if (pvt != i) {
                    SLC_DSWAP(&k, &a1[pvt], &lda1, &a1[i], &lda1);
                    SLC_DSWAP(&col2, &a2[pvt], &lda2, &a2[i], &lda2);
                    SLC_DSWAP(&q, &b[pvt], &ldb, &b[i], &ldb);
                    i32 itemp = ipvt[pvt];
                    ipvt[pvt] = ipvt[i];
                    ipvt[i] = itemp;
                    dwork[pvt] = dwork[i];
                    dwork[k + pvt] = dwork[k + i];
                    dwork[2 * k + pvt] = dwork[2 * k + i];
                }

                f64 alpha2 = zero, tau2 = zero, alpha, tau1;

                if (col2 > 1) {
                    SLC_DLARFG(&col2, &a2[i], &a2[i + lda2], &lda2, &tau2);
                    alpha2 = a2[i];
                    if (k > i + 1) {
                        a2[i] = one;
                        i32 rows = k - i - 1;
                        SLC_DLARF("Right", &rows, &col2, &a2[i], &lda2, &tau2,
                                  &a2[i + 1], &lda2, &dwork[phv]);
                    }
                    a2[i] = tau2;
                } else if (col2 > 0) {
                    alpha2 = a2[i];
                    a2[i] = zero;
                }

                if (k > i + 1) {
                    i32 len = k - i;
                    SLC_DLARFG(&len, &a1[i + i * lda1], &a1[i + (i + 1) * lda1], &lda1, &tau1);
                    alpha = a1[i + i * lda1];
                    a1[i + i * lda1] = one;
                    i32 rows = k - i - 1;
                    SLC_DLARF("Right", &rows, &len, &a1[i + i * lda1], &lda1, &tau1,
                              &a1[i + 1 + i * lda1], &lda1, &dwork[phv]);
                    cs[pst2 + i] = tau1;
                } else {
                    alpha = a1[i + i * lda1];
                }

                if (col2 > 0) {
                    f64 temp = alpha;
                    f64 c, s;
                    SLC_DLARTG(&temp, &alpha2, &c, &s, &alpha);
                    if (k > i + 1) {
                        i32 rows = k - i - 1;
                        SLC_DROT(&rows, &a1[i + 1 + i * lda1], &int1,
                                 &a2[i + 1], &int1, &c, &s);
                    }
                    cs[2 * k + (i * 2)] = c;
                    cs[2 * k + (i * 2) + 1] = s;
                }
                a1[i + i * lda1] = alpha;

                f64 beta_val = zero;
                if (q > 1) {
                    SLC_DLARFG(&q, &b[i], &b[i + ldb], &ldb, &tau2);
                    beta_val = b[i];
                    if (k > i + 1) {
                        b[i] = one;
                        i32 rows = k - i - 1;
                        SLC_DLARF("Right", &rows, &q, &b[i], &ldb, &tau2,
                                  &b[i + 1], &ldb, &dwork[phv]);
                    }
                    b[i] = tau2;
                } else if (q > 0) {
                    beta_val = b[i];
                    b[i] = zero;
                }

                f64 c, s;
                ma02fd(&a1[i + i * lda1], beta_val, &c, &s, &ierr);
                if (ierr != 0) {
                    *info = 1;
                    return;
                }

                if (k > i + 1) {
                    i32 len = k - i - 1;
                    f64 inv_c = one / c;
                    f64 neg_s_c = -s / c;
                    f64 neg_s = -s;
                    SLC_DSCAL(&len, &inv_c, &a1[i + 1 + i * lda1], &int1);
                    SLC_DAXPY(&len, &neg_s_c, &b[i + 1], &int1, &a1[i + 1 + i * lda1], &int1);
                    SLC_DSCAL(&len, &c, &b[i + 1], &int1);
                    SLC_DAXPY(&len, &neg_s, &a1[i + 1 + i * lda1], &int1, &b[i + 1], &int1);
                }
                cs[(i * 2)] = c;
                cs[(i * 2) + 1] = s;

                for (i32 j = i + 1; j < k; j++) {
                    f64 temp = fabs(a1[j + i * lda1]) / dwork[j];
                    temp = fmax((one + temp) * (one - temp), zero);
                    f64 ratio = dwork[j] / dwork[k + j];
                    f64 temp2 = temp * ratio * ratio;
                    if (temp2 <= tolz) {
                        i32 rem = k - i - 1;
                        f64 n1 = SLC_DNRM2(&rem, &a1[j + (i + 1) * lda1], &lda1);
                        f64 n2 = SLC_DNRM2(&col2, &a2[j], &lda2);
                        dwork[j] = SLC_DLAPY2(&n1, &n2);
                        dwork[k + j] = dwork[j];
                        dwork[2 * k + j] = SLC_DNRM2(&q, &b[j], &ldb);
                    } else {
                        if (temp >= zero) {
                            dwork[j] = dwork[j] * sqrt(temp);
                        } else {
                            dwork[j] = -dwork[j] * sqrt(-temp);
                        }
                    }
                }

                (*rnk)++;
            } else if (fabs(dmax) < tol) {
                for (i32 j = i; j < k; j++) {
                    dwork[j] = SLC_DNRM2(&q, &b[j], &ldb);
                    dwork[k + j] = dwork[j];
                }

                i32 len = q;
                i32 pos = 0;

                for (i32 j = i; j < k; j++) {
                    i32 rem = k - j;
                    i32 pvt = (j - 1) + SLC_IDAMAX(&rem, &dwork[j], &int1);

                    if (pvt != j) {
                        SLC_DSWAP(&k, &a1[pvt], &lda1, &a1[j], &lda1);
                        SLC_DSWAP(&col2, &a2[pvt], &lda2, &a2[j], &lda2);
                        SLC_DSWAP(&q, &b[pvt], &ldb, &b[j], &ldb);
                        i32 itemp = ipvt[pvt];
                        ipvt[pvt] = ipvt[j];
                        ipvt[j] = itemp;
                        dwork[pvt] = dwork[j];
                        dwork[k + pvt] = dwork[k + j];
                    }

                    f64 alpha2 = zero, tau2 = zero;
                    if (col2 > 1) {
                        SLC_DLARFG(&col2, &a2[j], &a2[j + lda2], &lda2, &tau2);
                        alpha2 = a2[j];
                        if (k > j + 1) {
                            a2[j] = one;
                            i32 rows = k - j - 1;
                            SLC_DLARF("Right", &rows, &col2, &a2[j], &lda2,
                                      &tau2, &a2[j + 1], &lda2, &dwork[phv]);
                        }
                        a2[j] = tau2;
                    } else if (col2 > 0) {
                        alpha2 = a2[j];
                        a2[j] = zero;
                    }

                    f64 alpha, tau1;
                    if (k > j + 1) {
                        i32 llen = k - j;
                        SLC_DLARFG(&llen, &a1[j + j * lda1], &a1[j + (j + 1) * lda1],
                                   &lda1, &tau1);
                        alpha = a1[j + j * lda1];
                        a1[j + j * lda1] = one;
                        i32 rows = k - j - 1;
                        SLC_DLARF("Right", &rows, &llen, &a1[j + j * lda1], &lda1,
                                  &tau1, &a1[j + 1 + j * lda1], &lda1, &dwork[phv]);
                        cs[pst2 + j] = tau1;
                    } else {
                        alpha = a1[j + j * lda1];
                    }

                    if (col2 > 0) {
                        f64 temp = alpha;
                        f64 c, s;
                        SLC_DLARTG(&temp, &alpha2, &c, &s, &alpha);
                        if (k > j + 1) {
                            i32 rows = k - j - 1;
                            SLC_DROT(&rows, &a1[j + 1 + j * lda1], &int1,
                                     &a2[j + 1], &int1, &c, &s);
                        }
                        cs[2 * k + (j * 2)] = c;
                        cs[2 * k + (j * 2) + 1] = s;
                    }
                    a1[j + j * lda1] = alpha;

                    f64 beta_val;
                    if (len > 1) {
                        SLC_DLARFG(&len, &b[j + pos * ldb], &b[j + (pos + 1) * ldb],
                                   &ldb, &tau2);
                        beta_val = b[j + pos * ldb];
                        if (k > j + 1) {
                            b[j + pos * ldb] = one;
                            i32 rows = k - j - 1;
                            SLC_DLARF("Right", &rows, &len, &b[j + pos * ldb], &ldb,
                                      &tau2, &b[j + 1 + pos * ldb], &ldb, &dwork[phv]);
                        }
                        b[j + pos * ldb] = beta_val;
                        cs[(j * 2)] = tau2;
                    }

                    for (i32 jj = j + 1; jj < k; jj++) {
                        if (dwork[jj] != zero) {
                            f64 temp = fabs(b[jj + pos * ldb]) / dwork[jj];
                            temp = fmax((one + temp) * (one - temp), zero);
                            f64 ratio = dwork[jj] / dwork[k + jj];
                            f64 temp2 = temp * ratio * ratio;
                            if (temp2 <= tolz) {
                                i32 rem = len - 1;
                                dwork[jj] = SLC_DNRM2(&rem, &b[jj + (pos + 1) * ldb], &ldb);
                                dwork[k + jj] = dwork[jj];
                            } else {
                                if (temp >= zero) {
                                    dwork[jj] = dwork[jj] * sqrt(temp);
                                } else {
                                    dwork[jj] = -dwork[jj] * sqrt(-temp);
                                }
                            }
                        }
                    }

                    len--;
                    pos++;
                }

                return;
            } else {
                *info = 1;
                return;
            }
        }
    } else if (lcol) {
        i32 pst2;
        i32 nbl;
        i32 start_j = 0;

        if (col2 > 0) {
            nbl = (col2 < nb) ? col2 : nb;
            if (nbl > 0) {
                for (i32 i = 0; i < k - nbl + 1; i += nbl) {
                    i32 ib = (k - i < nbl) ? (k - i) : nbl;
                    SLC_DGELQ2(&ib, &col2, &a2[i], &lda2, &cs[4 * k + i], dwork, &ierr);
                    if (i + ib < k) {
                        SLC_DLARFT("Forward", "Rowwise", &col2, &ib, &a2[i], &lda2,
                                   &cs[4 * k + i], dwork, &k);
                        i32 rows = k - i - ib;
                        SLC_DLARFB("Right", "No Transpose", "Forward", "Rowwise",
                                   &rows, &col2, &ib, &a2[i], &lda2, dwork, &k,
                                   &a2[i + ib], &lda2, &dwork[ib], &k);
                    }

                    for (i32 j = i; j < i + ib; j++) {
                        f64 alpha2, tau2;
                        if (col2 > 1) {
                            i32 llen = (j - i + 1 < col2) ? (j - i + 1) : col2;
                            SLC_DLARFG(&llen, &a2[j], &a2[j + lda2], &lda2, &tau2);
                            alpha2 = a2[j];
                            if (k > j + 1) {
                                a2[j] = one;
                                i32 rows = k - j - 1;
                                SLC_DLARF("Right", &rows, &llen, &a2[j], &lda2,
                                          &tau2, &a2[j + 1], &lda2, dwork);
                            }
                            a2[j] = tau2;
                        } else {
                            alpha2 = a2[j];
                            a2[j] = zero;
                        }
                        f64 alpha = a1[j + j * lda1];
                        f64 c, s;
                        SLC_DLARTG(&alpha, &alpha2, &c, &s, &a1[j + j * lda1]);
                        if (k > j + 1) {
                            i32 rows = k - j - 1;
                            SLC_DROT(&rows, &a1[j + 1 + j * lda1], &int1,
                                     &a2[j + 1], &int1, &c, &s);
                        }
                        cs[2 * k + (j * 2)] = c;
                        cs[2 * k + (j * 2) + 1] = s;
                    }
                    start_j = i + ib;
                }
            }

            for (i32 j = start_j; j < k; j++) {
                f64 alpha2, tau2;
                if (col2 > 1) {
                    SLC_DLARFG(&col2, &a2[j], &a2[j + lda2], &lda2, &tau2);
                    alpha2 = a2[j];
                    if (k > j + 1) {
                        a2[j] = one;
                        i32 rows = k - j - 1;
                        SLC_DLARF("Right", &rows, &col2, &a2[j], &lda2,
                                  &tau2, &a2[j + 1], &lda2, dwork);
                    }
                    a2[j] = tau2;
                } else {
                    alpha2 = a2[j];
                    a2[j] = zero;
                }
                f64 alpha = a1[j + j * lda1];
                f64 c, s;
                SLC_DLARTG(&alpha, &alpha2, &c, &s, &a1[j + j * lda1]);
                if (k > j + 1) {
                    i32 rows = k - j - 1;
                    SLC_DROT(&rows, &a1[j + 1 + j * lda1], &int1,
                             &a2[j + 1], &int1, &c, &s);
                }
                cs[2 * k + (j * 2)] = c;
                cs[2 * k + (j * 2) + 1] = s;
            }

            pst2 = 5 * k;
        } else {
            pst2 = 2 * k;
        }

        nbl = (nb < q) ? nb : q;
        start_j = 0;
        if (nbl > 0) {
            for (i32 i = 0; i < k - nbl + 1; i += nbl) {
                i32 ib = (k - i < nbl) ? (k - i) : nbl;
                SLC_DGELQ2(&ib, &q, &b[i], &ldb, &cs[pst2 + i], dwork, &ierr);
                if (i + ib < k) {
                    SLC_DLARFT("Forward", "Rowwise", &q, &ib, &b[i], &ldb,
                               &cs[pst2 + i], dwork, &k);
                    i32 rows = k - i - ib;
                    SLC_DLARFB("Right", "No Transpose", "Forward", "Rowwise",
                               &rows, &q, &ib, &b[i], &ldb, dwork, &k,
                               &b[i + ib], &ldb, &dwork[ib], &k);
                }

                for (i32 j = i; j < i + ib; j++) {
                    f64 alpha2, tau2;
                    if (q > 1) {
                        i32 llen = j - i + 1;
                        SLC_DLARFG(&llen, &b[j], &b[j + ldb], &ldb, &tau2);
                        alpha2 = b[j];
                        if (k > j + 1) {
                            b[j] = one;
                            i32 rows = k - j - 1;
                            SLC_DLARF("Right", &rows, &llen, &b[j], &ldb,
                                      &tau2, &b[j + 1], &ldb, dwork);
                        }
                        b[j] = tau2;
                    } else {
                        alpha2 = b[j];
                        b[j] = zero;
                    }

                    f64 c, s;
                    ma02fd(&a1[j + j * lda1], alpha2, &c, &s, &ierr);
                    if (ierr != 0) {
                        *info = 1;
                        return;
                    }

                    if (k > j + 1) {
                        i32 len = k - j - 1;
                        f64 inv_c = one / c;
                        f64 neg_s_c = -s / c;
                        f64 neg_s = -s;
                        SLC_DSCAL(&len, &inv_c, &a1[j + 1 + j * lda1], &int1);
                        SLC_DAXPY(&len, &neg_s_c, &b[j + 1], &int1, &a1[j + 1 + j * lda1], &int1);
                        SLC_DSCAL(&len, &c, &b[j + 1], &int1);
                        SLC_DAXPY(&len, &neg_s, &a1[j + 1 + j * lda1], &int1, &b[j + 1], &int1);
                    }
                    cs[(j * 2)] = c;
                    cs[(j * 2) + 1] = s;
                }
                start_j = i + ib;
            }
        }

        for (i32 j = start_j; j < k; j++) {
            f64 alpha2, tau2;
            if (q > 1) {
                SLC_DLARFG(&q, &b[j], &b[j + ldb], &ldb, &tau2);
                alpha2 = b[j];
                if (k > j + 1) {
                    b[j] = one;
                    i32 rows = k - j - 1;
                    SLC_DLARF("Right", &rows, &q, &b[j], &ldb,
                              &tau2, &b[j + 1], &ldb, dwork);
                }
                b[j] = tau2;
            } else if (q > 0) {
                alpha2 = b[j];
                b[j] = zero;
            }
            if (q > 0) {
                f64 c, s;
                ma02fd(&a1[j + j * lda1], alpha2, &c, &s, &ierr);
                if (ierr != 0) {
                    *info = 1;
                    return;
                }

                if (k > j + 1) {
                    i32 len = k - j - 1;
                    f64 inv_c = one / c;
                    f64 neg_s_c = -s / c;
                    f64 neg_s = -s;
                    SLC_DSCAL(&len, &inv_c, &a1[j + 1 + j * lda1], &int1);
                    SLC_DAXPY(&len, &neg_s_c, &b[j + 1], &int1, &a1[j + 1 + j * lda1], &int1);
                    SLC_DSCAL(&len, &c, &b[j + 1], &int1);
                    SLC_DAXPY(&len, &neg_s, &a1[j + 1 + j * lda1], &int1, &b[j + 1], &int1);
                }
                cs[(j * 2)] = c;
                cs[(j * 2) + 1] = s;
            }
        }
    } else {
        i32 pst2;
        i32 nbl;
        i32 i = 0;

        if (col2 > 0) {
            nbl = (nb < col2) ? nb : col2;
            if (nbl > 0) {
                for (i = 0; i < k - nbl + 1; i += nbl) {
                    i32 ib = (k - i < nbl) ? (k - i) : nbl;
                    SLC_DGEQR2(&col2, &ib, &a2[i * lda2], &lda2, &cs[4 * k + i], dwork, &ierr);
                    if (i + ib < k) {
                        SLC_DLARFT("Forward", "Columnwise", &col2, &ib, &a2[i * lda2], &lda2,
                                   &cs[4 * k + i], dwork, &k);
                        i32 cols = k - i - ib;
                        SLC_DLARFB("Left", "Transpose", "Forward", "Columnwise",
                                   &col2, &cols, &ib, &a2[i * lda2], &lda2, dwork, &k,
                                   &a2[(i + ib) * lda2], &lda2, &dwork[ib], &k);
                    }

                    for (i32 j = i; j < i + ib; j++) {
                        f64 alpha2, tau2;
                        if (col2 > 1) {
                            i32 llen = (j - i + 1 < col2) ? (j - i + 1) : col2;
                            SLC_DLARFG(&llen, &a2[j * lda2], &a2[1 + j * lda2], &int1, &tau2);
                            alpha2 = a2[j * lda2];
                            if (k > j + 1) {
                                a2[j * lda2] = one;
                                i32 cols = k - j - 1;
                                SLC_DLARF("Left", &llen, &cols, &a2[j * lda2], &int1,
                                          &tau2, &a2[(j + 1) * lda2], &lda2, dwork);
                            }
                            a2[j * lda2] = tau2;
                        } else {
                            alpha2 = a2[j * lda2];
                            a2[j * lda2] = zero;
                        }
                        f64 alpha = a1[j + j * lda1];
                        f64 c, s;
                        SLC_DLARTG(&alpha, &alpha2, &c, &s, &a1[j + j * lda1]);
                        if (k > j + 1) {
                            i32 cols = k - j - 1;
                            SLC_DROT(&cols, &a1[j + (j + 1) * lda1], &lda1,
                                     &a2[(j + 1) * lda2], &lda2, &c, &s);
                        }
                        cs[2 * k + (j * 2)] = c;
                        cs[2 * k + (j * 2) + 1] = s;
                    }
                }
            }

            for (i32 j = i; j < k; j++) {
                f64 alpha2, tau2;
                if (col2 > 1) {
                    SLC_DLARFG(&col2, &a2[j * lda2], &a2[1 + j * lda2], &int1, &tau2);
                    alpha2 = a2[j * lda2];
                    if (k > j + 1) {
                        a2[j * lda2] = one;
                        i32 cols = k - j - 1;
                        SLC_DLARF("Left", &col2, &cols, &a2[j * lda2], &int1,
                                  &tau2, &a2[(j + 1) * lda2], &lda2, dwork);
                    }
                    a2[j * lda2] = tau2;
                } else {
                    alpha2 = a2[j * lda2];
                    a2[j * lda2] = zero;
                }
                f64 alpha = a1[j + j * lda1];
                f64 c, s;
                SLC_DLARTG(&alpha, &alpha2, &c, &s, &a1[j + j * lda1]);
                if (k > j + 1) {
                    i32 cols = k - j - 1;
                    SLC_DROT(&cols, &a1[j + (j + 1) * lda1], &lda1,
                             &a2[(j + 1) * lda2], &lda2, &c, &s);
                }
                cs[2 * k + (j * 2)] = c;
                cs[2 * k + (j * 2) + 1] = s;
            }

            pst2 = 5 * k;
        } else {
            pst2 = 2 * k;
        }

        i = 0;
        nbl = (nb < q) ? nb : q;
        if (nbl > 0) {
            for (; i < k - nbl + 1; i += nbl) {
                i32 ib = (k - i < nbl) ? (k - i) : nbl;
                SLC_DGEQR2(&q, &ib, &b[i * ldb], &ldb, &cs[pst2 + i], dwork, &ierr);
                if (i + ib < k) {
                    SLC_DLARFT("Forward", "Columnwise", &q, &ib, &b[i * ldb], &ldb,
                               &cs[pst2 + i], dwork, &k);
                    i32 cols = k - i - ib;
                    SLC_DLARFB("Left", "Transpose", "Forward", "Columnwise",
                               &q, &cols, &ib, &b[i * ldb], &ldb, dwork, &k,
                               &b[(i + ib) * ldb], &ldb, &dwork[ib], &k);
                }

                for (i32 j = i; j < i + ib; j++) {
                    f64 alpha2, tau2;
                    if (q > 1) {
                        i32 llen = j - i + 1;
                        SLC_DLARFG(&llen, &b[j * ldb], &b[1 + j * ldb], &int1, &tau2);
                        alpha2 = b[j * ldb];
                        if (k > j + 1) {
                            b[j * ldb] = one;
                            i32 cols = k - j - 1;
                            SLC_DLARF("Left", &llen, &cols, &b[j * ldb], &int1,
                                      &tau2, &b[(j + 1) * ldb], &ldb, dwork);
                        }
                        b[j * ldb] = tau2;
                    } else {
                        alpha2 = b[j * ldb];
                        b[j * ldb] = zero;
                    }

                    f64 c, s;
                    ma02fd(&a1[j + j * lda1], alpha2, &c, &s, &ierr);
                    if (ierr != 0) {
                        *info = 1;
                        return;
                    }

                    if (k > j + 1) {
                        i32 len = k - j - 1;
                        f64 inv_c = one / c;
                        f64 neg_s_c = -s / c;
                        f64 neg_s = -s;
                        SLC_DSCAL(&len, &inv_c, &a1[j + (j + 1) * lda1], &lda1);
                        SLC_DAXPY(&len, &neg_s_c, &b[(j + 1) * ldb], &ldb,
                                  &a1[j + (j + 1) * lda1], &lda1);
                        SLC_DSCAL(&len, &c, &b[(j + 1) * ldb], &ldb);
                        SLC_DAXPY(&len, &neg_s, &a1[j + (j + 1) * lda1], &lda1,
                                  &b[(j + 1) * ldb], &ldb);
                    }
                    cs[(j * 2)] = c;
                    cs[(j * 2) + 1] = s;
                }
            }
        } else {
            i = 0;
        }

        for (i32 j = i; j < k; j++) {
            f64 alpha2, tau2;
            if (q > 1) {
                SLC_DLARFG(&q, &b[j * ldb], &b[1 + j * ldb], &int1, &tau2);
                alpha2 = b[j * ldb];
                if (k > j + 1) {
                    b[j * ldb] = one;
                    i32 cols = k - j - 1;
                    SLC_DLARF("Left", &q, &cols, &b[j * ldb], &int1,
                              &tau2, &b[(j + 1) * ldb], &ldb, dwork);
                }
                b[j * ldb] = tau2;
            } else if (q > 0) {
                alpha2 = b[j * ldb];
                b[j * ldb] = zero;
            }
            if (q > 0) {
                f64 c, s;
                ma02fd(&a1[j + j * lda1], alpha2, &c, &s, &ierr);
                if (ierr != 0) {
                    *info = 1;
                    return;
                }

                if (k > j + 1) {
                    i32 len = k - j - 1;
                    f64 inv_c = one / c;
                    f64 neg_s_c = -s / c;
                    f64 neg_s = -s;
                    SLC_DSCAL(&len, &inv_c, &a1[j + (j + 1) * lda1], &lda1);
                    SLC_DAXPY(&len, &neg_s_c, &b[(j + 1) * ldb], &ldb,
                              &a1[j + (j + 1) * lda1], &lda1);
                    SLC_DSCAL(&len, &c, &b[(j + 1) * ldb], &ldb);
                    SLC_DAXPY(&len, &neg_s, &a1[j + (j + 1) * lda1], &lda1,
                              &b[(j + 1) * ldb], &ldb);
                }
                cs[(j * 2)] = c;
                cs[(j * 2) + 1] = s;
            }
        }
    }
}
