/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

static inline i32 max_i32(i32 a, i32 b) {
    return a > b ? a : b;
}

static inline i32 min_i32(i32 a, i32 b) {
    return a < b ? a : b;
}

void sb01dd(
    i32 n,
    i32 m,
    i32 indcon,
    f64* a,
    i32 lda,
    f64* b,
    i32 ldb,
    const i32* nblk,
    f64* wr,
    f64* wi,
    f64* z,
    i32 ldz,
    const f64* y,
    i32* count,
    f64* g,
    i32 ldg,
    f64 tol,
    i32* iwork,
    f64* dwork,
    i32 ldwork,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    bool complx;
    i32 i, ia, indcn1, indcn2, indcrt, ip, irmx, iwrk;
    i32 k, kk, kmr, l, lp1, m1, maxwrk, mi, mp1, mr;
    i32 mr1, nblkcr, nc, ni, nj, np1, nr, nr1, rank;
    f64 p_val, q_val, r_val, s, svlmxa, svlmxb, toldef;
    f64 sval[3];

    sl_int n_sl = n, m_sl = m, lda_sl = lda, ldb_sl = ldb, ldz_sl = ldz, ldg_sl = ldg;
    sl_int int1 = 1;
    i32 info_sub;

    *info = 0;
    nr = 0;
    iwrk = max_i32(m*n, m*m + 2*n + 4*m + 1);

    if (n < 0) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (indcon < 0 || indcon > n) {
        *info = -3;
    } else if (lda < max_i32(1, n)) {
        *info = -5;
    } else if (ldb < max_i32(1, n)) {
        *info = -7;
    } else if (ldz < max_i32(1, n)) {
        *info = -12;
    } else if (ldg < max_i32(1, m)) {
        *info = -16;
    } else if (ldwork < iwrk) {
        *info = -20;
    }

    if (*info != 0) {
        return;
    }

    for (i = 0; i < min_i32(indcon, n); i++) {
        nr += nblk[i];
        if (i > 0 && nblk[i-1] < nblk[i]) {
            *info = -8;
        }
    }

    if (nr != n) {
        *info = -8;
    }

    if (*info != 0) {
        return;
    }

    if (min_i32(m, min_i32(n, indcon)) == 0) {
        *count = 0;
        dwork[0] = ONE;
        return;
    }

    maxwrk = iwrk;
    toldef = tol;
    if (toldef <= ZERO) {
        toldef = (f64)(n * n) * SLC_DLAMCH("Epsilon");
    }

    irmx = 2*n + 1;
    iwrk = irmx + m*m;
    m1 = nblk[0];
    *count = 1;
    indcrt = indcon;
    nblkcr = nblk[indcrt - 1];

    nr = m1;
    nc = 1;
    sl_int nr_sl = m1, nc_sl = 1;
    sl_int m_blas = m;
    svlmxb = SLC_DLANGE("Frobenius", &nr_sl, &m_blas, b, &ldb_sl, dwork);
    svlmxa = ZERO;

    for (i = 0; i < indcrt - 1; i++) {
        nr += nblk[i + 1];
        nr_sl = nr;
        nc_sl = nblk[i];
        f64 norm = SLC_DLANGE("Frobenius", &nr_sl, &nc_sl, &a[nc - 1], &lda_sl, dwork);
        svlmxa = sqrt(svlmxa * svlmxa + norm * norm);
        nc += nblk[i];
    }

    nr_sl = n;
    nc_sl = nblkcr;
    f64 norm = SLC_DLANGE("Frobenius", &nr_sl, &nc_sl, &a[(nc - 1) * lda], &lda_sl, dwork);
    svlmxa = sqrt(svlmxa * svlmxa + norm * norm);

    l = 1;
    mr = nblkcr;
    nr = n - mr + 1;

    while (indcrt > 1) {
        lp1 = l + m1;
        indcn1 = indcrt - 1;
        mr1 = nblk[indcn1 - 1];
        nr1 = nr - mr1;
        complx = (wi[l - 1] != ZERO);

        SLC_DCOPY(&mr, &y[*count - 1], &int1, &dwork[nr - 1], &int1);
        *count += mr;
        nc = 1;

        if (complx) {
            sl_int mr_sl = mr;
            SLC_DCOPY(&mr_sl, &y[*count - 1], &int1, &dwork[n + nr - 1], &int1);
            *count += mr;
            wi[l] = wi[l - 1] * wi[l];
            nc = 2;
        }

        for (ip = 1; ip <= indcrt; ip++) {
            if (ip != indcrt) {
                sl_int mr_sl = mr, mr1_sl = mr1, m_blas = m;
                SLC_DLACPY("Full", &mr_sl, &mr1_sl, &a[(nr - 1) + (nr1 - 1) * lda],
                           &lda_sl, &dwork[irmx - 1], &m_blas);

                if (ip == 1) {
                    mp1 = mr;
                    np1 = nr + mp1;
                } else {
                    mp1 = mr + 1;
                    np1 = nr + mp1;
                    sl_int mp1_sl = mp1;
                    s = SLC_DASUM(&mp1_sl, &dwork[nr - 1], &int1);
                    if (complx) {
                        s += SLC_DASUM(&mp1_sl, &dwork[n + nr - 1], &int1);
                    }
                    if (s != ZERO) {
                        f64 one_over_s = ONE / s;
                        SLC_DSCAL(&mp1_sl, &one_over_s, &dwork[nr - 1], &int1);
                        if (complx) {
                            SLC_DSCAL(&mp1_sl, &one_over_s, &dwork[n + nr - 1], &int1);
                            if (np1 <= n) {
                                dwork[n + np1 - 1] = dwork[n + np1 - 1] / s;
                            }
                        }
                    }
                }

                sl_int mr_copy = mr;
                SLC_DCOPY(&mr_copy, &dwork[nr - 1], &int1, &dwork[nr1 - 1], &int1);
                f64 wr_l = wr[l - 1];
                SLC_DSCAL(&mr_copy, &wr_l, &dwork[nr1 - 1], &int1);

                sl_int mp1_sl = mp1;
                f64 neg_one = -ONE;
                SLC_DGEMV("No transpose", &mr_copy, &mp1_sl, &neg_one,
                          &a[(nr - 1) + (nr - 1) * lda], &lda_sl,
                          &dwork[nr - 1], &int1, &ONE, &dwork[nr1 - 1], &int1);

                if (complx) {
                    f64 wi_lp1 = wi[l];
                    SLC_DAXPY(&mr_copy, &wi_lp1, &dwork[n + nr - 1], &int1,
                              &dwork[nr1 - 1], &int1);

                    SLC_DCOPY(&mr_copy, &dwork[nr - 1], &int1, &dwork[n + nr1 - 1], &int1);
                    f64 wr_lp1 = wr[l];
                    SLC_DAXPY(&mr_copy, &wr_lp1, &dwork[n + nr - 1], &int1,
                              &dwork[n + nr1 - 1], &int1);

                    SLC_DGEMV("No transpose", &mr_copy, &mp1_sl, &neg_one,
                              &a[(nr - 1) + (nr - 1) * lda], &lda_sl,
                              &dwork[n + nr - 1], &int1, &ONE, &dwork[n + nr1 - 1], &int1);

                    if (np1 <= n) {
                        f64 neg_dwork = -dwork[n + np1 - 1];
                        SLC_DAXPY(&mr_copy, &neg_dwork, &a[(nr - 1) + (np1 - 1) * lda], &int1,
                                  &dwork[n + nr1 - 1], &int1);
                    }
                }

                sl_int ldwork_sub = ldwork - iwrk + 1;
                mb02qd("FreeElements", "NoPermuting", mr, mr1, nc, toldef, svlmxa,
                       &dwork[irmx - 1], m, &dwork[nr1 - 1], n, &y[*count - 1],
                       iwork, &rank, sval, &dwork[iwrk - 1], ldwork_sub, &info_sub);

                maxwrk = max_i32(maxwrk, (i32)dwork[iwrk - 1] + iwrk - 1);
                if (rank < mr) {
                    goto exit_rank_deficient;
                }

                *count += (mr1 - mr) * nc;
                nj = nr1;
            } else {
                nj = nr;
            }

            ni = nr + mr - 1;
            if (ip == 1) {
                kmr = mr - 1;
            } else {
                kmr = mr;
                if (ip == 2) {
                    ni = ni + nblkcr;
                } else {
                    ni = ni + nblk[indcrt - ip + 1] + 1;
                    if (complx) {
                        ni = min_i32(ni + 1, n);
                    }
                }
            }

            for (kk = 1; kk <= kmr; kk++) {
                k = nr + mr - kk;
                if (ip == 1) {
                    k = n - kk;
                }
                SLC_DLARTG(&dwork[k - 1], &dwork[k], &p_val, &q_val, &r_val);
                dwork[k - 1] = r_val;
                dwork[k] = ZERO;

                sl_int n_minus_nj_plus1 = n - nj + 1;
                SLC_DROT(&n_minus_nj_plus1, &a[(k - 1) + (nj - 1) * lda], &lda_sl,
                         &a[k + (nj - 1) * lda], &lda_sl, &p_val, &q_val);

                sl_int ni_sl = ni;
                SLC_DROT(&ni_sl, &a[(k - 1) * lda], &int1,
                         &a[k * lda], &int1, &p_val, &q_val);

                if (k < lp1) {
                    SLC_DROT(&m_sl, &b[(k - 1)], &ldb_sl,
                             &b[k], &ldb_sl, &p_val, &q_val);
                }

                SLC_DROT(&n_sl, &z[(k - 1) * ldz], &int1,
                         &z[k * ldz], &int1, &p_val, &q_val);

                if (complx) {
                    sl_int int1_rot = 1;
                    SLC_DROT(&int1_rot, &dwork[n + k - 1], &int1,
                             &dwork[n + k], &int1, &p_val, &q_val);
                    k = k + 1;
                    if (k < n) {
                        SLC_DLARTG(&dwork[n + k - 1], &dwork[n + k], &p_val, &q_val, &r_val);
                        dwork[n + k - 1] = r_val;
                        dwork[n + k] = ZERO;

                        sl_int n_minus_nj_plus1_2 = n - nj + 1;
                        SLC_DROT(&n_minus_nj_plus1_2, &a[(k - 1) + (nj - 1) * lda], &lda_sl,
                                 &a[k + (nj - 1) * lda], &lda_sl, &p_val, &q_val);

                        SLC_DROT(&ni_sl, &a[(k - 1) * lda], &int1,
                                 &a[k * lda], &int1, &p_val, &q_val);

                        if (k <= lp1) {
                            SLC_DROT(&m_sl, &b[(k - 1)], &ldb_sl,
                                     &b[k], &ldb_sl, &p_val, &q_val);
                        }

                        SLC_DROT(&n_sl, &z[(k - 1) * ldz], &int1,
                                 &z[k * ldz], &int1, &p_val, &q_val);
                    }
                }
            }

            if (ip != indcrt) {
                mr = mr1;
                nr = nr1;
                if (ip != indcn1) {
                    indcn2 = indcrt - ip - 1;
                    mr1 = nblk[indcn2 - 1];
                    nr1 = nr1 - mr1;
                }
            }
        }

        if (!complx) {
            sl_int m1_sl = m1;
            SLC_DLACPY("Full", &m1_sl, &m_sl, &b[l], &ldb_sl, &dwork[irmx - 1], &m_sl);
            SLC_DCOPY(&m1_sl, &a[l + (l - 1) * lda], &int1, &g[(l - 1) * ldg], &int1);
        } else {
            if (lp1 < n) {
                lp1 = lp1 + 1;
                k = l + 2;
            } else {
                k = l + 1;
            }
            sl_int m1_sl = m1, two = 2;
            SLC_DLACPY("Full", &m1_sl, &m_sl, &b[k - 1], &ldb_sl, &dwork[irmx - 1], &m_sl);
            SLC_DLACPY("Full", &m1_sl, &two, &a[(k - 1) + (l - 1) * lda], &lda_sl,
                       &g[(l - 1) * ldg], &ldg_sl);
            if (k == l + 1) {
                g[(l - 1) * ldg] = g[(l - 1) * ldg]
                    - (dwork[n + l] / dwork[l - 1]) * wi[l];
                g[l * ldg] = g[l * ldg] - wr[l]
                    + (dwork[n + l - 1] / dwork[l - 1]) * wi[l];
            }
        }

        sl_int nc_sl = nc;
        sl_int ldwork_sub = ldwork - iwrk + 1;
        mb02qd("FreeElements", "NoPermuting", m1, m, nc, toldef, svlmxb,
               &dwork[irmx - 1], m, &g[(l - 1) * ldg], ldg, &y[*count - 1],
               iwork, &rank, sval, &dwork[iwrk - 1], ldwork_sub, &info_sub);

        maxwrk = max_i32(maxwrk, (i32)dwork[iwrk - 1] + iwrk - 1);
        if (rank < m1) {
            goto exit_rank_deficient;
        }

        *count += (m - m1) * nc;

        sl_int lp1_sl = lp1;
        f64 neg_one = -ONE;
        SLC_DGEMM("No transpose", "No transpose", &lp1_sl, &nc_sl, &m_sl, &neg_one,
                  b, &ldb_sl, &g[(l - 1) * ldg], &ldg_sl, &ONE, &a[(l - 1) * lda], &lda_sl);

        l = l + 1;
        nblkcr = nblkcr - 1;
        if (nblkcr == 0) {
            indcrt = indcrt - 1;
            nblkcr = nblk[indcrt - 1];
        }

        if (complx) {
            wi[l - 1] = -wi[l - 2];
            l = l + 1;
            nblkcr = nblkcr - 1;
            if (nblkcr == 0) {
                indcrt = indcrt - 1;
                if (indcrt > 0) {
                    nblkcr = nblk[indcrt - 1];
                }
            }
        }

        mr = nblkcr;
        nr = n - mr + 1;
    }

    if (l <= n) {
        for (i = 1; i <= mr - 1; i++) {
            ia = l + i - 1;
            mi = mr - i + 1;
            sl_int mi_sl = mi;
            SLC_DCOPY(&mi_sl, &y[*count - 1], &int1, dwork, &int1);
            *count += mi;

            f64 tau;
            SLC_DLARFG(&mi_sl, dwork, &dwork[1], &int1, &tau);
            dwork[0] = ONE;

            sl_int mr_sl = mr;
            SLC_DLARF("Left", &mi_sl, &mr_sl, dwork, &int1, &tau,
                      &a[(ia - 1) + (l - 1) * lda], &lda_sl, &dwork[n]);
            SLC_DLARF("Right", &n_sl, &mi_sl, dwork, &int1, &tau,
                      &a[(ia - 1) * lda], &lda_sl, &dwork[n]);

            SLC_DLARF("Left", &mi_sl, &m_sl, dwork, &int1, &tau,
                      &b[ia - 1], &ldb_sl, &dwork[n]);

            SLC_DLARF("Right", &n_sl, &mi_sl, dwork, &int1, &tau,
                      &z[(ia - 1) * ldz], &ldz_sl, &dwork[n]);
        }

        i = 0;
        while (i < mr) {
            i = i + 1;
            ia = l + i - 1;
            if (wi[ia - 1] == ZERO) {
                sl_int mr_sl = mr;
                SLC_DCOPY(&mr_sl, &a[(ia - 1) + (l - 1) * lda], &lda_sl,
                          &g[(i - 1) + (l - 1) * ldg], &ldg_sl);

                i32 mr_minus_i = mr - i;
                if (mr_minus_i > 0) {
                    sl_int mr_minus_i_sl = mr_minus_i;
                    f64 neg_one = -ONE;
                    SLC_DAXPY(&mr_minus_i_sl, &neg_one, &y[*count - 1], &int1,
                              &g[(i - 1) + (l + i - 1) * ldg], &ldg_sl);
                    *count += mr_minus_i;
                }
                g[(i - 1) + (ia - 1) * ldg] -= wr[ia - 1];
            } else {
                sl_int two = 2, mr_sl = mr;
                SLC_DLACPY("Full", &two, &mr_sl, &a[(ia - 1) + (l - 1) * lda], &lda_sl,
                           &g[(i - 1) + (l - 1) * ldg], &ldg_sl);

                i32 mr_minus_i_minus1 = mr - i - 1;
                if (mr_minus_i_minus1 > 0) {
                    sl_int mr_minus_i_minus1_sl = mr_minus_i_minus1;
                    sl_int two_inc = 2;
                    f64 neg_one = -ONE;
                    SLC_DAXPY(&mr_minus_i_minus1_sl, &neg_one, &y[*count - 1], &two_inc,
                              &g[(i - 1) + (l + i) * ldg], &ldg_sl);
                    SLC_DAXPY(&mr_minus_i_minus1_sl, &neg_one, &y[*count], &two_inc,
                              &g[i + (l + i) * ldg], &ldg_sl);
                    *count += 2 * mr_minus_i_minus1;
                }
                g[(i - 1) + (ia - 1) * ldg] -= wr[ia - 1];
                g[(i - 1) + ia * ldg] -= wi[ia - 1];
                g[i + (ia - 1) * ldg] -= wi[ia];
                g[i + ia * ldg] -= wr[ia];
                i = i + 1;
            }
        }

        sl_int mr_sl = mr;
        SLC_DLACPY("Full", &mr_sl, &m_sl, &b[l - 1], &ldb_sl, &dwork[irmx - 1], &m_sl);

        sl_int ldwork_sub = ldwork - iwrk + 1;
        mb02qd("FreeElements", "NoPermuting", mr, m, mr, toldef, svlmxb,
               &dwork[irmx - 1], m, &g[(l - 1) * ldg], ldg, &y[*count - 1],
               iwork, &rank, sval, &dwork[iwrk - 1], ldwork_sub, &info_sub);

        maxwrk = max_i32(maxwrk, (i32)dwork[iwrk - 1] + iwrk - 1);
        if (rank < mr) {
            goto exit_rank_deficient;
        }

        *count += (m - mr) * mr;

        f64 neg_one = -ONE;
        SLC_DGEMM("No transpose", "No transpose", &n_sl, &mr_sl, &m_sl, &neg_one,
                  b, &ldb_sl, &g[(l - 1) * ldg], &ldg_sl, &ONE, &a[(l - 1) * lda], &lda_sl);
    }

    f64 zero_val = ZERO;
    SLC_DGEMM("No transpose", "Transpose", &m_sl, &n_sl, &n_sl, &ONE,
              g, &ldg_sl, z, &ldz_sl, &zero_val, dwork, &m_sl);

    SLC_DLACPY("Full", &m_sl, &n_sl, dwork, &m_sl, g, &ldg_sl);

    *count = *count - 1;

    if (n > 2) {
        sl_int n_minus2 = n - 2;
        SLC_DLASET("Lower", &n_minus2, &n_minus2, &ZERO, &ZERO, &a[2], &lda_sl);
    }

    dwork[0] = (f64)maxwrk;
    return;

exit_rank_deficient:
    *info = 1;
    return;
}
