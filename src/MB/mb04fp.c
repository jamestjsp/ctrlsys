/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 *
 * MB04FP - Eigenvalues and orthogonal decomposition of a real
 *          skew-Hamiltonian/skew-Hamiltonian pencil
 *          (panel-based version for better performance on large matrices)
 *
 * Computes eigenvalues of aS - bT where:
 *   S = [[A, D], [E, A']] with D, E skew-symmetric
 *   T = [[B, F], [G, B']] with F, G skew-symmetric
 *
 * For small matrices (M <= MMIN=32) or when panel size equals M,
 * this routine delegates to MB04FD.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <ctype.h>
#include <stdlib.h>

void mb04fp(const char *job, const char *compq, i32 n,
            f64 *a, i32 lda, f64 *de, i32 ldde, f64 *b, i32 ldb, f64 *fg, i32 ldfg,
            f64 *q, i32 ldq, f64 *alphar, f64 *alphai, f64 *beta,
            i32 *iwork, f64 *dwork, i32 ldwork, i32 *info_in_out) {

    const f64 ZERO = 0.0;
    const f64 HALF = 0.5;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 THREE = 3.0;
    const f64 TEN = 10.0;
    const i32 MMIN = 32;

    i32 nb = *info_in_out;
    i32 m = n / 2;
    i32 mm = m * m;
    i32 m1 = (1 > m) ? 1 : m;

    if (nb <= 0) {
        i32 lwork_query = -1;
        i32 info_qr;
        f64 work_opt;
        SLC_DGEQRF(&m, &m, a, &lda, dwork, &work_opt, &lwork_query, &info_qr);
        i32 block_size = (i32)(work_opt) / m1;
        nb = (block_size < 2) ? 2 : block_size;
        nb = (nb > m) ? m : nb;
    }

    if (nb == m || m <= MMIN) {
        mb04fd(job, compq, n, a, lda, de, ldde, b, ldb, fg, ldfg, q, ldq,
               alphar, alphai, beta, iwork, dwork, ldwork, info_in_out);
        return;
    }

    char job_upper = (char)toupper((unsigned char)job[0]);
    char compq_upper = (char)toupper((unsigned char)compq[0]);

    bool ltri = (job_upper == 'T');
    bool liniq = (compq_upper == 'I');
    bool lupdq = (compq_upper == 'U');
    bool lcmpq = liniq || lupdq;
    bool lquery = (ldwork == -1);

    i32 iq1 = 0;
    i32 iq2, iwrk, mindw;
    const char *cmpq_str, *cmpz_str;

    if (lcmpq) {
        cmpq_str = "I";
        cmpz_str = "I";
        iq2 = iq1 + mm;
        iwrk = iq2 + mm;
        i32 temp = iwrk - 1 + mm;
        mindw = (3 > temp) ? 3 : temp;
    } else if (ltri) {
        cmpq_str = "I";
        cmpz_str = "N";
        iq2 = 0;
        iwrk = iq2 + mm;
        i32 temp = iwrk - 1 + m;
        mindw = (3 > temp) ? 3 : temp;
    } else {
        cmpq_str = "N";
        cmpz_str = "N";
        iq2 = 0;
        iwrk = 0;
        i32 temp1 = (3 > m) ? 3 : m;
        i32 temp2 = 2 * n - 6;
        mindw = (temp1 > temp2) ? temp1 : temp2;
    }
    i32 k_work = iwrk - 1;

    const char *cmpsc = ltri ? "S" : "E";

    *info_in_out = 0;
    if (!(job_upper == 'E' || ltri)) {
        *info_in_out = -1;
    } else if (!(compq_upper == 'N' || lcmpq)) {
        *info_in_out = -2;
    } else if (n < 0 || (n % 2) != 0) {
        *info_in_out = -3;
    } else if (lda < m1) {
        *info_in_out = -5;
    } else if (ldde < m1) {
        *info_in_out = -7;
    } else if (ldb < m1) {
        *info_in_out = -9;
    } else if (ldfg < m1) {
        *info_in_out = -11;
    } else if (ldq < 1 || (lcmpq && ldq < n)) {
        *info_in_out = -13;
    } else if (!lquery && ldwork < mindw) {
        dwork[0] = (f64)mindw;
        *info_in_out = -19;
    }

    if (*info_in_out != 0) {
        return;
    }

    if (n > 0 && lquery) {
        i32 lwork_query = -1;
        i32 info_dhgeqz;
        f64 work_opt;
        SLC_DHGEQZ(cmpsc, cmpq_str, cmpz_str, &m, &(i32){1}, &m, b, &ldb, a, &lda,
                   alphar, alphai, beta, dwork, &m1, dwork, &m1,
                   &work_opt, &lwork_query, &info_dhgeqz);

        i32 optdw;
        if (lcmpq) {
            i32 temp = (k_work > (i32)work_opt) ? k_work : (i32)work_opt;
            optdw = k_work + temp;
        } else if (ltri) {
            i32 diff = k_work - m;
            i32 temp = (diff > (i32)work_opt) ? diff : (i32)work_opt;
            optdw = k_work + temp;
        } else {
            optdw = (i32)work_opt;
        }
        dwork[0] = (f64)((optdw > mindw) ? optdw : mindw);
        return;
    } else if (lquery) {
        dwork[0] = (f64)mindw;
        return;
    }

    if (n == 0) {
        iwork[0] = 0;
        dwork[0] = THREE;
        dwork[1] = ZERO;
        dwork[2] = ZERO;
        return;
    }

    i32 ninf = 0;
    f64 nrms, nrmt, nrm;

    if (m == 1) {
        nrm = ZERO;
    } else {
        i32 mm1 = m - 1;
        nrm = SLC_DLANTR("M", "L", "N", &mm1, &mm1, &de[1], &ldde, dwork) +
              SLC_DLANTR("M", "U", "N", &mm1, &mm1, &de[ldde], &ldde, dwork);
    }

    if (nrm == ZERO) {
        if (m == 1) {
            nrms = fabs(a[0]);
            if (nrms == ZERO) {
                ninf = 1;
            }
        } else {
            i32 mm1 = m - 1;
            f64 norm_lower = SLC_DLANTR("M", "L", "N", &mm1, &mm1, &a[1], &lda, dwork);
            f64 norm_upper = SLC_DLANTR("M", "U", "N", &mm1, &mm1, &a[lda], &lda, dwork);
            if (norm_lower == ZERO && norm_upper == ZERO) {
                f64 tmp1 = ZERO;
                f64 tmp2 = ONE;
                i32 ldap1 = lda + 1;
                SLC_DLASSQ(&m, a, &ldap1, &tmp1, &tmp2);
                nrms = tmp1 * sqrt(tmp2);
                for (i32 j = 0; j < m; j++) {
                    if (a[j + j * lda] == ZERO) {
                        ninf++;
                    }
                }
            } else {
                i32 nzr, nzc;
                ma02pd(m, m, a, lda, &nzr, &nzc);
                ninf = (nzr > nzc) ? nzr : nzc;
                nrms = SLC_DLANGE("F", &m, &m, a, &lda, dwork);
            }
        }
        nrms = nrms * sqrt(TWO);
    } else {
        ninf = ma02od("S", m, a, lda, de, ldde);
        if ((ninf % 2) > 0) {
            ninf++;
        }
        ninf = ninf / 2;
        nrms = ma02id("S", "F", m, a, lda, de, ldde, dwork);
    }

    nrmt = ma02id("S", "F", m, b, ldb, fg, ldfg, dwork);

    if (liniq) {
        SLC_DLASET("F", &n, &n, &ZERO, &ONE, q, &ldq);
    }

    f64 dum0 = ZERO;
    i32 one = 1;
    i32 zero_inc = 0;

    for (i32 k = 0; k < m - 1; k++) {
        i32 mk2 = (k + 2 < m) ? k + 2 : m - 1;
        i32 mk3 = mk2 + 1;
        f64 tmp1 = de[(k + 1) + k * ldde];
        i32 len = m - k - 1;
        f64 nu;
        SLC_DLARFG(&len, &tmp1, &de[mk2 + k * ldde], &one, &nu);

        if (nu != ZERO) {
            de[(k + 1) + k * ldde] = ONE;
            i32 info_dummy;

            mb01md('L', m - k - 1, nu, &de[(k + 1) + (k + 1) * ldde], ldde,
                   &de[(k + 1) + k * ldde], 1, ZERO, dwork, 1, &info_dummy);

            f64 dot = SLC_DDOT(&len, dwork, &one, &de[(k + 1) + k * ldde], &one);
            f64 mu = -HALF * nu * dot;
            SLC_DAXPY(&len, &mu, &de[(k + 1) + k * ldde], &one, dwork, &one);

            mb01nd('L', m - k - 1, ONE, &de[(k + 1) + k * ldde], 1, dwork, 1,
                   &de[(k + 1) + (k + 1) * ldde], ldde, &info_dummy);

            i32 kp1 = k + 1;
            SLC_DLARF("L", &len, &kp1, &de[(k + 1) + k * ldde], &one, &nu,
                      &fg[(k + 1)], &ldfg, dwork);

            mb01md('L', m - k - 1, nu, &fg[(k + 1) + (k + 1) * ldfg], ldfg,
                   &de[(k + 1) + k * ldde], 1, ZERO, dwork, 1, &info_dummy);

            dot = SLC_DDOT(&len, dwork, &one, &de[(k + 1) + k * ldde], &one);
            mu = -HALF * nu * dot;
            SLC_DAXPY(&len, &mu, &de[(k + 1) + k * ldde], &one, dwork, &one);

            mb01nd('L', m - k - 1, ONE, &de[(k + 1) + k * ldde], 1, dwork, 1,
                   &fg[(k + 1) + (k + 1) * ldfg], ldfg, &info_dummy);

            SLC_DLARF("R", &m, &len, &de[(k + 1) + k * ldde], &one, &nu,
                      &a[(k + 1) * lda], &lda, dwork);
            SLC_DLARF("R", &m, &len, &de[(k + 1) + k * ldde], &one, &nu,
                      &b[(k + 1) * ldb], &ldb, dwork);

            if (lcmpq) {
                SLC_DLARF("R", &n, &len, &de[(k + 1) + k * ldde], &one, &nu,
                          &q[(k + 1) * ldq], &ldq, dwork);
            }
            de[(k + 1) + k * ldde] = tmp1;
        }

        f64 tmp2 = a[(k + 1) + k * lda];
        f64 co, si;
        SLC_DLARTG(&tmp2, &tmp1, &co, &si, &a[(k + 1) + k * lda]);

        i32 len2 = m - k - 2;
        if (len2 > 0) {
            SLC_DROT(&len2, &de[mk2 + (k + 1) * ldde], &one, &a[(k + 1) + mk2 * lda], &lda, &co, &si);
        }
        i32 kp1 = k + 1;
        SLC_DROT(&kp1, &a[(k + 1) * lda], &one, &de[(k + 2) * ldde], &one, &co, &si);
        if (len2 > 0) {
            SLC_DROT(&len2, &de[(k + 1) + mk3 * ldde], &ldde, &a[mk2 + (k + 1) * lda], &one, &co, &si);
        }

        f64 neg_si = -si;
        SLC_DROT(&kp1, &fg[(k + 1)], &ldfg, &b[(k + 1)], &ldb, &co, &neg_si);
        if (len2 > 0) {
            SLC_DROT(&len2, &fg[mk2 + (k + 1) * ldfg], &one, &b[(k + 1) + mk2 * ldb], &ldb, &co, &si);
        }
        SLC_DROT(&kp1, &b[(k + 1) * ldb], &one, &fg[(k + 2) * ldfg], &one, &co, &si);
        if (len2 > 0) {
            SLC_DROT(&len2, &fg[(k + 1) + mk3 * ldfg], &ldfg, &b[mk2 + (k + 1) * ldb], &one, &co, &si);
        }

        if (lcmpq) {
            SLC_DROT(&n, &q[(m + k + 1) * ldq], &one, &q[(k + 1) * ldq], &one, &co, &neg_si);
        }

        tmp1 = a[k + k * lda];
        i32 len_p = m - k;
        SLC_DLARFG(&len_p, &tmp1, &a[(k + 1) + k * lda], &one, &nu);

        if (nu != ZERO) {
            a[k + k * lda] = ONE;
            i32 info_dummy;

            SLC_DLARF("L", &len_p, &len, &a[k + k * lda], &one, &nu,
                      &a[k + (k + 1) * lda], &lda, dwork);

            if (k > 0) {
                i32 km = k;
                SLC_DLARF("R", &km, &len_p, &a[k + k * lda], &one, &nu,
                          &de[(k + 1) * ldde], &ldde, dwork);
            }

            mb01md('U', m - k, nu, &de[k + (k + 1) * ldde], ldde,
                   &a[k + k * lda], 1, ZERO, dwork, 1, &info_dummy);

            f64 dot = SLC_DDOT(&len_p, dwork, &one, &a[k + k * lda], &one);
            f64 mu = -HALF * nu * dot;
            SLC_DAXPY(&len_p, &mu, &a[k + k * lda], &one, dwork, &one);

            mb01nd('U', m - k, ONE, &a[k + k * lda], 1, dwork, 1,
                   &de[k + (k + 1) * ldde], ldde, &info_dummy);

            SLC_DLARF("L", &len_p, &m, &a[k + k * lda], &one, &nu,
                      &b[k], &ldb, dwork);

            if (k > 0) {
                i32 km = k;
                SLC_DLARF("R", &km, &len_p, &a[k + k * lda], &one, &nu,
                          &fg[(k + 1) * ldfg], &ldfg, dwork);
            }

            mb01md('U', m - k, nu, &fg[k + (k + 1) * ldfg], ldfg,
                   &a[k + k * lda], 1, ZERO, dwork, 1, &info_dummy);

            dot = SLC_DDOT(&len_p, dwork, &one, &a[k + k * lda], &one);
            mu = -HALF * nu * dot;
            SLC_DAXPY(&len_p, &mu, &a[k + k * lda], &one, dwork, &one);

            mb01nd('U', m - k, ONE, &a[k + k * lda], 1, dwork, 1,
                   &fg[k + (k + 1) * ldfg], ldfg, &info_dummy);

            if (lcmpq) {
                SLC_DLARF("R", &n, &len_p, &a[k + k * lda], &one, &nu,
                          &q[(m + k) * ldq], &ldq, dwork);
            }
            a[k + k * lda] = tmp1;
        }

        SLC_DCOPY(&len, &dum0, &zero_inc, &a[(k + 1) + k * lda], &one);
    }

    for (i32 k = 0; k < m - 1; k++) {
        i32 js = k + 1;
        i32 je = (js + nb - 1 < m) ? js + nb - 1 : m - 1;
        i32 ic = 0;
        i32 jc = 2 * (m - k - 1);

        for (i32 j = k + 1; j < m - 1; j++) {
            i32 mj2 = (j + 2 < m) ? j + 2 : m - 1;
            i32 mj3 = mj2 + 1;

            f64 co, si, tmp1;
            SLC_DLARTG(&fg[(j + 1) + k * ldfg], &fg[j + k * ldfg], &co, &si, &tmp1);
            dwork[ic] = co;
            dwork[ic + 1] = si;
            ic += 2;

            SLC_DROT(&m, &b[(j + 1) * ldb], &one, &b[j * ldb], &one, &co, &si);
            fg[(j + 1) + k * ldfg] = tmp1;
            i32 len_rot = m - j - 2;
            if (len_rot > 0) {
                SLC_DROT(&len_rot, &fg[mj2 + (j + 1) * ldfg], &one, &fg[mj2 + j * ldfg], &one, &co, &si);
            }

            if (j == je && je < m - 1) {
                i32 js_next = je + 1;
                i32 je_next = (je + nb < m) ? je + nb : m - 1;
                i32 nc = je_next - js_next + 1;
                i32 ja = 2 * (m - k - 1);
                for (i32 i = k + 1; i < j; i++) {
                    SLC_DROT(&nc, &a[i + js_next * lda], &lda, &a[(i + 1) + js_next * lda], &lda,
                             &dwork[ja], &dwork[ja + 1]);
                    ja += 2;
                }
                ja = 2 * (m - k - 1);
                for (i32 i = k + 1; i < j; i++) {
                    SLC_DROT(&nc, &de[i + (js_next + 1) * ldde], &ldde, &de[(i + 1) + (js_next + 1) * ldde], &ldde,
                             &dwork[ja], &dwork[ja + 1]);
                    ja += 2;
                }
                ja = 2 * (m - k - 1);
                for (i32 i = k + 1; i < j; i++) {
                    SLC_DROT(&nc, &fg[i + (js_next + 1) * ldfg], &ldfg, &fg[(i + 1) + (js_next + 1) * ldfg], &ldfg,
                             &dwork[ja], &dwork[ja + 1]);
                    ja += 2;
                }
                js = js_next;
                je = je_next;
            }

            i32 jp1 = j + 1;
            SLC_DROT(&jp1, &a[(j + 1) * lda], &one, &a[j * lda], &one, &co, &si);
            tmp1 = -si * a[(j + 1) + (j + 1) * lda];
            a[(j + 1) + (j + 1) * lda] = co * a[(j + 1) + (j + 1) * lda];

            if (lcmpq) {
                SLC_DROT(&n, &q[(j + 1) * ldq], &one, &q[j * ldq], &one, &co, &si);
            }

            f64 tmp2;
            SLC_DLARTG(&a[j + j * lda], &tmp1, &co, &si, &tmp2);
            dwork[jc] = co;
            dwork[jc + 1] = si;
            jc += 2;

            i32 nc_local = (je - j < je - js + 1) ? je - j : je - js + 1;
            nc_local = (nc_local > 0) ? nc_local : 0;
            a[j + j * lda] = tmp2;
            if (nc_local > 0) {
                SLC_DROT(&nc_local, &a[j + (j + 1) * lda], &lda, &a[(j + 1) + (j + 1) * lda], &lda, &co, &si);
            }
            i32 jm1 = j;
            if (jm1 > 0) {
                SLC_DROT(&jm1, &de[(j + 1) * ldde], &one, &de[(j + 2) * ldde], &one, &co, &si);
            }
            i32 nc_de = nc_local - 1;
            if (nc_de > 0) {
                SLC_DROT(&nc_de, &de[j + mj3 * ldde], &ldde, &de[(j + 1) + mj3 * ldde], &ldde, &co, &si);
            }

            if (jm1 > 0) {
                SLC_DROT(&jm1, &fg[(j + 1) * ldfg], &one, &fg[(j + 2) * ldfg], &one, &co, &si);
            }
            if (nc_de > 0) {
                SLC_DROT(&nc_de, &fg[j + mj3 * ldfg], &ldfg, &fg[(j + 1) + mj3 * ldfg], &ldfg, &co, &si);
            }

            if (lcmpq) {
                SLC_DROT(&n, &q[(m + j) * ldq], &one, &q[(m + j + 1) * ldq], &one, &co, &si);
            }
        }

        for (i32 js_b = 0; js_b < m; js_b += nb) {
            i32 nc = ((js_b + nb - 1 < m - 1) ? js_b + nb - 1 : m - 1) - js_b + 1;
            i32 jc_b = 2 * (m - k - 1);
            for (i32 j = k + 1; j < m - 1; j++) {
                SLC_DROT(&nc, &b[j + js_b * ldb], &ldb, &b[(j + 1) + js_b * ldb], &ldb,
                         &dwork[jc_b], &dwork[jc_b + 1]);
                jc_b += 2;
            }
        }

        i32 ics = 2;
        i32 je_w = k;

        while (je_w < m - 2) {
            i32 nc_w = 0;
            i32 ic_w = ics;
            ics += 2 * nb;
            for (i32 j = je_w + 2; j < m - 1; j++) {
                nc_w = nc_w + 1;
                if (nc_w > nb) nc_w = nb;
                i32 js_w = je_w + 1;
                SLC_DROT(&nc_w, &fg[(j + 1) + js_w * ldfg], &ldfg, &fg[j + js_w * ldfg], &ldfg,
                         &dwork[ic_w], &dwork[ic_w + 1]);
                ic_w += 2;
            }
            je_w += nb;
        }

        f64 co, si, tmp1;
        f64 neg_fg = -fg[(m - 1) + k * ldfg];
        SLC_DLARTG(&b[(m - 1) + k * ldb], &neg_fg, &co, &si, &tmp1);

        b[(m - 1) + k * ldb] = tmp1;
        fg[(m - 1) + k * ldfg] = ZERO;
        i32 mm1 = m - 1;
        SLC_DROT(&mm1, &fg[m * ldfg], &one, &b[(m - 1) * ldb], &one, &co, &si);
        i32 len_rot = m - k - 2;
        if (len_rot > 0) {
            SLC_DROT(&len_rot, &fg[(m - 1) + (k + 1) * ldfg], &ldfg, &b[(m - 1) + (k + 1) * ldb], &ldb, &co, &si);
        }

        SLC_DROT(&mm1, &de[m * ldde], &one, &a[(m - 1) * lda], &one, &co, &si);

        if (lcmpq) {
            SLC_DROT(&n, &q[(n - 1) * ldq], &one, &q[(m - 1) * ldq], &one, &co, &si);
        }

        ic = 0;
        jc = 2 * (m - k - 1);
        for (i32 j = m - 1; j >= k + 2; j--) {
            f64 co_inner, si_inner, tmp1_inner;
            SLC_DLARTG(&b[(j - 1) + k * ldb], &b[j + k * ldb], &co_inner, &si_inner, &tmp1_inner);
            dwork[ic] = co_inner;
            dwork[ic + 1] = si_inner;
            ic += 2;

            b[(j - 1) + k * ldb] = tmp1_inner;
            b[j + k * ldb] = ZERO;
            i32 len_inner = j - 1;
            if (len_inner > 0) {
                SLC_DROT(&len_inner, &fg[j * ldfg], &one, &fg[(j + 1) * ldfg], &one, &co_inner, &si_inner);
            }

            tmp1_inner = -si_inner * a[(j - 1) + (j - 1) * lda];
            a[(j - 1) + (j - 1) * lda] = co_inner * a[(j - 1) + (j - 1) * lda];
            SLC_DROT(&one, &a[(j - 1) + j * lda], &lda, &a[j + j * lda], &lda, &co_inner, &si_inner);
            if (len_inner > 0) {
                SLC_DROT(&len_inner, &de[j * ldde], &one, &de[(j + 1) * ldde], &one, &co_inner, &si_inner);
            }

            if (lcmpq) {
                SLC_DROT(&n, &q[(m + j - 1) * ldq], &one, &q[(m + j) * ldq], &one, &co_inner, &si_inner);
            }

            f64 tmp2_inner;
            SLC_DLARTG(&a[j + j * lda], &tmp1_inner, &co_inner, &si_inner, &tmp2_inner);
            dwork[jc] = co_inner;
            dwork[jc + 1] = si_inner;
            jc += 2;

            a[j + j * lda] = tmp2_inner;
            i32 jval = j;
            SLC_DROT(&jval, &a[j * lda], &one, &a[(j - 1) * lda], &one, &co_inner, &si_inner);

            SLC_DROT(&m, &b[j * ldb], &one, &b[(j - 1) * ldb], &one, &co_inner, &si_inner);

            if (lcmpq) {
                SLC_DROT(&n, &q[j * ldq], &one, &q[(j - 1) * ldq], &one, &co_inner, &si_inner);
            }
        }

        for (i32 js_b = k + 1; js_b < m; js_b += nb) {
            i32 nc = ((js_b + nb - 1 < m - 1) ? js_b + nb - 1 : m - 1) - js_b + 1;
            i32 ic_b = 0;
            for (i32 j = m - 1; j >= k + 2; j--) {
                SLC_DROT(&nc, &b[(j - 1) + js_b * ldb], &ldb, &b[j + js_b * ldb], &ldb,
                         &dwork[ic_b], &dwork[ic_b + 1]);
                ic_b += 2;
            }
        }

        ics = 2;
        je_w = m;

        while (je_w > 3) {
            i32 nc_w = 0;
            i32 ic_w = ics;
            ics += 2 * nb;
            for (i32 j = je_w - 1; j >= k + 3; j--) {
                nc_w = nc_w + 1;
                if (nc_w > nb) nc_w = nb;
                i32 js_w = je_w - nc_w + 2;
                SLC_DROT(&nc_w, &fg[(j - 2) + (js_w - 1) * ldfg], &ldfg, &fg[(j - 1) + (js_w - 1) * ldfg], &ldfg,
                         &dwork[ic_w], &dwork[ic_w + 1]);
                ic_w += 2;
            }
            je_w -= nb;
        }

        ics = 2;
        je_w = m;

        while (je_w > 2) {
            i32 nc_w = 0;
            i32 ic_w = ics;
            ics += 2 * nb;
            for (i32 j = je_w - 1; j >= k + 3; j--) {
                nc_w = nc_w + 1;
                if (nc_w > nb) nc_w = nb;
                i32 js_w = je_w - nc_w + 1;
                SLC_DROT(&nc_w, &a[(j - 2) + (js_w - 1) * lda], &lda, &a[(j - 1) + (js_w - 1) * lda], &lda,
                         &dwork[ic_w], &dwork[ic_w + 1]);
                ic_w += 2;
            }
            je_w -= nb;
        }

        ics = 2;
        je_w = m;

        while (je_w > 3) {
            i32 nc_w = 0;
            i32 ic_w = ics;
            ics += 2 * nb;
            for (i32 j = je_w - 1; j >= k + 3; j--) {
                nc_w = nc_w + 1;
                if (nc_w > nb) nc_w = nb;
                i32 js_w = je_w - nc_w + 2;
                SLC_DROT(&nc_w, &de[(j - 2) + (js_w - 1) * ldde], &ldde, &de[(j - 1) + (js_w - 1) * ldde], &ldde,
                         &dwork[ic_w], &dwork[ic_w + 1]);
                ic_w += 2;
            }
            je_w -= nb;
        }

        i32 ics_fg = 2 * (m - k - 1);
        for (i32 js_fg = k; js_fg < m - 1; js_fg += nb) {
            i32 ic_fg = ics_fg;
            for (i32 j = m; j >= js_fg + 3; j--) {
                i32 nc_fg = j - js_fg - 1;
                if (nc_fg > nb) nc_fg = nb;
                if (nc_fg > 0) {
                    SLC_DROT(&nc_fg, &fg[(j - 1) + js_fg * ldfg], &ldfg, &fg[(j - 2) + js_fg * ldfg], &ldfg,
                             &dwork[ic_fg], &dwork[ic_fg + 1]);
                }
                ic_fg += 2;
            }
        }

        ic = 2 * (m - k - 1);
        for (i32 j = m; j >= k + 3; j--) {
            i32 mj1 = (j + 1 < m) ? j + 1 : m - 1;
            i32 len_fg = m - j;
            if (len_fg > 0) {
                SLC_DROT(&len_fg, &fg[mj1 + (j - 1) * ldfg], &one, &fg[mj1 + (j - 2) * ldfg], &one,
                         &dwork[ic], &dwork[ic + 1]);
            }
            ic += 2;
        }
    }

    i32 lwork_avail = ldwork - iwrk;
    if (lwork_avail < 1) lwork_avail = 1;
    i32 info_dhgeqz = 0;

    SLC_DHGEQZ(cmpsc, cmpq_str, cmpz_str, &m, &(i32){1}, &m, b, &ldb, a, &lda,
               alphar, alphai, beta, &dwork[iq1], &m, &dwork[iq2], &m,
               &dwork[iwrk], &lwork_avail, &info_dhgeqz);

    if (info_dhgeqz > 0) {
        *info_in_out = 1;
        return;
    }

    i32 optdw = (mindw > (i32)dwork[iwrk] + iwrk - 1) ? mindw : (i32)dwork[iwrk] + iwrk - 1;

    i32 j_eig = 0;
    i32 nbeta0 = 0;
    while (j_eig < m) {
        if (alphai[j_eig] != ZERO) {
            f64 tmp1_e, tmp2_e;
            if (beta[j_eig] >= beta[j_eig + 1]) {
                tmp2_e = beta[j_eig + 1] / beta[j_eig];
                tmp1_e = (alphar[j_eig] * tmp2_e + alphar[j_eig + 1]) / TWO;
                tmp2_e = (alphai[j_eig] * tmp2_e - alphai[j_eig + 1]) / TWO;
                beta[j_eig] = beta[j_eig + 1];
            } else {
                tmp2_e = beta[j_eig] / beta[j_eig + 1];
                tmp1_e = (alphar[j_eig + 1] * tmp2_e + alphar[j_eig]) / TWO;
                tmp2_e = (alphai[j_eig + 1] * tmp2_e - alphai[j_eig]) / TWO;
                beta[j_eig + 1] = beta[j_eig];
            }
            alphar[j_eig] = tmp1_e;
            alphar[j_eig + 1] = tmp1_e;
            alphai[j_eig] = tmp2_e;
            alphai[j_eig + 1] = -tmp2_e;
            j_eig += 2;
        } else {
            if (ninf > 0) {
                if (beta[j_eig] == ZERO) {
                    nbeta0++;
                }
            }
            j_eig++;
        }
    }
    if (j_eig == m && ninf > 0) {
        if (beta[m - 1] == ZERO) {
            nbeta0++;
        }
    }

    if (ninf > 0) {
        for (i32 jj = 0; jj < ninf - nbeta0; jj++) {
            f64 tmp1_inf = ZERO;
            f64 tmp2_inf = ONE;
            i32 p = 0;
            for (i32 kk = 0; kk < m; kk++) {
                if (beta[kk] > ZERO) {
                    if (fabs(alphar[kk]) * tmp2_inf > tmp1_inf * beta[kk]) {
                        tmp1_inf = fabs(alphar[kk]);
                        tmp2_inf = beta[kk];
                        p = kk;
                    }
                }
            }
            beta[p] = ZERO;
        }
    }

    if (ltri) {
        i32 info_mb01ld = 0;
        mb01ld("U", "T", m, m, ZERO, ONE, &de[ldde], ldde, &dwork[iq1], m,
               &de[ldde], ldde, &dwork[iwrk], ldwork - iwrk, &info_mb01ld);

        mb01ld("U", "T", m, m, ZERO, ONE, &fg[ldfg], ldfg, &dwork[iq1], m,
               &fg[ldfg], ldfg, &dwork[iwrk], ldwork - iwrk, &info_mb01ld);
    }

    if (lcmpq) {
        if (ldwork >= n * n) {
            SLC_DGEMM("N", "N", &n, &m, &m, &ONE, q, &ldq, &dwork[iq2], &m, &ZERO, &dwork[iwrk], &n);
            SLC_DLACPY("F", &n, &m, &dwork[iwrk], &n, q, &ldq);
            SLC_DGEMM("N", "N", &n, &m, &m, &ONE, &q[m * ldq], &ldq, &dwork[iq1], &m, &ZERO, &dwork[iwrk], &n);
            SLC_DLACPY("F", &n, &m, &dwork[iwrk], &n, &q[m * ldq], &ldq);
        } else {
            SLC_DGEMM("N", "N", &m, &m, &m, &ONE, q, &ldq, &dwork[iq2], &m, &ZERO, &dwork[iwrk], &m);
            SLC_DLACPY("F", &m, &m, &dwork[iwrk], &m, q, &ldq);
            SLC_DGEMM("N", "N", &m, &m, &m, &ONE, &q[m], &ldq, &dwork[iq2], &m, &ZERO, &dwork[iwrk], &m);
            SLC_DLACPY("F", &m, &m, &dwork[iwrk], &m, &q[m], &ldq);
            SLC_DGEMM("N", "N", &m, &m, &m, &ONE, &q[m * ldq], &ldq, &dwork[iq1], &m, &ZERO, &dwork[iwrk], &m);
            SLC_DLACPY("F", &m, &m, &dwork[iwrk], &m, &q[m * ldq], &ldq);
            SLC_DGEMM("N", "N", &m, &m, &m, &ONE, &q[m + m * ldq], &ldq, &dwork[iq1], &m, &ZERO, &dwork[iwrk], &m);
            SLC_DLACPY("F", &m, &m, &dwork[iwrk], &m, &q[m + m * ldq], &ldq);
        }
    }

    f64 tols = TEN * SLC_DLAMCH("P") * nrms;
    f64 tolt = TEN * SLC_DLAMCH("P") * nrmt;
    i32 p = 0;
    i32 kk = 0;
    bool sing, psng;

    while (kk < m) {
        if (beta[kk] != ZERO) {
            if (alphai[kk] == ZERO) {
                sing = fabs(b[kk + kk * ldb]) < tolt;
                if (sing) {
                    p++;
                    iwork[p] = kk + 1;
                }
                if (fabs(a[kk + kk * lda]) < tols) {
                    if (sing) {
                        *info_in_out = 2;
                    } else {
                        p++;
                        iwork[p] = kk + 1;
                    }
                }
            } else {
                f64 x1 = b[kk + kk * ldb];
                f64 x2 = b[(kk + 1) + kk * ldb];
                f64 x3 = b[kk + (kk + 1) * ldb];
                f64 x4 = b[(kk + 1) + (kk + 1) * ldb];
                i32 two = 2;
                f64 norm_b = SLC_DLANGE("F", &two, &two, &b[kk + kk * ldb], &ldb, dwork);

                f64 sdet = (fabs(x1) > fabs(x4) ? fabs(x1) : fabs(x4)) / norm_b *
                           (fabs(x1) < fabs(x4) ? fabs(x1) : fabs(x4)) *
                           (x1 >= 0 ? ONE : -ONE) * (x4 >= 0 ? ONE : -ONE) -
                           (fabs(x2) > fabs(x3) ? fabs(x2) : fabs(x3)) / norm_b *
                           (fabs(x2) < fabs(x3) ? fabs(x2) : fabs(x3)) *
                           (x2 >= 0 ? ONE : -ONE) * (x3 >= 0 ? ONE : -ONE);

                if (norm_b > ONE) {
                    psng = fabs(sdet) < tolt / norm_b;
                } else {
                    psng = fabs(sdet) * norm_b < tolt;
                }

                if (psng) {
                    f64 co, si, tmp_val;
                    if (fabs(x1) >= fabs(x4)) {
                        SLC_DLARTG(&x1, &x2, &co, &si, &tmp_val);
                        x1 = tmp_val;
                        tmp_val = co * x3 + si * x4;
                        x4 = co * x4 - si * x3;
                        x3 = tmp_val;
                    } else {
                        SLC_DLARTG(&x4, &x2, &co, &si, &tmp_val);
                        x4 = tmp_val;
                        tmp_val = co * x3 + si * x1;
                        x1 = co * x1 - si * x3;
                        x3 = tmp_val;
                    }
                    f64 tmp1_sv, tmp2_sv;
                    SLC_DLAS2(&x1, &x3, &x4, &tmp1_sv, &tmp2_sv);
                    sing = tmp1_sv < tolt;
                    if (sing) {
                        p++;
                        iwork[p] = -(kk + 1);
                        *info_in_out = 2;
                    }
                }

                x1 = a[kk + kk * lda];
                x4 = a[(kk + 1) + (kk + 1) * lda];
                norm_b = sqrt(x1 * x1 + x4 * x4);
                f64 max_x = (fabs(x1) > fabs(x4)) ? fabs(x1) : fabs(x4);
                f64 min_x = (fabs(x1) < fabs(x4)) ? fabs(x1) : fabs(x4);
                sdet = (max_x / norm_b) * min_x;
                if (fabs(sdet) < tols) {
                    if (sing) {
                        *info_in_out = 2;
                    } else {
                        p++;
                        iwork[p] = -(kk + 1);
                    }
                }
                kk++;
            }
        } else {
            iwork[kk + 1] = 0;
        }
        kk++;
    }
    iwork[0] = p;

    dwork[0] = (f64)optdw;
    dwork[1] = nrms;
    dwork[2] = nrmt;
}
