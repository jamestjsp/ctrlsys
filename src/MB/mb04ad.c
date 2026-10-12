/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

void mb04ad(const char *job, const char *compq1, const char *compq2,
            const char *compu1, const char *compu2, i32 n,
            f64 *z, i32 ldz, f64 *h, i32 ldh,
            f64 *q1, i32 ldq1, f64 *q2, i32 ldq2,
            f64 *u11, i32 ldu11, f64 *u12, i32 ldu12,
            f64 *u21, i32 ldu21, f64 *u22, i32 ldu22,
            f64 *t, i32 ldt,
            f64 *alphar, f64 *alphai, f64 *beta,
            i32 *iwork, i32 liwork,
            f64 *dwork, i32 ldwork,
            i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 HALF = 0.5;
    const f64 ONE = 1.0;
    const f64 SEVEN = 7.0;

    bool ltri, liniq1, lupdq1, liniq2, lupdq2, liniu1, lupdu1, liniu2, lupdu2;
    bool lcmpq1, lcmpq2, lcmpu1, lcmpu2, lquery, unrel;
    i32 m, mm, mindw, optdw, emin, emax;
    i32 i, j, k, l, p, itau, iwrk, imat, iq, iw;
    i32 i11, i22, i2x2, ninf, nbeta0;
    f64 base, co, si, temp, tmp1, tmp2;
    f64 dum[6];
    i32 idum[1];
    i32 int1 = 1, int0 = 0;
    f64 dbl0 = ZERO, dbl1 = ONE;

    m = n / 2;
    mm = m * m;

    ltri = (*job == 'T' || *job == 't');
    liniq1 = (*compq1 == 'I' || *compq1 == 'i');
    lupdq1 = (*compq1 == 'U' || *compq1 == 'u');
    liniq2 = (*compq2 == 'I' || *compq2 == 'i');
    lupdq2 = (*compq2 == 'U' || *compq2 == 'u');
    liniu1 = (*compu1 == 'I' || *compu1 == 'i');
    lupdu1 = (*compu1 == 'U' || *compu1 == 'u');
    liniu2 = (*compu2 == 'I' || *compu2 == 'i');
    lupdu2 = (*compu2 == 'U' || *compu2 == 'u');
    lcmpq1 = liniq1 || lupdq1;
    lcmpq2 = liniq2 || lupdq2;
    lcmpu1 = liniu1 || lupdu1;
    lcmpu2 = liniu2 || lupdu2;

    if (n == 0) {
        mindw = 7;
    } else if (ltri || lcmpq1 || lcmpq2 || lcmpu1 || lcmpu2) {
        mindw = 12 * mm + (6 * n > 54 ? 6 * n : 54);
    } else {
        mindw = 6 * mm + (6 * n > 54 ? 6 * n : 54);
    }
    lquery = (ldwork == -1);

    *info = 0;
    if (!(*job == 'E' || *job == 'e') && !ltri) {
        *info = -1;
    } else if (!(*compq1 == 'N' || *compq1 == 'n') && !lcmpq1) {
        *info = -2;
    } else if (!(*compq2 == 'N' || *compq2 == 'n') && !lcmpq2) {
        *info = -3;
    } else if (!(*compu1 == 'N' || *compu1 == 'n') && !lcmpu1) {
        *info = -4;
    } else if (!(*compu2 == 'N' || *compu2 == 'n') && !lcmpu2) {
        *info = -5;
    } else if (n < 0 || (n % 2) != 0) {
        *info = -6;
    } else if (ldz < (1 > n ? 1 : n)) {
        *info = -8;
    } else if (ldh < (1 > n ? 1 : n)) {
        *info = -10;
    } else if (ldq1 < 1 || (lcmpq1 && ldq1 < n)) {
        *info = -12;
    } else if (ldq2 < 1 || (lcmpq2 && ldq2 < n)) {
        *info = -14;
    } else if (ldu11 < 1 || (lcmpu1 && ldu11 < m)) {
        *info = -16;
    } else if (ldu12 < 1 || (lcmpu1 && ldu12 < m)) {
        *info = -18;
    } else if (ldu21 < 1 || (lcmpu2 && ldu21 < m)) {
        *info = -20;
    } else if (ldu22 < 1 || (lcmpu2 && ldu22 < m)) {
        *info = -22;
    } else if (ldt < (1 > n ? 1 : n)) {
        *info = -24;
    } else if (liwork < n + 18) {
        *info = -29;
    } else if (!lquery && ldwork < mindw) {
        dwork[0] = (f64)mindw;
        *info = -31;
    }

    if (*info != 0) {
        i32 neg_info = -(*info);
        SLC_XERBLA("MB04AD", &neg_info);
        return;
    }

    if (n > 0 && lquery) {
        i32 info_temp = 0;
        SLC_DORMQR("Left", "Transpose", &n, &n, &m, t, &ldt, dwork, t, &ldt, dwork, &int1, &info_temp);
        i32 neg1 = -1;
        SLC_DORMRQ("Right", "Transpose", &n, &m, &m, z, &ldz, dwork, h, &ldh, dum, &neg1, &info_temp);
        i32 opt1 = (i32)dwork[0];
        i32 opt2 = (i32)dum[0];
        optdw = opt1 > opt2 ? opt1 : opt2;
        SLC_DGERQF(&m, &m, dwork, &m, dwork, dwork, &neg1, &info_temp);
        SLC_DORMRQ("Left", "No Transpose", &m, &n, &m, dwork, &m, dwork, h, &ldh, dum, &neg1, &info_temp);
        i32 opt3 = (i32)dwork[0];
        i32 opt4 = (i32)dum[0];
        i32 opt34 = opt3 > opt4 ? opt3 : opt4;
        optdw = m + (optdw > (mm + opt34) ? optdw : (mm + opt34));
        if (lcmpq1) {
            optdw = optdw > (m + mm + opt2) ? optdw : (m + mm + opt2);
        }
        SLC_DGEQRF(&n, &m, dwork, &n, dwork, dwork, &neg1, &info_temp);
        SLC_DORMQR("Right", "No Transpose", &n, &n, &m, dwork, &n, dwork, h, &ldh, dum, &neg1, &info_temp);
        opt3 = (i32)dwork[0];
        opt4 = (i32)dum[0];
        opt34 = opt3 > opt4 ? opt3 : opt4;
        optdw = optdw > (m + m * n + opt34) ? optdw : (m + m * n + opt34);
        if (lcmpq2) {
            SLC_DORMQR("Left", "No Transpose", &n, &n, &m, dwork, &n, dwork, q2, &ldq2, dwork, &neg1, &info_temp);
            optdw = optdw > (m + m * n + (i32)dwork[0]) ? optdw : (m + m * n + (i32)dwork[0]);
        }
        dwork[0] = (f64)(optdw > mindw ? optdw : mindw);
        return;
    }

    dum[0] = ZERO;

    if (n == 0) {
        iwork[0] = 0;
        dwork[0] = SEVEN;
        SLC_DCOPY(&int1, dum, &int1, &dwork[1], &int1);
        for (i = 2; i < 7; i++) dwork[i] = ZERO;
        return;
    }

    base = SLC_DLAMCH("Base");
    emin = (i32)SLC_DLAMCH("Minimum Exponent");
    emax = (i32)SLC_DLAMCH("Largest Exponent");

    ninf = 0;
    if (n == 1) {
        if (z[0] == ZERO) ninf = 1;
    } else {
        i32 nm1 = n - 1;
        f64 norm_lower = SLC_DLANTR("Max", "Lower", "No-diag", &nm1, &nm1, &z[1], &ldz, dwork);
        f64 norm_upper = SLC_DLANTR("Max", "Upper", "No-diag", &nm1, &nm1, &z[ldz], &ldz, dwork);
        if (norm_lower == ZERO && norm_upper == ZERO) {
            for (j = 0; j < m; j++) {
                if (z[j + j * ldz] == ZERO || z[(j + m) + (j + m) * ldz] == ZERO) {
                    ninf++;
                }
            }
        } else {
            for (j = 0; j < m; j++) {
                i32 idx_i = SLC_IDAMAX(&n, &z[j * ldz], &int1) - 1;
                i32 idx_k = SLC_IDAMAX(&n, &z[(m + j) * ldz], &int1) - 1;
                i32 idx_l = SLC_IDAMAX(&n, &z[j], &ldz) - 1;
                i32 idx_p = SLC_IDAMAX(&n, &z[m + j], &ldz) - 1;
                if (z[idx_i + j * ldz] == ZERO || z[idx_k + (m + j) * ldz] == ZERO ||
                    z[j + idx_l * ldz] == ZERO || z[(m + j) + idx_p * ldz] == ZERO) {
                    ninf++;
                }
            }
        }
    }

    ma02ad("Full", m, m, &z[(m) + (m) * ldz], ldz, t, ldt);
    ma02ad("Full", m, m, &z[(m) * ldz], ldz, &t[(m) * ldt], ldt);

    for (i = 0; i < m; i++) {
        SLC_DSCAL(&m, &(f64){-ONE}, &t[(m + i) * ldt], &int1);
    }

    ma02ad("Full", m, m, &z[m], ldz, &t[m], ldt);

    for (i = 0; i < m; i++) {
        SLC_DSCAL(&m, &(f64){-ONE}, &t[m + i * ldt], &int1);
    }

    ma02ad("Full", m, m, z, ldz, &t[m + m * ldt], ldt);

    if (liniq1) {
        SLC_DLASET("Full", &n, &n, &dbl0, &dbl1, q1, &ldq1);
    }

    if (liniq2) {
        SLC_DLASET("Full", &n, &n, &dbl0, &dbl1, q2, &ldq2);
    }

    if (liniu1) {
        SLC_DLASET("Full", &m, &m, &dbl0, &dbl1, u11, &ldu11);
        SLC_DLASET("Full", &m, &m, &dbl0, &dbl0, u12, &ldu12);
    }

    if (liniu2) {
        SLC_DLASET("Full", &m, &m, &dbl0, &dbl1, u21, &ldu21);
        SLC_DLASET("Full", &m, &m, &dbl0, &dbl0, u22, &ldu22);
    }

    itau = 0;
    iwrk = itau + m;

    {
        i32 ldwork_rem = ldwork - iwrk;
        i32 info_temp = 0;
        SLC_DGEQRF(&n, &m, t, &ldt, &dwork[itau], &dwork[iwrk], &ldwork_rem, &info_temp);
    }

    {
        i32 ldwork_rem = ldwork - iwrk;
        i32 info_temp = 0;
        SLC_DORMQR("Left", "Transpose", &n, &m, &m, t, &ldt, &dwork[itau],
                   &t[m * ldt], &ldt, &dwork[iwrk], &ldwork_rem, &info_temp);
    }

    {
        i32 ldwork_rem = ldwork - iwrk;
        i32 info_temp = 0;
        SLC_DORMQR("Left", "Transpose", &n, &n, &m, t, &ldt, &dwork[itau],
                   h, &ldh, &dwork[iwrk], &ldwork_rem, &info_temp);
        optdw = mindw > ((i32)dwork[iwrk] + iwrk) ? mindw : ((i32)dwork[iwrk] + iwrk);
    }

    if (lcmpq1) {
        i32 ldwork_rem = ldwork - iwrk;
        i32 info_temp = 0;
        SLC_DORMQR("Right", "No Transpose", &n, &n, &m, t, &ldt, &dwork[itau],
                   q1, &ldq1, &dwork[iwrk], &ldwork_rem, &info_temp);
    }

    {
        i32 nm1 = n - 1;
        SLC_DLASET("Lower", &nm1, &m, &dbl0, &dbl0, &t[1], &ldt);
    }

    itau = mm;
    iwrk = itau + m;

    ma02ad("Full", m, m, &t[m + m * ldt], ldt, dwork, m);

    {
        i32 ldwork_rem = ldwork - iwrk;
        i32 info_temp = 0;
        SLC_DGERQF(&m, &m, dwork, &m, &dwork[itau], &dwork[iwrk], &ldwork_rem, &info_temp);
        optdw = optdw > ((i32)dwork[iwrk] + iwrk) ? optdw : ((i32)dwork[iwrk] + iwrk);
    }

    ma02ad("Upper", m, m, dwork, m, &t[m + m * ldt], ldt);

    if (m > 1) {
        i32 mm1 = m - 1;
        SLC_DLASET("Upper", &mm1, &mm1, &dbl0, &dbl0, &t[m + (m + 1) * ldt], &ldt);
    }

    {
        i32 ldwork_rem = ldwork - iwrk;
        i32 info_temp = 0;
        SLC_DORMRQ("Left", "No Transpose", &m, &n, &m, dwork, &m, &dwork[itau],
                   &h[m], &ldh, &dwork[iwrk], &ldwork_rem, &info_temp);
        optdw = optdw > ((i32)dwork[iwrk] + iwrk) ? optdw : ((i32)dwork[iwrk] + iwrk);
    }

    if (lcmpq1) {
        i32 ldwork_rem = ldwork - iwrk;
        i32 info_temp = 0;
        SLC_DORMRQ("Right", "Transpose", &n, &m, &m, dwork, &m, &dwork[itau],
                   &q1[m * ldq1], &ldq1, &dwork[iwrk], &ldwork_rem, &info_temp);
        optdw = optdw > ((i32)dwork[iwrk] + iwrk) ? optdw : ((i32)dwork[iwrk] + iwrk);
    }

    itau = m * n;
    iwrk = itau + m;

    ma02ad("Full", m, n, &z[m], ldz, dwork, n);

    {
        i32 ldwork_rem = ldwork - iwrk;
        i32 info_temp = 0;
        SLC_DGEQRF(&n, &m, dwork, &n, &dwork[itau], &dwork[iwrk], &ldwork_rem, &info_temp);
        optdw = optdw > ((i32)dwork[iwrk] + iwrk) ? optdw : ((i32)dwork[iwrk] + iwrk);
    }

    {
        i32 ldwork_rem = ldwork - iwrk;
        i32 info_temp = 0;
        SLC_DORMQR("Right", "No Transpose", &m, &n, &m, dwork, &n, &dwork[itau],
                   z, &ldz, &dwork[iwrk], &ldwork_rem, &info_temp);
    }

    ma02ad("Upper", m, m, dwork, n, &z[m + m * ldz], ldz);

    if (m > 1) {
        i32 mm1 = m - 1;
        SLC_DLASET("Upper", &mm1, &mm1, &dbl0, &dbl0, &z[m + (m + 1) * ldz], &ldz);
    }

    {
        i32 ldwork_rem = ldwork - iwrk;
        i32 info_temp = 0;
        SLC_DORMQR("Right", "No Transpose", &n, &n, &m, dwork, &n, &dwork[itau],
                   h, &ldh, &dwork[iwrk], &ldwork_rem, &info_temp);
        optdw = optdw > ((i32)dwork[iwrk] + iwrk) ? optdw : ((i32)dwork[iwrk] + iwrk);
    }

    for (i = 0; i < m; i++) {
        SLC_DSWAP(&n, &h[i * ldh], &int1, &h[(m + i) * ldh], &int1);
    }

    if (lcmpq2) {
        i32 ldwork_rem = ldwork - iwrk;
        i32 info_temp = 0;
        SLC_DORMQR("Right", "No Transpose", &n, &n, &m, dwork, &n, &dwork[itau],
                   q2, &ldq2, &dwork[iwrk], &ldwork_rem, &info_temp);
        optdw = optdw > ((i32)dwork[iwrk] + iwrk) ? optdw : ((i32)dwork[iwrk] + iwrk);

        for (i = 0; i < m; i++) {
            SLC_DSWAP(&n, &q2[i * ldq2], &int1, &q2[(m + i) * ldq2], &int1);
        }
    }

    itau = 0;
    iwrk = itau + m;

    {
        i32 ldwork_rem = ldwork - iwrk;
        i32 info_temp = 0;
        SLC_DGERQF(&m, &m, &z[m * ldz], &ldz, &dwork[itau], &dwork[iwrk], &ldwork_rem, &info_temp);
    }

    {
        i32 ldwork_rem = ldwork - iwrk;
        i32 info_temp = 0;
        SLC_DORMRQ("Right", "Transpose", &n, &m, &m, &z[m * ldz], &ldz, &dwork[itau],
                   h, &ldh, &dwork[iwrk], &ldwork_rem, &info_temp);
        optdw = optdw > ((i32)dwork[iwrk] + iwrk) ? optdw : ((i32)dwork[iwrk] + iwrk);
    }

    if (lcmpq2) {
        i32 ldwork_rem = ldwork - iwrk;
        i32 info_temp = 0;
        SLC_DORMRQ("Right", "Transpose", &n, &m, &m, &z[m * ldz], &ldz, &dwork[itau],
                   q2, &ldq2, &dwork[iwrk], &ldwork_rem, &info_temp);
    }

    for (i = 0; i < m - 1; i++) {
        SLC_DSWAP(&m, &z[i * ldz], &int1, &z[(m + i) * ldz], &int1);
        i32 len = m - i - 1;
        SLC_DCOPY(&len, &dbl0, &int0, &z[(i + 1) + i * ldz], &int1);
    }
    SLC_DSWAP(&m, &z[(m - 1) * ldz], &int1, &z[(n - 1) * ldz], &int1);

    for (k = 0; k < m; k++) {
        for (j = k; j < m - 1; j++) {
            SLC_DLARTG(&h[(m + j + 1) + k * ldh], &h[(m + j) + k * ldh], &co, &si, &tmp1);

            h[(m + j + 1) + k * ldh] = tmp1;
            h[(m + j) + k * ldh] = ZERO;
            i32 len = n - k - 1;
            SLC_DROT(&len, &h[(m + j + 1) + (k + 1) * ldh], &ldh, &h[(m + j) + (k + 1) * ldh], &ldh, &co, &si);

            i32 len2 = j + 2;
            SLC_DROT(&len2, &t[(m + j + 1) + m * ldt], &ldt, &t[(m + j) + m * ldt], &ldt, &co, &si);

            if (lcmpq1) {
                SLC_DROT(&n, &q1[(m + j + 1) * ldq1], &int1, &q1[(m + j) * ldq1], &int1, &co, &si);
            }

            SLC_DLARTG(&t[(m + j) + (m + j) * ldt], &t[(m + j) + (m + j + 1) * ldt], &co, &si, &tmp1);

            SLC_DROT(&m, &t[(m + j) * ldt], &int1, &t[(m + j + 1) * ldt], &int1, &co, &si);
            t[(m + j) + (m + j) * ldt] = tmp1;
            t[(m + j) + (m + j + 1) * ldt] = ZERO;
            i32 len3 = m - j - 1;
            SLC_DROT(&len3, &t[(m + j + 1) + (m + j) * ldt], &int1, &t[(m + j + 1) + (m + j + 1) * ldt], &int1, &co, &si);
            i32 len4 = j + 2;
            SLC_DROT(&len4, &t[j * ldt], &int1, &t[(j + 1) * ldt], &int1, &co, &si);

            if (lcmpu1) {
                SLC_DROT(&m, &u11[j * ldu11], &int1, &u11[(j + 1) * ldu11], &int1, &co, &si);
                SLC_DROT(&m, &u12[j * ldu12], &int1, &u12[(j + 1) * ldu12], &int1, &co, &si);
            }

            SLC_DLARTG(&t[j + j * ldt], &t[(j + 1) + j * ldt], &co, &si, &tmp1);

            t[j + j * ldt] = tmp1;
            t[(j + 1) + j * ldt] = ZERO;
            i32 len5 = n - j - 1;
            SLC_DROT(&len5, &t[j + (j + 1) * ldt], &ldt, &t[(j + 1) + (j + 1) * ldt], &ldt, &co, &si);

            i32 len6 = n - k;
            SLC_DROT(&len6, &h[j + k * ldh], &ldh, &h[(j + 1) + k * ldh], &ldh, &co, &si);

            if (lcmpq1) {
                SLC_DROT(&n, &q1[j * ldq1], &int1, &q1[(j + 1) * ldq1], &int1, &co, &si);
            }
        }

        SLC_DLARTG(&h[(m - 1) + k * ldh], &h[(n - 1) + k * ldh], &co, &si, &tmp1);

        h[(m - 1) + k * ldh] = tmp1;
        h[(n - 1) + k * ldh] = ZERO;
        i32 len = n - k - 1;
        SLC_DROT(&len, &h[(m - 1) + (k + 1) * ldh], &ldh, &h[(n - 1) + (k + 1) * ldh], &ldh, &co, &si);

        SLC_DROT(&m, &t[(m - 1) + m * ldt], &ldt, &t[(n - 1) + m * ldt], &ldt, &co, &si);
        tmp1 = -si * t[(m - 1) + (m - 1) * ldt];
        t[(m - 1) + (m - 1) * ldt] = co * t[(m - 1) + (m - 1) * ldt];

        if (lcmpq1) {
            SLC_DROT(&n, &q1[(m - 1) * ldq1], &int1, &q1[(n - 1) * ldq1], &int1, &co, &si);
        }

        SLC_DLARTG(&t[(n - 1) + (n - 1) * ldt], &tmp1, &co, &si, &tmp2);

        SLC_DROT(&m, &t[(n - 1) * ldt], &int1, &t[(m - 1) * ldt], &int1, &co, &si);
        t[(n - 1) + (n - 1) * ldt] = tmp2;

        if (lcmpu1) {
            SLC_DROT(&m, &u12[(m - 1) * ldu12], &int1, &u11[(m - 1) * ldu11], &int1, &co, &si);
        }

        for (j = m - 1; j >= k + 1; j--) {
            SLC_DLARTG(&h[(j - 1) + k * ldh], &h[j + k * ldh], &co, &si, &tmp1);

            h[(j - 1) + k * ldh] = tmp1;
            h[j + k * ldh] = ZERO;
            i32 len = n - k - 1;
            SLC_DROT(&len, &h[(j - 1) + (k + 1) * ldh], &ldh, &h[j + (k + 1) * ldh], &ldh, &co, &si);

            i32 len2 = n - j + 1;
            SLC_DROT(&len2, &t[(j - 1) + (j - 1) * ldt], &ldt, &t[j + (j - 1) * ldt], &ldt, &co, &si);

            if (lcmpq1) {
                SLC_DROT(&n, &q1[(j - 1) * ldq1], &int1, &q1[j * ldq1], &int1, &co, &si);
            }

            SLC_DLARTG(&t[j + j * ldt], &t[j + (j - 1) * ldt], &co, &si, &tmp1);

            SLC_DROT(&m, &t[(m + j) * ldt], &int1, &t[(m + j - 1) * ldt], &int1, &co, &si);
            i32 len3 = m - j + 1;
            SLC_DROT(&len3, &t[(m + j - 1) + (m + j) * ldt], &int1, &t[(m + j - 1) + (m + j - 1) * ldt], &int1, &co, &si);
            t[j + j * ldt] = tmp1;
            t[j + (j - 1) * ldt] = ZERO;
            SLC_DROT(&j, &t[j * ldt], &int1, &t[(j - 1) * ldt], &int1, &co, &si);

            if (lcmpu1) {
                SLC_DROT(&m, &u11[j * ldu11], &int1, &u11[(j - 1) * ldu11], &int1, &co, &si);
                SLC_DROT(&m, &u12[j * ldu12], &int1, &u12[(j - 1) * ldu12], &int1, &co, &si);
            }

            SLC_DLARTG(&t[(m + j) + (m + j) * ldt], &t[(m + j - 1) + (m + j) * ldt], &co, &si, &tmp1);

            t[(m + j) + (m + j) * ldt] = tmp1;
            t[(m + j - 1) + (m + j) * ldt] = ZERO;
            SLC_DROT(&j, &t[(m + j) + m * ldt], &ldt, &t[(m + j - 1) + m * ldt], &ldt, &co, &si);

            i32 len4 = n - k;
            SLC_DROT(&len4, &h[(m + j) + k * ldh], &ldh, &h[(m + j - 1) + k * ldh], &ldh, &co, &si);

            if (lcmpq1) {
                SLC_DROT(&n, &q1[(m + j) * ldq1], &int1, &q1[(m + j - 1) * ldq1], &int1, &co, &si);
            }
        }

        for (j = k + 1; j < m - 1; j++) {
            SLC_DLARTG(&h[(m + k) + (j + 1) * ldh], &h[(m + k) + j * ldh], &co, &si, &tmp1);

            SLC_DROT(&m, &h[(j + 1) * ldh], &int1, &h[j * ldh], &int1, &co, &si);
            h[(m + k) + (j + 1) * ldh] = tmp1;
            h[(m + k) + j * ldh] = ZERO;
            i32 len = m - k - 1;
            SLC_DROT(&len, &h[(m + k + 1) + (j + 1) * ldh], &int1, &h[(m + k + 1) + j * ldh], &int1, &co, &si);

            i32 len2 = j + 2;
            SLC_DROT(&len2, &z[(j + 1) * ldz], &int1, &z[j * ldz], &int1, &co, &si);

            if (lcmpq2) {
                SLC_DROT(&n, &q2[(j + 1) * ldq2], &int1, &q2[j * ldq2], &int1, &co, &si);
            }

            SLC_DLARTG(&z[j + j * ldz], &z[(j + 1) + j * ldz], &co, &si, &tmp1);

            z[j + j * ldz] = tmp1;
            z[(j + 1) + j * ldz] = ZERO;
            i32 len3 = n - j - 1;
            SLC_DROT(&len3, &z[j + (j + 1) * ldz], &ldz, &z[(j + 1) + (j + 1) * ldz], &ldz, &co, &si);
            i32 len4 = j + 2;
            SLC_DROT(&len4, &z[(m + j) + m * ldz], &ldz, &z[(m + j + 1) + m * ldz], &ldz, &co, &si);

            if (lcmpu2) {
                SLC_DROT(&m, &u21[j * ldu21], &int1, &u21[(j + 1) * ldu21], &int1, &co, &si);
                SLC_DROT(&m, &u22[j * ldu22], &int1, &u22[(j + 1) * ldu22], &int1, &co, &si);
            }

            SLC_DLARTG(&z[(m + j) + (m + j) * ldz], &z[(m + j) + (m + j + 1) * ldz], &co, &si, &tmp1);

            z[(m + j) + (m + j) * ldz] = tmp1;
            z[(m + j) + (m + j + 1) * ldz] = ZERO;
            SLC_DROT(&m, &z[(m + j) * ldz], &int1, &z[(m + j + 1) * ldz], &int1, &co, &si);
            i32 len5 = m - j - 1;
            SLC_DROT(&len5, &z[(m + j + 1) + (m + j) * ldz], &int1, &z[(m + j + 1) + (m + j + 1) * ldz], &int1, &co, &si);

            SLC_DROT(&m, &h[(m + j) * ldh], &int1, &h[(m + j + 1) * ldh], &int1, &co, &si);
            i32 len6 = m - k;
            SLC_DROT(&len6, &h[(m + k) + (m + j) * ldh], &int1, &h[(m + k) + (m + j + 1) * ldh], &int1, &co, &si);

            if (lcmpq2) {
                SLC_DROT(&n, &q2[(m + j) * ldq2], &int1, &q2[(m + j + 1) * ldq2], &int1, &co, &si);
            }
        }

        if (k < m - 1) {
            SLC_DLARTG(&h[(m + k) + (n - 1) * ldh], &h[(m + k) + (m - 1) * ldh], &co, &si, &tmp1);

            h[(m + k) + (n - 1) * ldh] = tmp1;
            h[(m + k) + (m - 1) * ldh] = ZERO;
            SLC_DROT(&m, &h[(n - 1) * ldh], &int1, &h[(m - 1) * ldh], &int1, &co, &si);
            i32 len = m - k - 1;
            SLC_DROT(&len, &h[(m + k + 1) + (n - 1) * ldh], &int1, &h[(m + k + 1) + (m - 1) * ldh], &int1, &co, &si);

            SLC_DROT(&m, &z[(n - 1) * ldz], &int1, &z[(m - 1) * ldz], &int1, &co, &si);
            tmp1 = -si * z[(n - 1) + (n - 1) * ldz];
            z[(n - 1) + (n - 1) * ldz] = co * z[(n - 1) + (n - 1) * ldz];

            if (lcmpq2) {
                SLC_DROT(&n, &q2[(n - 1) * ldq2], &int1, &q2[(m - 1) * ldq2], &int1, &co, &si);
            }

            SLC_DLARTG(&z[(m - 1) + (m - 1) * ldz], &tmp1, &co, &si, &tmp2);

            SLC_DROT(&m, &z[(m - 1) + m * ldz], &ldz, &z[(n - 1) + m * ldz], &ldz, &co, &si);
            z[(m - 1) + (m - 1) * ldz] = tmp2;

            if (lcmpu2) {
                SLC_DROT(&m, &u21[(m - 1) * ldu21], &int1, &u22[(m - 1) * ldu22], &int1, &co, &si);
            }
        } else {
            SLC_DLARTG(&h[(m - 1) + (m - 1) * ldh], &h[(n - 1) + (m - 1) * ldh], &co, &si, &tmp1);

            h[(m - 1) + (m - 1) * ldh] = tmp1;
            h[(n - 1) + (m - 1) * ldh] = ZERO;
            SLC_DROT(&m, &h[(m - 1) + m * ldh], &ldh, &h[(n - 1) + m * ldh], &ldh, &co, &si);

            SLC_DROT(&m, &t[(m - 1) + m * ldt], &ldt, &t[(n - 1) + m * ldt], &ldt, &co, &si);
            t[(m - 1) + (m - 1) * ldt] = co * t[(m - 1) + (m - 1) * ldt];

            if (lcmpq1) {
                SLC_DROT(&n, &q1[(m - 1) * ldq1], &int1, &q1[(n - 1) * ldq1], &int1, &co, &si);
            }

            f64 neg_si_tm = -si * t[(m - 1) + (m - 1) * ldt];
            SLC_DLARTG(&t[(n - 1) + (n - 1) * ldt], &neg_si_tm, &co, &si, &tmp2);

            SLC_DROT(&m, &t[(n - 1) * ldt], &int1, &t[(m - 1) * ldt], &int1, &co, &si);
            t[(n - 1) + (n - 1) * ldt] = tmp2;

            if (lcmpu1) {
                SLC_DROT(&m, &u12[(m - 1) * ldu12], &int1, &u11[(m - 1) * ldu11], &int1, &co, &si);
            }
        }

        for (j = m - 1; j >= k + 2; j--) {
            SLC_DLARTG(&h[(m + k) + (m + j - 1) * ldh], &h[(m + k) + (m + j) * ldh], &co, &si, &tmp1);

            SLC_DROT(&m, &h[(m + j - 1) * ldh], &int1, &h[(m + j) * ldh], &int1, &co, &si);
            h[(m + k) + (m + j - 1) * ldh] = tmp1;
            h[(m + k) + (m + j) * ldh] = ZERO;
            i32 len = m - k - 1;
            SLC_DROT(&len, &h[(m + k + 1) + (m + j - 1) * ldh], &int1, &h[(m + k + 1) + (m + j) * ldh], &int1, &co, &si);

            SLC_DROT(&m, &z[(m + j - 1) * ldz], &int1, &z[(m + j) * ldz], &int1, &co, &si);
            i32 len2 = m - j + 1;
            SLC_DROT(&len2, &z[(m + j - 1) + (m + j - 1) * ldz], &int1, &z[(m + j - 1) + (m + j) * ldz], &int1, &co, &si);

            if (lcmpq2) {
                SLC_DROT(&n, &q2[(m + j - 1) * ldq2], &int1, &q2[(m + j) * ldq2], &int1, &co, &si);
            }

            SLC_DLARTG(&z[(m + j) + (m + j) * ldz], &z[(m + j - 1) + (m + j) * ldz], &co, &si, &tmp1);

            z[(m + j) + (m + j) * ldz] = tmp1;
            z[(m + j - 1) + (m + j) * ldz] = ZERO;
            SLC_DROT(&j, &z[(m + j) + m * ldz], &ldz, &z[(m + j - 1) + m * ldz], &ldz, &co, &si);
            i32 len3 = n - j + 1;
            SLC_DROT(&len3, &z[j + (j - 1) * ldz], &ldz, &z[(j - 1) + (j - 1) * ldz], &ldz, &co, &si);

            if (lcmpu2) {
                SLC_DROT(&m, &u21[j * ldu21], &int1, &u21[(j - 1) * ldu21], &int1, &co, &si);
                SLC_DROT(&m, &u22[j * ldu22], &int1, &u22[(j - 1) * ldu22], &int1, &co, &si);
            }

            SLC_DLARTG(&z[j + j * ldz], &z[j + (j - 1) * ldz], &co, &si, &tmp1);

            z[j + j * ldz] = tmp1;
            z[j + (j - 1) * ldz] = ZERO;
            SLC_DROT(&j, &z[j * ldz], &int1, &z[(j - 1) * ldz], &int1, &co, &si);

            SLC_DROT(&m, &h[j * ldh], &int1, &h[(j - 1) * ldh], &int1, &co, &si);
            i32 len4 = m - k;
            SLC_DROT(&len4, &h[(m + k) + j * ldh], &int1, &h[(m + k) + (j - 1) * ldh], &int1, &co, &si);

            if (lcmpq2) {
                SLC_DROT(&n, &q2[j * ldq2], &int1, &q2[(j - 1) * ldq2], &int1, &co, &si);
            }
        }
    }

    iq = 0;
    if (ltri || lcmpq1 || lcmpq2 || lcmpu1 || lcmpu2) {
        imat = 6 * mm;
        iwrk = 12 * mm;
    } else {
        imat = 0;
        iwrk = 6 * mm;
    }

    ma02ad("Lower", m, m, &h[m + m * ldh], ldh, &dwork[imat], m);
    if (m > 1) {
        i32 mm1 = m - 1;
        SLC_DCOPY(&mm1, &h[m + (m + 1) * ldh], &(i32){ldh + 1}, &dwork[imat + 1], &(i32){m + 1});
        i32 mm2 = m - 2;
        SLC_DLASET("Lower", &mm2, &mm2, &dbl0, &dbl0, &dwork[imat + 2], &m);
    }
    ma02ad("Lower", m, m, &t[m + m * ldt], ldt, &dwork[imat + mm], m);
    if (m > 1) {
        i32 mm1 = m - 1;
        SLC_DLASET("Lower", &mm1, &mm1, &dbl0, &dbl0, &dwork[imat + mm + 1], &m);
    }
    SLC_DLACPY("Upper", &m, &m, t, &ldt, &dwork[imat + 2 * mm], &m);
    if (m > 1) {
        i32 mm1 = m - 1;
        SLC_DLASET("Lower", &mm1, &mm1, &dbl0, &dbl0, &dwork[imat + 2 * mm + 1], &m);
    }
    SLC_DLACPY("Upper", &m, &m, h, &ldh, &dwork[imat + 3 * mm], &m);
    if (m > 1) {
        i32 mm1 = m - 1;
        SLC_DLASET("Lower", &mm1, &mm1, &dbl0, &dbl0, &dwork[imat + 3 * mm + 1], &m);
    }
    SLC_DLACPY("Upper", &m, &m, z, &ldz, &dwork[imat + 4 * mm], &m);
    if (m > 1) {
        i32 mm1 = m - 1;
        SLC_DLASET("Lower", &mm1, &mm1, &dbl0, &dbl0, &dwork[imat + 4 * mm + 1], &m);
    }
    ma02ad("Lower", m, m, &z[m + m * ldz], ldz, &dwork[imat + 5 * mm], m);
    if (m > 1) {
        i32 mm1 = m - 1;
        SLC_DLASET("Lower", &mm1, &mm1, &dbl0, &dbl0, &dwork[imat + 5 * mm + 1], &m);
    }

    iwork[0] = 1;
    iwork[1] = -1;
    iwork[2] = -1;
    iwork[3] = 1;
    iwork[4] = -1;
    iwork[5] = -1;

    const char *cmpq_str = (ltri || lcmpq1 || lcmpq2 || lcmpu1 || lcmpu2) ? "Initialize" : "No Computation";
    const char *cmpsc_str = ltri ? "Schur Form" : "Eigenvalues Only";

    i32 six = 6;
    i32 ilo = 1;

    i32 ldwork_rem = ldwork - iwrk;
    i32 liwork_rem = liwork - (m + 6);
    i32 mb03bd_info = 0;
    i32 mb03bd_iwarn = 0;
    mb03bd(cmpsc_str, "Careful", cmpq_str, idum, six, m, ilo, ilo, m, iwork,
           &dwork[imat], m, m, &dwork[iq], m, m, alphar, alphai, beta,
           &iwork[6], &iwork[m + 6], liwork_rem, &dwork[iwrk], ldwork_rem, &mb03bd_iwarn, &mb03bd_info);

    if (mb03bd_info > 0) {
        *info = 2;
        return;
    }

    nbeta0 = 0;
    i11 = 0;
    i22 = 0;
    i2x2 = 0;

    i = 0;
    while (i < m) {
        if (ninf > 0) {
            if (beta[i] == ZERO) nbeta0++;
        }
        if (iwork[i + 6] >= 2 * emin && iwork[i + 6] <= 2 * emax) {
            beta[i] = beta[i] / pow(base, HALF * iwork[i + 6]);
            if (beta[i] != ZERO) {
                if (iwork[m + i + 7] < 0) {
                    i22++;
                } else if (iwork[m + i + 7] > 0) {
                    i11++;
                }
                double complex eig = csqrt(alphar[i] + I * alphai[i]);
                alphar[i] = cimag(eig);
                alphai[i] = creal(eig);
                if (alphar[i] < ZERO) alphar[i] = -alphar[i];
                if (alphai[i] < ZERO) alphai[i] = -alphai[i];
                if (alphar[i] != ZERO && alphai[i] != ZERO) {
                    alphar[i + 1] = -alphar[i];
                    alphai[i + 1] = alphai[i];
                    beta[i + 1] = beta[i];
                    i2x2++;
                    i++;
                }
            }
        } else if (iwork[i + 6] < 2 * emin) {
            alphar[i] = ZERO;
            alphai[i] = ZERO;
            i11++;
        } else {
            if (ninf > 0) nbeta0++;
            beta[i] = ZERO;
            i11++;
        }
        i++;
    }

    iwork[0] = i11 + i22;

    l = 0;
    if (ninf > 0) {
        for (j = 0; j < ninf - nbeta0; j++) {
            tmp1 = ZERO;
            tmp2 = ONE;
            p = 0;
            for (i = 0; i < m; i++) {
                if (beta[i] > ZERO) {
                    temp = SLC_DLAPY2(&alphar[i], &alphai[i]);
                    if (temp > tmp1 && tmp2 >= beta[i]) {
                        tmp1 = temp;
                        tmp2 = beta[i];
                        p = i;
                    }
                }
            }
            l++;
            beta[p] = ZERO;
        }

        if (l == iwork[0]) {
            *info = 0;
            i11 = 0;
            i22 = 0;
            iwork[0] = 0;
        }
    }

    for (i = 0; i < 6; i++) {
        dum[i] = dwork[iwrk + 1 + i];
    }

    iw = iwork[0];
    i = 0;
    j = 0;
    l = 6 * (m - 2 * i2x2) + iwrk;
    unrel = false;

    while (i < m) {
        if (j < iw) {
            i32 abs_idx = iwork[m + i + 7] < 0 ? -iwork[m + i + 7] : iwork[m + i + 7];
            unrel = (i == abs_idx);
        }
        if (alphar[i] != ZERO && beta[i] != ZERO && alphai[i] != ZERO) {
            if (unrel) {
                j++;
                iwork[j] = iwork[m + i + 7];
                iwork[iw + j] = l - iwrk;
                unrel = false;
            }
            SLC_DLACPY("Full", &(i32){2}, &(i32){2}, &dwork[imat + (m + 1) * i], &m, &dwork[l], &(i32){2});
            SLC_DLACPY("Full", &(i32){2}, &(i32){2}, &dwork[imat + (m + 1) * i + mm], &m, &dwork[l + 4], &(i32){2});
            SLC_DLACPY("Full", &(i32){2}, &(i32){2}, &dwork[imat + (m + 1) * i + 2 * mm], &m, &dwork[l + 8], &(i32){2});
            SLC_DLACPY("Full", &(i32){2}, &(i32){2}, &dwork[imat + (m + 1) * i + 3 * mm], &m, &dwork[l + 12], &(i32){2});
            SLC_DLACPY("Full", &(i32){2}, &(i32){2}, &dwork[imat + (m + 1) * i + 4 * mm], &m, &dwork[l + 16], &(i32){2});
            SLC_DLACPY("Full", &(i32){2}, &(i32){2}, &dwork[imat + (m + 1) * i + 5 * mm], &m, &dwork[l + 20], &(i32){2});
            l += 24;
            i += 2;
        } else {
            if (unrel) {
                j++;
                iwork[j] = i;
                iwork[iw + j] = iwrk - iwrk;
                unrel = false;
            }
            i32 six_val = 6;
            SLC_DCOPY(&six_val, &dwork[imat + (m + 1) * i], &mm, &dwork[iwrk], &int1);
            iwrk += 6;
            i++;
        }
    }

    iwork[2 * iw + 1] = i11;
    iwork[2 * iw + 2] = i22;
    iwork[2 * iw + 3] = i2x2;

    if (ltri) {
        SLC_DLACPY("Upper", &m, &m, &dwork[imat + 3 * mm], &m, h, &ldh);
        ma02ad("Full", m, m, &dwork[imat], m, &h[m + m * ldh], ldh);
        SLC_DGEMM("Transpose", "No Transpose", &m, &m, &m, &dbl1,
                  &dwork[iq + 3 * mm], &m, &h[m * ldh], &ldh, &dbl0, &dwork[imat], &m);
        SLC_DGEMM("No Transpose", "No Transpose", &m, &m, &m, &dbl1,
                  &dwork[imat], &m, dwork, &m, &dbl0, &h[m * ldh], &ldh);

        SLC_DLACPY("Upper", &m, &m, &dwork[imat + 2 * mm], &m, t, &ldt);
        SLC_DGEMM("Transpose", "No Transpose", &m, &m, &m, &dbl1,
                  &dwork[iq + 3 * mm], &m, &t[m * ldt], &ldt, &dbl0, &dwork[imat], &m);
        SLC_DGEMM("No Transpose", "No Transpose", &m, &m, &m, &dbl1,
                  &dwork[imat], &m, &dwork[iq + 2 * mm], &m, &dbl0, &t[m * ldt], &ldt);
        ma02ad("Upper", m, m, &dwork[imat + mm], m, &t[m + m * ldt], ldt);

        SLC_DLACPY("Upper", &m, &m, &dwork[imat + 4 * mm], &m, z, &ldz);
        SLC_DGEMM("Transpose", "No Transpose", &m, &m, &m, &dbl1,
                  &dwork[iq + 5 * mm], &m, &z[m * ldz], &ldz, &dbl0, &dwork[imat], &m);
        SLC_DGEMM("No Transpose", "No Transpose", &m, &m, &m, &dbl1,
                  &dwork[imat], &m, dwork, &m, &dbl0, &z[m * ldz], &ldz);
        ma02ad("Upper", m, m, &dwork[imat + 5 * mm], m, &z[m + m * ldz], ldz);

        if (lcmpq1) {
            SLC_DGEMM("No Transpose", "No Transpose", &n, &m, &m, &dbl1,
                      q1, &ldq1, &dwork[iq + 3 * mm], &m, &dbl0, &dwork[imat], &n);
            SLC_DLACPY("Full", &n, &m, &dwork[imat], &n, q1, &ldq1);
            SLC_DGEMM("No Transpose", "No Transpose", &n, &m, &m, &dbl1,
                      &q1[m * ldq1], &ldq1, &dwork[iq + mm], &m, &dbl0, &dwork[imat], &n);
            SLC_DLACPY("Full", &n, &m, &dwork[imat], &n, &q1[m * ldq1], &ldq1);
        }

        if (lcmpq2) {
            SLC_DGEMM("No Transpose", "No Transpose", &n, &m, &m, &dbl1,
                      q2, &ldq2, &dwork[iq + 4 * mm], &m, &dbl0, &dwork[imat], &n);
            SLC_DLACPY("Full", &n, &m, &dwork[imat], &n, q2, &ldq2);
            SLC_DGEMM("No Transpose", "No Transpose", &n, &m, &m, &dbl1,
                      &q2[m * ldq2], &ldq2, dwork, &m, &dbl0, &dwork[imat], &n);
            SLC_DLACPY("Full", &n, &m, &dwork[imat], &n, &q2[m * ldq2], &ldq2);
        }

        if (lcmpu1) {
            SLC_DGEMM("No Transpose", "No Transpose", &m, &m, &m, &dbl1,
                      u11, &ldu11, &dwork[iq + 2 * mm], &m, &dbl0, &dwork[imat], &m);
            SLC_DLACPY("Full", &m, &m, &dwork[imat], &m, u11, &ldu11);
            SLC_DGEMM("No Transpose", "No Transpose", &m, &m, &m, &dbl1,
                      u12, &ldu12, &dwork[iq + 2 * mm], &m, &dbl0, &dwork[imat], &m);
            SLC_DLACPY("Full", &m, &m, &dwork[imat], &m, u12, &ldu12);
        }

        if (lcmpu2) {
            SLC_DGEMM("No Transpose", "No Transpose", &m, &m, &m, &dbl1,
                      u21, &ldu21, &dwork[iq + 5 * mm], &m, &dbl0, &dwork[imat], &m);
            SLC_DLACPY("Full", &m, &m, &dwork[imat], &m, u21, &ldu21);
            SLC_DGEMM("No Transpose", "No Transpose", &m, &m, &m, &dbl1,
                      u22, &ldu22, &dwork[iq + 5 * mm], &m, &dbl0, &dwork[imat], &m);
            SLC_DLACPY("Full", &m, &m, &dwork[imat], &m, u22, &ldu22);
        }
    }

    k = 6 * (m - 2 * i2x2) + 24 * i2x2;
    if (k > 0) {
        memmove(&dwork[7], &dwork[6 * mm], (size_t)k * sizeof(f64));
    }
    for (i = 0; i < 6; i++) {
        dwork[1 + i] = dum[i];
    }

    dwork[0] = (f64)optdw;
}
