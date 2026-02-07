/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void mb04yd(const char *jobu, const char *jobv, const i32 m, const i32 n,
            i32 *rank, f64 *theta, f64 *q, f64 *e, f64 *u, const i32 ldu,
            f64 *v, const i32 ldv, bool *inul, const f64 tol, const f64 reltol,
            f64 *dwork, const i32 ldwork, i32 *iwarn, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TEN = 10.0;
    const f64 HNDRD = 100.0;
    const f64 MEIGTH = -0.125;
    const i32 MAXITR = 30;

    i32 p = (m < n) ? m : n;
    *info = 0;
    *iwarn = 0;

    bool ljobui = (jobu[0] == 'I' || jobu[0] == 'i');
    bool ljobvi = (jobv[0] == 'I' || jobv[0] == 'i');
    bool ljobua = ljobui || (jobu[0] == 'U' || jobu[0] == 'u');
    bool ljobva = ljobvi || (jobv[0] == 'U' || jobv[0] == 'u');

    if (!ljobua && !(jobu[0] == 'N' || jobu[0] == 'n')) {
        *info = -1;
    } else if (!ljobva && !(jobv[0] == 'N' || jobv[0] == 'n')) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (*rank > p) {
        *info = -5;
    } else if (*rank < 0 && *theta < ZERO) {
        *info = -6;
    } else if ((!ljobua && ldu < 1) || (ljobua && ldu < (m > 1 ? m : 1))) {
        *info = -10;
    } else if ((!ljobva && ldv < 1) || (ljobva && ldv < (n > 1 ? n : 1))) {
        *info = -12;
    } else if ((ljobua || ljobva) && ldwork < (p > 0 ? 6*p - 5 : 1)) {
        *info = -17;
    } else if (!(ljobua || ljobva) && ldwork < (p > 0 ? 4*p - 3 : 1)) {
        *info = -17;
    }

    if (*info != 0) {
        return;
    }

    if (p == 0) {
        if (*rank >= 0) {
            *theta = ZERO;
        }
        *rank = 0;
        return;
    }

    f64 tolabs = tol;
    f64 tolrel = reltol;
    f64 smax = fabs(q[p - 1]);

    for (i32 j = 0; j < p - 1; j++) {
        f64 aq = fabs(q[j]);
        f64 ae = fabs(e[j]);
        if (aq > smax) smax = aq;
        if (ae > smax) smax = ae;
    }

    f64 safemn = SLC_DLAMCH("Safe minimum");
    f64 eps = SLC_DLAMCH("Epsilon");
    if (tolabs <= ZERO) tolabs = eps * smax;
    f64 base_eps = SLC_DLAMCH("Base") * eps;
    if (tolrel <= base_eps) tolrel = base_eps;

    f64 thresh = TEN;
    f64 tmp = HNDRD;
    if (tmp < thresh) thresh = tmp;
    tmp = pow(eps, MEIGTH);
    if (tmp < thresh) thresh = tmp;
    thresh *= eps;

    f64 smlnum = safemn / eps;
    f64 rmin = sqrt(smlnum);
    f64 rmax = ONE / rmin;
    tmp = ONE / sqrt(sqrt(safemn));
    if (tmp < rmax) rmax = tmp;

    f64 thetac = *theta;

    i32 iascl = 0;
    f64 sigma = 1.0;
    if (smax > ZERO && smax < rmin) {
        iascl = 1;
        sigma = rmin / smax;
    } else if (smax > rmax) {
        iascl = 1;
        sigma = rmax / smax;
    }

    if (iascl == 1) {
        i32 int1 = 1;
        SLC_DSCAL(&p, &sigma, q, &int1);
        i32 pm1 = p - 1;
        SLC_DSCAL(&pm1, &sigma, e, &int1);
        thetac = sigma * (*theta);
        tolabs = sigma * tolabs;
    }

    f64 pivmin = q[p - 1] * q[p - 1];
    dwork[p - 1] = pivmin;

    for (i32 j = 0; j < p - 1; j++) {
        dwork[j] = q[j] * q[j];
        dwork[p + j] = e[j] * e[j];
        if (dwork[j] > pivmin) pivmin = dwork[j];
        if (dwork[p + j] > pivmin) pivmin = dwork[p + j];
    }

    if (pivmin * safemn > safemn) {
        pivmin = pivmin * safemn;
    } else {
        pivmin = safemn;
    }

    if (ljobui) {
        SLC_DLASET("Full", &m, &p, &ZERO, &ONE, u, &ldu);
    }
    if (ljobvi) {
        SLC_DLASET("Full", &n, &p, &ZERO, &ONE, v, &ldv);
    }

    i32 r;
    i32 j_val;
    i32 info1 = 0;

    if (*rank >= 0) {
        j_val = p - *rank;
        mb03md(p, &j_val, &thetac, q, e, dwork, &dwork[p], pivmin,
               tolabs, tolrel, iwarn, &info1);
        *theta = thetac;
        if (iascl == 1) *theta = *theta / sigma;
        if (j_val <= 0) {
            if (iascl == 1) {
                i32 int1 = 1;
                f64 inv_sigma = ONE / sigma;
                SLC_DSCAL(&p, &inv_sigma, q, &int1);
                i32 pm1 = p - 1;
                SLC_DSCAL(&pm1, &inv_sigma, e, &int1);
            }
            *rank = p - j_val;
            return;
        }
        r = p - j_val;
    } else {
        r = p - mb03nd(p, thetac, dwork, &dwork[p], pivmin, &info1);
    }

    *rank = p;
    for (i32 i = 0; i < p; i++) {
        if (inul[i]) (*rank)--;
    }

    i32 k = p;
    i32 oldi = -1;
    i32 oldk = -1;
    i32 iter = 0;
    i32 maxit = MAXITR * p;

    i32 i_idx = 0;
    f64 x = 0.0;
    i32 numeig = 0;

    while (*rank > r && k > 0) {
        while (k > 0 && inul[k - 1]) {
            k--;
        }

        if (k == 0) {
            if (iascl == 1) {
                i32 int1 = 1;
                f64 inv_sigma = ONE / sigma;
                SLC_DSCAL(&p, &inv_sigma, q, &int1);
                i32 pm1 = p - 1;
                SLC_DSCAL(&pm1, &inv_sigma, e, &int1);
            }
            return;
        }

        bool noc12 = true;

        while (iter < maxit && noc12) {
            i_idx = k;
            x = fabs(q[i_idx - 1]);
            f64 shift = x;

            while (i_idx > 1) {
                if (x > tolabs && fabs(e[i_idx - 2]) > tolabs) {
                    i_idx--;
                    x = fabs(q[i_idx - 1]);
                    if (x < shift) shift = x;
                } else {
                    break;
                }
            }

            j_val = k - i_idx + 1;
            numeig = 0;

            if (x <= tolabs || k == i_idx) {
                noc12 = false;
            } else {
                numeig = mb03nd(j_val, thetac, &dwork[i_idx - 1],
                               &dwork[p + i_idx - 1], pivmin, &info1);
                if (numeig >= j_val || numeig <= 0) noc12 = false;
            }

            if (noc12) {
                if (j_val == 2) {
                    f64 sigmn, sigmx, sinr, cosr, sinl, cosl;
                    SLC_DLASV2(&q[i_idx - 1], &e[i_idx - 1], &q[k - 1],
                              &sigmn, &sigmx, &sinr, &cosr, &sinl, &cosl);
                    q[i_idx - 1] = sigmx;
                    q[k - 1] = sigmn;
                    e[i_idx - 1] = ZERO;
                    (*rank)--;
                    inul[k - 1] = true;
                    noc12 = false;

                    if (ljobua) {
                        i32 int1 = 1;
                        SLC_DROT(&m, &u[(i_idx - 1) * ldu], &int1,
                                &u[(k - 1) * ldu], &int1, &cosl, &sinl);
                    }
                    if (ljobva) {
                        i32 int1 = 1;
                        SLC_DROT(&n, &v[(i_idx - 1) * ldv], &int1,
                                &v[(k - 1) * ldv], &int1, &cosr, &sinr);
                    }
                } else {
                    bool qrit;
                    if (i_idx != oldi || k != oldk) {
                        qrit = fabs(q[i_idx - 1]) >= fabs(q[k - 1]);
                    } else {
                        qrit = fabs(q[i_idx - 1]) >= fabs(q[k - 1]);
                    }
                    oldi = i_idx;

                    if (qrit) {
                        if (fabs(e[k - 2]) <= thresh * fabs(q[k - 1])) {
                            e[k - 2] = ZERO;
                        }
                    } else {
                        if (fabs(e[i_idx - 1]) <= thresh * fabs(q[i_idx - 1])) {
                            e[i_idx - 1] = ZERO;
                        }
                    }

                    if (shift > thetac) shift = ZERO;

                    mb04yw(qrit, ljobua, ljobva, m, n, i_idx, k, shift,
                           q, e, u, ldu, v, ldv, &dwork[2*p - 1]);

                    if (qrit) {
                        if (fabs(e[k - 2]) <= tolabs) e[k - 2] = ZERO;
                    } else {
                        if (fabs(e[i_idx - 1]) <= tolabs) e[i_idx - 1] = ZERO;
                    }

                    dwork[k - 1] = q[k - 1] * q[k - 1];
                    for (i32 i1 = i_idx - 1; i1 < k - 1; i1++) {
                        dwork[i1] = q[i1] * q[i1];
                        dwork[p + i1] = e[i1] * e[i1];
                    }

                    iter++;
                }
            }
        }

        if (iter >= maxit) {
            *info = 1;
            break;
        }

        if (x <= tolabs) {
            mb02ny(ljobua, ljobva, m, n, i_idx, k, q, e, u, ldu, v, ldv, &dwork[2*p - 1]);
            inul[i_idx - 1] = true;
            (*rank)--;
        } else {
            if (j_val >= 2) {
                if (numeig == j_val) {
                    for (i32 i1 = i_idx - 1; i1 < k; i1++) {
                        inul[i1] = true;
                    }
                    *rank -= j_val;
                    k -= j_val;
                } else {
                    k = i_idx - 1;
                }
            } else {
                if (x <= (thetac + tolabs)) {
                    inul[i_idx - 1] = true;
                    (*rank)--;
                }
                k--;
            }
            oldk = k;
        }
    }

    if (iascl == 1) {
        i32 int1 = 1;
        f64 inv_sigma = ONE / sigma;
        SLC_DSCAL(&p, &inv_sigma, q, &int1);
        i32 pm1 = p - 1;
        SLC_DSCAL(&pm1, &inv_sigma, e, &int1);
    }
}
