// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>

void mb04dp(const char *job, i32 n, f64 thresh, f64 *a, i32 lda,
            f64 *de, i32 ldde, f64 *c, i32 ldc, f64 *vw, i32 ldvw,
            i32 *ilo, f64 *lscale, f64 *rscale, f64 *dwork,
            i32 *iwarn, i32 *info) {

    const f64 HALF = 0.5;
    const f64 ONE = 1.0;
    const f64 TEN = 10.0;
    const f64 TWO = 2.0;
    const f64 ZERO = 0.0;
    const f64 MXGAIN = 100.0;
    const f64 SCLFAC = 10.0;

    bool evnorm, loop, lperm, lscal, stormn;
    i32 i, icab, iloold, ir, irab, it, iter, ith, j;
    i32 jc, k, kount, ks, kw1, kw2, kw3, kw4, kw5, kw6, kw7;
    i32 lrab, lsfmax, lsfmin, nr, nrp2;
    f64 ab, alpha, basl, beta, cab, cmax, coef, coef2;
    f64 coef5, cor, denom, eps, ew, gamma, gap, minpro;
    f64 minrat, mn, mx, mxcond, mxnorm, mxs, nh, nh0;
    f64 nhs, ns, ns0, nss, pgamma, prod, rab, ratio;
    f64 sfmax, sfmin, sum, t, ta, tc, td, te, th, th0;
    f64 ths, tv, tw;
    i32 int1 = 1;
    f64 neg_one = -ONE;

    *info = 0;
    *iwarn = 0;
    lperm = (job[0] == 'P' || job[0] == 'p' || job[0] == 'B' || job[0] == 'b');
    lscal = (job[0] == 'S' || job[0] == 's' || job[0] == 'B' || job[0] == 'b');

    if (!lperm && !lscal && job[0] != 'N' && job[0] != 'n') {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -5;
    } else if (ldde < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldc < (n > 1 ? n : 1)) {
        *info = -9;
    } else if (ldvw < (n > 1 ? n : 1)) {
        *info = -11;
    }
    if (*info != 0) {
        return;
    }

    *ilo = 1;

    if (n == 0) {
        return;
    }

    if ((!lperm && !lscal) || n == 1) {
        for (i = 0; i < n; i++) {
            lscale[i] = ONE;
            rscale[i] = ONE;
        }
        if (n == 1 && lscal) {
            ns0 = ma02id("skew-Hamiltonian", "1-norm", n, a, lda, de, ldde, dwork);
            nh0 = ma02id("Hamiltonian", "1-norm", n, c, ldc, vw, ldvw, dwork);
            dwork[0] = ns0;
            dwork[1] = nh0;
            dwork[2] = ns0;
            dwork[3] = nh0;
            dwork[4] = thresh;
        }
        return;
    }

    if (lperm) {
        iloold = 0;

        while (*ilo != iloold) {
            iloold = *ilo;

            i = *ilo - 1;
            while (i < n && *ilo == iloold) {
                bool skip = false;
                for (j = *ilo - 1; j < i && !skip; j++) {
                    if (a[j + i * lda] != ZERO || c[j + i * ldc] != ZERO) {
                        skip = true;
                    }
                }
                if (!skip) {
                    for (j = i + 1; j < n && !skip; j++) {
                        if (a[j + i * lda] != ZERO || c[j + i * ldc] != ZERO) {
                            skip = true;
                        }
                    }
                }
                if (!skip) {
                    for (j = *ilo - 1; j < i && !skip; j++) {
                        if (de[i + j * ldde] != ZERO || vw[i + j * ldvw] != ZERO) {
                            skip = true;
                        }
                    }
                }
                if (!skip) {
                    if (vw[i + i * ldvw] != ZERO) {
                        skip = true;
                    }
                }
                if (!skip) {
                    for (j = i + 1; j < n && !skip; j++) {
                        if (de[j + i * ldde] != ZERO || vw[j + i * ldvw] != ZERO) {
                            skip = true;
                        }
                    }
                }

                if (skip) {
                    i++;
                    continue;
                }

                lscale[*ilo - 1] = (f64)(i + 1);
                rscale[*ilo - 1] = (f64)(i + 1);

                if (*ilo - 1 != i) {
                    i32 ilo_idx = *ilo - 1;
                    SLC_DSWAP(&n, &a[ilo_idx * lda], &int1, &a[i * lda], &int1);
                    i32 len = n - ilo_idx;
                    SLC_DSWAP(&len, &a[ilo_idx + ilo_idx * lda], &lda, &a[i + ilo_idx * lda], &lda);

                    if (i < n - 1) {
                        len = n - i - 1;
                        SLC_DSWAP(&len, &de[(i + 1) + ilo_idx * ldde], &int1, &de[(i + 1) + i * ldde], &int1);
                    }
                    if (i > ilo_idx + 1) {
                        len = i - ilo_idx - 1;
                        SLC_DSCAL(&len, &neg_one, &de[(ilo_idx + 1) + ilo_idx * ldde], &int1);
                        SLC_DSWAP(&len, &de[(ilo_idx + 1) + ilo_idx * ldde], &int1, &de[i + (ilo_idx + 1) * ldde], &ldde);
                    }

                    len = ilo_idx;
                    SLC_DSWAP(&len, &de[(ilo_idx + 1) * ldde], &int1, &de[(i + 1) * ldde], &int1);
                    if (n > i + 1) {
                        len = n - i - 1;
                        SLC_DSWAP(&len, &de[i + (i + 2) * ldde], &ldde, &de[ilo_idx + (i + 2) * ldde], &ldde);
                    }
                    if (i > ilo_idx + 1) {
                        len = i - ilo_idx - 1;
                        SLC_DSCAL(&len, &neg_one, &de[(ilo_idx + 1) + (i + 1) * ldde], &int1);
                        SLC_DSWAP(&len, &de[ilo_idx + (ilo_idx + 2) * ldde], &ldde, &de[(ilo_idx + 1) + (i + 1) * ldde], &int1);
                    }
                    len = i - ilo_idx;
                    SLC_DSCAL(&len, &neg_one, &de[ilo_idx + (i + 1) * ldde], &int1);

                    SLC_DSWAP(&n, &c[ilo_idx * ldc], &int1, &c[i * ldc], &int1);
                    len = n - ilo_idx;
                    SLC_DSWAP(&len, &c[ilo_idx + ilo_idx * ldc], &ldc, &c[i + ilo_idx * ldc], &ldc);

                    t = vw[i + ilo_idx * ldvw];
                    vw[i + ilo_idx * ldvw] = vw[ilo_idx + ilo_idx * ldvw];
                    vw[ilo_idx + ilo_idx * ldvw] = t;
                    len = n - i;
                    SLC_DSWAP(&len, &vw[i + ilo_idx * ldvw], &int1, &vw[i + i * ldvw], &int1);
                    len = i - ilo_idx;
                    SLC_DSWAP(&len, &vw[ilo_idx + ilo_idx * ldvw], &int1, &vw[i + ilo_idx * ldvw], &ldvw);

                    len = ilo_idx + 1;
                    SLC_DSWAP(&len, &vw[(ilo_idx + 1) * ldvw], &int1, &vw[(i + 1) * ldvw], &int1);
                    len = n - i;
                    SLC_DSWAP(&len, &vw[i + (i + 1) * ldvw], &ldvw, &vw[ilo_idx + (i + 1) * ldvw], &ldvw);
                    len = i - ilo_idx;
                    SLC_DSWAP(&len, &vw[ilo_idx + (ilo_idx + 1) * ldvw], &ldvw, &vw[ilo_idx + (i + 1) * ldvw], &int1);
                }
                (*ilo)++;
                i = *ilo - 1;
                continue;
            }

            i = *ilo - 1;
            while (i < n && *ilo == iloold) {
                bool skip = false;
                for (j = *ilo - 1; j < i && !skip; j++) {
                    if (a[i + j * lda] != ZERO || c[i + j * ldc] != ZERO) {
                        skip = true;
                    }
                }
                if (!skip) {
                    for (j = i + 1; j < n && !skip; j++) {
                        if (a[i + j * lda] != ZERO || c[i + j * ldc] != ZERO) {
                            skip = true;
                        }
                    }
                }
                if (!skip) {
                    for (j = *ilo - 1; j < i && !skip; j++) {
                        if (de[j + (i + 1) * ldde] != ZERO || vw[j + (i + 1) * ldvw] != ZERO) {
                            skip = true;
                        }
                    }
                }
                if (!skip) {
                    if (vw[i + (i + 1) * ldvw] != ZERO) {
                        skip = true;
                    }
                }
                if (!skip) {
                    for (j = i + 1; j < n && !skip; j++) {
                        if (de[i + (j + 1) * ldde] != ZERO || vw[i + (j + 1) * ldvw] != ZERO) {
                            skip = true;
                        }
                    }
                }

                if (skip) {
                    i++;
                    continue;
                }

                lscale[*ilo - 1] = (f64)(n + i + 1);
                rscale[*ilo - 1] = (f64)(n + i + 1);

                i32 len = i - (*ilo - 1);
                SLC_DSWAP(&len, &a[i + (*ilo - 1) * lda], &lda, &de[i + (*ilo - 1) * ldde], &ldde);
                SLC_DSCAL(&len, &neg_one, &a[i + (*ilo - 1) * lda], &lda);
                if (n > i + 1) {
                    len = n - i - 1;
                    SLC_DSWAP(&len, &a[i + (i + 1) * lda], &lda, &de[(i + 1) + i * ldde], &int1);
                    SLC_DSCAL(&len, &neg_one, &de[(i + 1) + i * ldde], &int1);
                }
                len = i;
                SLC_DSWAP(&len, &a[i * lda], &int1, &de[(i + 1) * ldde], &int1);
                SLC_DSCAL(&len, &neg_one, &a[i * lda], &int1);
                if (n > i + 1) {
                    len = n - i - 1;
                    SLC_DSCAL(&len, &neg_one, &a[(i + 1) + i * lda], &int1);
                    SLC_DSWAP(&len, &a[(i + 1) + i * lda], &int1, &de[i + (i + 2) * ldde], &ldde);
                }

                len = i - (*ilo - 1);
                SLC_DSWAP(&len, &c[i + (*ilo - 1) * ldc], &ldc, &vw[i + (*ilo - 1) * ldvw], &ldvw);
                SLC_DSCAL(&len, &neg_one, &c[i + (*ilo - 1) * ldc], &ldc);
                if (n > i + 1) {
                    len = n - i - 1;
                    SLC_DSWAP(&len, &c[i + (i + 1) * ldc], &ldc, &vw[(i + 1) + i * ldvw], &int1);
                    SLC_DSCAL(&len, &neg_one, &c[i + (i + 1) * ldc], &ldc);
                }
                len = i;
                SLC_DSWAP(&len, &c[i * ldc], &int1, &vw[(i + 1) * ldvw], &int1);
                SLC_DSCAL(&len, &neg_one, &c[i * ldc], &int1);
                if (n > i + 1) {
                    len = n - i - 1;
                    SLC_DSWAP(&len, &c[(i + 1) + i * ldc], &int1, &vw[i + (i + 2) * ldvw], &ldvw);
                    SLC_DSCAL(&len, &neg_one, &c[(i + 1) + i * ldc], &int1);
                }
                c[i + i * ldc] = -c[i + i * ldc];
                t = vw[i + i * ldvw];
                vw[i + i * ldvw] = -vw[i + (i + 1) * ldvw];
                vw[i + (i + 1) * ldvw] = -t;

                if (*ilo - 1 != i) {
                    i32 ilo_idx = *ilo - 1;
                    SLC_DSWAP(&n, &a[ilo_idx * lda], &int1, &a[i * lda], &int1);
                    len = n - ilo_idx;
                    SLC_DSWAP(&len, &a[ilo_idx + ilo_idx * lda], &lda, &a[i + ilo_idx * lda], &lda);

                    if (i < n - 1) {
                        len = n - i - 1;
                        SLC_DSWAP(&len, &de[(i + 1) + ilo_idx * ldde], &int1, &de[(i + 1) + i * ldde], &int1);
                    }
                    if (i > ilo_idx + 1) {
                        len = i - ilo_idx - 1;
                        SLC_DSCAL(&len, &neg_one, &de[(ilo_idx + 1) + ilo_idx * ldde], &int1);
                        SLC_DSWAP(&len, &de[(ilo_idx + 1) + ilo_idx * ldde], &int1, &de[i + (ilo_idx + 1) * ldde], &ldde);
                    }

                    len = ilo_idx;
                    SLC_DSWAP(&len, &de[(ilo_idx + 1) * ldde], &int1, &de[(i + 1) * ldde], &int1);
                    if (n > i + 1) {
                        len = n - i - 1;
                        SLC_DSWAP(&len, &de[i + (i + 2) * ldde], &ldde, &de[ilo_idx + (i + 2) * ldde], &ldde);
                    }
                    if (i > ilo_idx + 1) {
                        len = i - ilo_idx - 1;
                        SLC_DSCAL(&len, &neg_one, &de[(ilo_idx + 1) + (i + 1) * ldde], &int1);
                        SLC_DSWAP(&len, &de[ilo_idx + (ilo_idx + 2) * ldde], &ldde, &de[(ilo_idx + 1) + (i + 1) * ldde], &int1);
                    }
                    len = i - ilo_idx;
                    SLC_DSCAL(&len, &neg_one, &de[ilo_idx + (i + 1) * ldde], &int1);

                    SLC_DSWAP(&n, &c[ilo_idx * ldc], &int1, &c[i * ldc], &int1);
                    len = n - ilo_idx;
                    SLC_DSWAP(&len, &c[ilo_idx + ilo_idx * ldc], &ldc, &c[i + ilo_idx * ldc], &ldc);

                    t = vw[i + ilo_idx * ldvw];
                    vw[i + ilo_idx * ldvw] = vw[ilo_idx + ilo_idx * ldvw];
                    vw[ilo_idx + ilo_idx * ldvw] = t;
                    len = n - i;
                    SLC_DSWAP(&len, &vw[i + ilo_idx * ldvw], &int1, &vw[i + i * ldvw], &int1);
                    len = i - ilo_idx;
                    SLC_DSWAP(&len, &vw[ilo_idx + ilo_idx * ldvw], &int1, &vw[i + ilo_idx * ldvw], &ldvw);

                    len = ilo_idx + 1;
                    SLC_DSWAP(&len, &vw[(ilo_idx + 1) * ldvw], &int1, &vw[(i + 1) * ldvw], &int1);
                    len = n - i;
                    SLC_DSWAP(&len, &vw[i + (i + 1) * ldvw], &ldvw, &vw[ilo_idx + (i + 1) * ldvw], &ldvw);
                    len = i - ilo_idx;
                    SLC_DSWAP(&len, &vw[ilo_idx + (ilo_idx + 1) * ldvw], &ldvw, &vw[ilo_idx + (i + 1) * ldvw], &int1);
                }
                (*ilo)++;
                i = *ilo - 1;
                continue;
            }
        }

        for (i = *ilo - 1; i < n; i++) {
            lscale[i] = ONE;
            rscale[i] = ONE;
        }
        if (!lscal) {
            return;
        }
    }

    nr = n - *ilo + 1;

    ns0 = ma02id("skew-Hamiltonian", "1-norm", nr, &a[(*ilo - 1) + (*ilo - 1) * lda], lda,
                 &de[(*ilo - 1) + (*ilo - 1) * ldde], ldde, dwork);
    nh0 = ma02id("Hamiltonian", "1-norm", nr, &c[(*ilo - 1) + (*ilo - 1) * ldc], ldc,
                 &vw[(*ilo - 1) + (*ilo - 1) * ldvw], ldvw, dwork);

    if (*ilo == n) {
        dwork[0] = ns0;
        dwork[1] = nh0;
        dwork[2] = ns0;
        dwork[3] = nh0;
        dwork[4] = thresh;
        return;
    }

    kw1 = n;
    kw2 = kw1 + n;
    kw3 = kw2 + n;
    kw4 = kw3 + n;
    kw5 = kw4 + n;

    sfmin = SLC_DLAMCH("Safe minimum");
    sfmax = ONE / sfmin;
    basl = log10(SCLFAC);
    lsfmin = (i32)(log10(sfmin) / basl + ONE);
    lsfmax = (i32)(log10(sfmax) / basl);
    mxnorm = (ns0 > nh0) ? ns0 : nh0;
    loop = thresh < ZERO;

    if (loop) {
        ns = ns0;
        nss = ns0;
        nh = nh0;
        nhs = nh0;

        ith = (i32)thresh;
        mxs = mxnorm;
        mx = ZERO;
        mn = sfmax;
        if (ith >= -2) {
            if (ns < nh) {
                ratio = (nh / ns < sfmax) ? nh / ns : sfmax;
            } else {
                ratio = (ns / nh < sfmax) ? ns / nh : sfmax;
            }
            minrat = ratio;
        } else if (ith <= -10) {
            mxcond = -thresh;
        } else {
            denom = (ONE > mxnorm) ? ONE : mxnorm;
            prod = (ns / denom) * (nh / denom);
            minpro = prod;
        }
        stormn = false;
        evnorm = false;

        for (j = *ilo - 1; j < n; j++) {
            for (i = *ilo - 1; i < n; i++) {
                if (i != j) {
                    ab = fabs(a[i + j * lda]);
                    if (ab != ZERO) mn = (mn < ab) ? mn : ab;
                    mx = (mx > ab) ? mx : ab;
                }
            }
        }

        for (j = *ilo - 1; j < n; j++) {
            for (i = *ilo - 1; i < n; i++) {
                if (i != j) {
                    ab = fabs(de[i + j * ldde]);
                    if (ab != ZERO) mn = (mn < ab) ? mn : ab;
                    mx = (mx > ab) ? mx : ab;
                    ab = fabs(de[i + (j + 1) * ldde]);
                    if (ab != ZERO) mn = (mn < ab) ? mn : ab;
                    mx = (mx > ab) ? mx : ab;
                }
            }
        }

        for (j = *ilo - 1; j < n; j++) {
            for (i = *ilo - 1; i < n; i++) {
                if (i != j) {
                    ab = fabs(c[i + j * ldc]);
                    if (ab != ZERO) mn = (mn < ab) ? mn : ab;
                    mx = (mx > ab) ? mx : ab;
                }
            }
        }

        for (j = *ilo - 1; j < n; j++) {
            for (i = *ilo - 1; i < n; i++) {
                ab = fabs(vw[i + j * ldvw]);
                if (ab != ZERO) mn = (mn < ab) ? mn : ab;
                mx = (mx > ab) ? mx : ab;
                ab = fabs(vw[i + (j + 1) * ldvw]);
                if (ab != ZERO) mn = (mn < ab) ? mn : ab;
                mx = (mx > ab) ? mx : ab;
            }
        }

        if (mx * sfmin <= mn) {
            gap = mx / mn;
        } else {
            gap = sfmax;
        }
        eps = SLC_DLAMCH("Precision");
        iter = (i32)log10(gap);
        i32 ieps = -(i32)log10(eps);
        iter = (iter < ieps) ? iter : ieps;
        iter++;
        th = ((mn > mx * eps) ? mn : mx * eps) / ((mxnorm > sfmin) ? mxnorm : sfmin);
        ths = th;
        kw6 = kw5 + n + *ilo;
        kw7 = kw6 + n;
        SLC_DCOPY(&nr, &lscale[*ilo - 1], &int1, &dwork[kw6], &int1);
        SLC_DCOPY(&nr, &rscale[*ilo - 1], &int1, &dwork[kw7], &int1);

        if (ith > -10) {
            mxcond = ONE / sqrt(eps);
        }
    } else {
        th = mxnorm * thresh;
        iter = 1;
        evnorm = true;
    }
    th0 = th;

    coef = HALF / (f64)(2 * nr);
    coef2 = coef * coef;
    coef5 = HALF * coef2;
    nrp2 = nr + 2;
    beta = ZERO;

    for (k = 0; k < iter; k++) {
        for (i = 0; i < 6 * n; i++) dwork[i] = ZERO;
        for (i = 0; i < nr; i++) {
            lscale[*ilo - 1 + i] = ZERO;
            rscale[*ilo - 1 + i] = ZERO;
        }

        for (i = *ilo - 1; i < n; i++) {
            for (j = *ilo - 1; j < n; j++) {
                ta = fabs(a[i + j * lda]);
                tc = fabs(c[i + j * ldc]);
                if (j > i) {
                    td = fabs(de[i + (j + 1) * ldde]);
                } else if (j < i) {
                    td = fabs(de[j + (i + 1) * ldde]);
                } else {
                    td = ZERO;
                }
                if (j > i) {
                    te = fabs(de[j + i * ldde]);
                } else if (j < i) {
                    te = fabs(de[i + j * ldde]);
                } else {
                    te = ZERO;
                }
                if (j > i) {
                    tv = fabs(vw[i + (j + 1) * ldvw]);
                } else {
                    tv = fabs(vw[j + (i + 1) * ldvw]);
                }
                if (j > i) {
                    tw = fabs(vw[j + i * ldvw]);
                } else {
                    tw = fabs(vw[i + j * ldvw]);
                }
                if (ta > th) {
                    ta = log10(ta) / basl;
                } else {
                    ta = ZERO;
                }
                if (tc > th) {
                    tc = log10(tc) / basl;
                } else {
                    tc = ZERO;
                }
                if (td > th) {
                    td = log10(td) / basl;
                } else {
                    td = ZERO;
                }
                if (te > th) {
                    te = log10(te) / basl;
                } else {
                    te = ZERO;
                }
                if (tv > th) {
                    tv = log10(tv) / basl;
                } else {
                    tv = ZERO;
                }
                if (tw > th) {
                    tw = log10(tw) / basl;
                } else {
                    tw = ZERO;
                }
                dwork[i + kw4] = dwork[i + kw4] - ta - tc - td - tv;
                dwork[j + kw5] = dwork[j + kw5] - ta - tc - te - tw;
            }
        }

        it = 1;

        do {
            gamma = ZERO;
            for (i = *ilo - 1; i < n; i++) {
                gamma += dwork[i + kw4] * dwork[i + kw4];
                gamma += dwork[i + kw5] * dwork[i + kw5];
            }
            gamma *= TWO;

            ew = ZERO;
            for (i = *ilo - 1; i < n; i++) {
                ew += dwork[i + kw4] + dwork[i + kw5];
            }

            gamma = coef * gamma - TWO * coef2 * ew * ew;
            if (gamma == ZERO) break;
            if (it != 1) beta = gamma / pgamma;
            t = -TWO * coef5 * ew;

            SLC_DSCAL(&nr, &beta, &dwork[*ilo - 1], &int1);
            SLC_DSCAL(&nr, &beta, &dwork[*ilo - 1 + kw1], &int1);

            f64 coef_val = coef;
            SLC_DAXPY(&nr, &coef_val, &dwork[*ilo - 1 + kw4], &int1, &dwork[*ilo - 1 + kw1], &int1);
            SLC_DAXPY(&nr, &coef_val, &dwork[*ilo - 1 + kw5], &int1, &dwork[*ilo - 1], &int1);

            for (j = *ilo - 1; j < n; j++) {
                dwork[j] = dwork[j] + t;
                dwork[j + kw1] = dwork[j + kw1] + t;
            }

            for (i = *ilo - 1; i < n; i++) {
                kount = 0;
                sum = ZERO;
                for (j = *ilo - 1; j < n; j++) {
                    ks = kount;
                    if (a[i + j * lda] != ZERO) kount++;
                    if (c[i + j * ldc] != ZERO) kount++;
                    sum += (f64)(kount - ks) * dwork[j];

                    ks = kount;
                    if (j > i) {
                        if (de[i + (j + 1) * ldde] != ZERO) kount++;
                    } else if (j < i) {
                        if (de[j + (i + 1) * ldde] != ZERO) kount++;
                    }
                    if (j >= i) {
                        if (vw[i + (j + 1) * ldvw] != ZERO) kount++;
                    } else {
                        if (vw[j + (i + 1) * ldvw] != ZERO) kount++;
                    }
                    sum += (f64)(kount - ks) * dwork[j + kw1];
                }
                dwork[i + kw2] = (f64)kount * dwork[i + kw1] + sum;
            }

            for (j = *ilo - 1; j < n; j++) {
                kount = 0;
                sum = ZERO;
                for (i = *ilo - 1; i < n; i++) {
                    ks = kount;
                    if (a[i + j * lda] != ZERO) kount++;
                    if (c[i + j * ldc] != ZERO) kount++;
                    sum += (f64)(kount - ks) * dwork[i + kw1];

                    ks = kount;
                    if (j > i) {
                        if (de[j + i * ldde] != ZERO) kount++;
                    } else if (j < i) {
                        if (de[i + j * ldde] != ZERO) kount++;
                    }
                    if (j >= i) {
                        if (vw[j + i * ldvw] != ZERO) kount++;
                    } else {
                        if (vw[i + j * ldvw] != ZERO) kount++;
                    }
                    sum += (f64)(kount - ks) * dwork[i];
                }
                dwork[j + kw3] = (f64)kount * dwork[j] + sum;
            }

            sum = ZERO;
            for (i = *ilo - 1; i < n; i++) {
                sum += dwork[i + kw1] * dwork[i + kw2];
                sum += dwork[i] * dwork[i + kw3];
            }
            sum *= TWO;
            alpha = gamma / sum;

            cmax = ZERO;
            for (i = *ilo - 1; i < n; i++) {
                cor = alpha * dwork[i + kw1];
                if (fabs(cor) > cmax) cmax = fabs(cor);
                lscale[i] = lscale[i] + cor;
                cor = alpha * dwork[i];
                if (fabs(cor) > cmax) cmax = fabs(cor);
                rscale[i] = rscale[i] + cor;
            }

            if (cmax >= HALF) {
                f64 neg_alpha = -alpha;
                SLC_DAXPY(&n, &neg_alpha, &dwork[*ilo - 1 + kw2], &int1, &dwork[*ilo - 1 + kw4], &int1);
                SLC_DAXPY(&n, &neg_alpha, &dwork[*ilo - 1 + kw3], &int1, &dwork[*ilo - 1 + kw5], &int1);

                pgamma = gamma;
                it++;
                if (it > nrp2) break;
            } else {
                break;
            }
        } while (true);

        for (i = *ilo - 1; i < n; i++) {
            irab = SLC_IDAMAX(&nr, &a[i + (*ilo - 1) * lda], &lda);
            rab = fabs(a[i + (*ilo - 1 + irab - 1) * lda]);
            irab = SLC_IDAMAX(&nr, &c[i + (*ilo - 1) * ldc], &ldc);
            f64 tmp = fabs(c[i + (*ilo - 1 + irab - 1) * ldc]);
            rab = (rab > tmp) ? rab : tmp;
            if (i > *ilo - 1) {
                i32 len = i;
                irab = SLC_IDAMAX(&len, &de[(i + 1) * ldde], &int1);
                tmp = fabs(de[irab - 1 + (i + 1) * ldde]);
                rab = (rab > tmp) ? rab : tmp;
            }
            if (n > i + 1) {
                i32 len = n - i - 1;
                irab = SLC_IDAMAX(&len, &de[i + (i + 2) * ldde], &ldde);
                tmp = fabs(de[i + (i + irab + 1) * ldde]);
                rab = (rab > tmp) ? rab : tmp;
            }
            i32 len_i = i + 1;
            irab = SLC_IDAMAX(&len_i, &vw[(i + 1) * ldvw], &int1);
            tmp = fabs(vw[irab - 1 + (i + 1) * ldvw]);
            rab = (rab > tmp) ? rab : tmp;
            if (n > i + 2) {
                i32 len = n - i - 2;
                irab = SLC_IDAMAX(&len, &vw[i + (i + 2) * ldvw], &ldvw);
                tmp = fabs(vw[i + (i + irab + 1) * ldvw]);
                rab = (rab > tmp) ? rab : tmp;
            }

            lrab = (i32)(log10(rab + sfmin) / basl + ONE);
            ir = (i32)(lscale[i] + (lscale[i] >= 0 ? HALF : -HALF));
            ir = (ir > lsfmin) ? ir : lsfmin;
            ir = (ir < lsfmax) ? ir : lsfmax;
            ir = (ir < lsfmax - lrab) ? ir : lsfmax - lrab;
            lscale[i] = pow(SCLFAC, (f64)ir);

            icab = SLC_IDAMAX(&n, &a[i * lda], &int1);
            cab = fabs(a[icab - 1 + i * lda]);
            icab = SLC_IDAMAX(&n, &c[i * ldc], &int1);
            tmp = fabs(c[icab - 1 + i * ldc]);
            cab = (cab > tmp) ? cab : tmp;
            if (i > 0) {
                i32 len = i;
                icab = SLC_IDAMAX(&len, &de[i], &ldde);
                tmp = fabs(de[i + (icab - 1) * ldde]);
                cab = (cab > tmp) ? cab : tmp;
            }
            if (n > i + 1) {
                i32 len = n - i - 1;
                icab = SLC_IDAMAX(&len, &de[(i + 1) + i * ldde], &int1);
                tmp = fabs(de[i + icab + i * ldde]);
                cab = (cab > tmp) ? cab : tmp;
            }
            len_i = i + 1;
            icab = SLC_IDAMAX(&len_i, &vw[i], &ldvw);
            tmp = fabs(vw[i + (icab - 1) * ldvw]);
            cab = (cab > tmp) ? cab : tmp;
            if (n > i + 1) {
                i32 len = n - i - 1;
                icab = SLC_IDAMAX(&len, &vw[(i + 1) + i * ldvw], &int1);
                tmp = fabs(vw[i + icab + i * ldvw]);
                cab = (cab > tmp) ? cab : tmp;
            }

            lrab = (i32)(log10(cab + sfmin) / basl + ONE);
            jc = (i32)(rscale[i] + (rscale[i] >= 0 ? HALF : -HALF));
            jc = (jc > lsfmin) ? jc : lsfmin;
            jc = (jc < lsfmax) ? jc : lsfmax;
            jc = (jc < lsfmax - lrab) ? jc : lsfmax - lrab;
            rscale[i] = pow(SCLFAC, (f64)jc);
        }

        bool all_one = true;
        for (i = *ilo - 1; i < n && all_one; i++) {
            if (lscale[i] != ONE || rscale[i] != ONE) {
                all_one = false;
            }
        }

        if (all_one) {
            nss = ns0;
            nhs = nh0;
            ths = th0;
            goto label510;
        }

        if (loop) {
            if (ith <= -10) {
                ir = SLC_IDAMAX(&nr, &lscale[*ilo - 1], &int1);
                jc = SLC_IDAMAX(&nr, &rscale[*ilo - 1], &int1);
                t = (lscale[*ilo - 1 + ir - 1] > rscale[*ilo - 1 + jc - 1]) ?
                    lscale[*ilo - 1 + ir - 1] : rscale[*ilo - 1 + jc - 1];
                mn = t;
                for (i = *ilo - 1; i < n; i++) {
                    if (lscale[i] < mn) mn = lscale[i];
                }
                for (i = *ilo - 1; i < n; i++) {
                    if (rscale[i] < mn) mn = rscale[i];
                }
                t = mn / t;
                if (t < ONE / mxcond) {
                    th = th * TEN;
                    continue;
                } else {
                    ths = th;
                    evnorm = true;
                    goto label480;
                }
            }

            ns = ZERO;
            for (j = *ilo - 1; j < n; j++) {
                t = ZERO;
                for (i = *ilo - 1; i < n; i++) {
                    t += fabs(a[i + j * lda]) * lscale[i] * rscale[j];
                    if (i < j) {
                        t += fabs(de[j + i * ldde]) * rscale[i] * rscale[j];
                    } else if (i > j) {
                        t += fabs(de[i + j * ldde]) * rscale[i] * rscale[j];
                    }
                }
                if (t > ns) ns = t;
            }

            for (j = *ilo - 1; j < n; j++) {
                t = ZERO;
                for (i = *ilo - 1; i < n; i++) {
                    t += fabs(a[j + i * lda]) * lscale[j] * rscale[i];
                    if (i < j) {
                        t += fabs(de[i + (j + 1) * ldde]) * lscale[i] * lscale[j];
                    } else if (i > j) {
                        t += fabs(de[j + (i + 1) * ldde]) * lscale[i] * lscale[j];
                    }
                }
                if (t > ns) ns = t;
            }

            nh = ZERO;
            for (j = *ilo - 1; j < n; j++) {
                t = ZERO;
                for (i = *ilo - 1; i < n; i++) {
                    t += fabs(c[i + j * ldc]) * lscale[i] * rscale[j];
                    if (i <= j) {
                        t += fabs(vw[j + i * ldvw]) * rscale[i] * rscale[j];
                    } else {
                        t += fabs(vw[i + j * ldvw]) * rscale[i] * rscale[j];
                    }
                }
                if (t > nh) nh = t;
            }

            for (j = *ilo - 1; j < n; j++) {
                t = ZERO;
                for (i = *ilo - 1; i < n; i++) {
                    t += fabs(c[j + i * ldc]) * lscale[j] * rscale[i];
                    if (i <= j) {
                        t += fabs(vw[i + (j + 1) * ldvw]) * lscale[i] * lscale[j];
                    } else {
                        t += fabs(vw[j + (i + 1) * ldvw]) * lscale[i] * lscale[j];
                    }
                }
                if (t > nh) nh = t;
            }

            if (ith >= -4 && ith < -2) {
                prod = (ns / denom) * (nh / denom);
                if (minpro > prod) {
                    minpro = prod;
                    stormn = true;
                    SLC_DCOPY(&nr, &lscale[*ilo - 1], &int1, &dwork[kw6], &int1);
                    SLC_DCOPY(&nr, &rscale[*ilo - 1], &int1, &dwork[kw7], &int1);
                    nss = ns;
                    nhs = nh;
                    ths = th;
                }
            } else if (ith >= -2) {
                if (ns < nh) {
                    ratio = (nh / ns < sfmax) ? nh / ns : sfmax;
                } else {
                    ratio = (ns / nh < sfmax) ? ns / nh : sfmax;
                }
                if (minrat > ratio) {
                    minrat = ratio;
                    stormn = true;
                    SLC_DCOPY(&nr, &lscale[*ilo - 1], &int1, &dwork[kw6], &int1);
                    SLC_DCOPY(&nr, &rscale[*ilo - 1], &int1, &dwork[kw7], &int1);
                    mxs = (ns > nh) ? ns : nh;
                    nss = ns;
                    nhs = nh;
                    ths = th;
                }
            }
            th = th * TEN;
        }
    }

    if (loop) {
        if (ith <= -10) {
            for (i = 0; i < nr; i++) {
                lscale[*ilo - 1 + i] = ONE;
                rscale[*ilo - 1 + i] = ONE;
            }
            *iwarn = 1;
            goto label510;
        }

        if ((mxnorm < mxs && mxnorm < mxs / MXGAIN && ith == -2) || ith == -4) {
            ir = SLC_IDAMAX(&nr, &dwork[kw6], &int1);
            jc = SLC_IDAMAX(&nr, &dwork[kw7], &int1);
            t = (dwork[kw6 + ir - 1] > dwork[kw7 + jc - 1]) ? dwork[kw6 + ir - 1] : dwork[kw7 + jc - 1];
            mn = t;
            for (i = kw6; i < kw6 + nr; i++) {
                if (dwork[i] < mn) mn = dwork[i];
            }
            for (i = kw7; i < kw7 + nr; i++) {
                if (dwork[i] < mn) mn = dwork[i];
            }
            t = mn / t;
            if (t < ONE / mxcond) {
                for (i = 0; i < nr; i++) {
                    lscale[*ilo - 1 + i] = ONE;
                    rscale[*ilo - 1 + i] = ONE;
                }
                *iwarn = 1;
                nss = ns0;
                nhs = nh0;
                ths = th0;
                goto label510;
            }
        }
        if (stormn) {
            SLC_DCOPY(&nr, &dwork[kw6], &int1, &lscale[*ilo - 1], &int1);
            SLC_DCOPY(&nr, &dwork[kw7], &int1, &rscale[*ilo - 1], &int1);
        } else {
            nss = ns;
            nhs = nh;
            ths = th;
        }
    }

label480:
    for (i = *ilo - 1; i < n; i++) {
        SLC_DSCAL(&nr, &lscale[i], &a[i + (*ilo - 1) * lda], &lda);
        SLC_DSCAL(&nr, &lscale[i], &c[i + (*ilo - 1) * ldc], &ldc);
        i32 len = i;
        SLC_DSCAL(&len, &lscale[i], &de[(i + 1) * ldde], &int1);
        if (n > i + 1) {
            len = n - i - 1;
            SLC_DSCAL(&len, &lscale[i], &de[i + (i + 2) * ldde], &ldde);
        }
        len = i + 1;
        SLC_DSCAL(&len, &lscale[i], &vw[(i + 1) * ldvw], &int1);
        len = n - i;
        SLC_DSCAL(&len, &lscale[i], &vw[i + (i + 1) * ldvw], &ldvw);
    }

    for (j = *ilo - 1; j < n; j++) {
        SLC_DSCAL(&n, &rscale[j], &a[j * lda], &int1);
        SLC_DSCAL(&n, &rscale[j], &c[j * ldc], &int1);
        i32 len = j;
        SLC_DSCAL(&len, &rscale[j], &de[j], &ldde);
        if (n > j + 1) {
            len = n - j - 1;
            SLC_DSCAL(&len, &rscale[j], &de[(j + 1) + j * ldde], &int1);
        }
        len = j + 1;
        SLC_DSCAL(&len, &rscale[j], &vw[j], &ldvw);
        len = n - j;
        SLC_DSCAL(&len, &rscale[j], &vw[j + j * ldvw], &int1);
    }

label510:
    if (evnorm) {
        nss = ma02id("skew-Hamiltonian", "1-norm", nr, &a[(*ilo - 1) + (*ilo - 1) * lda], lda,
                     &de[(*ilo - 1) + (*ilo - 1) * ldde], ldde, dwork);
        nhs = ma02id("Hamiltonian", "1-norm", nr, &c[(*ilo - 1) + (*ilo - 1) * ldc], ldc,
                     &vw[(*ilo - 1) + (*ilo - 1) * ldvw], ldvw, dwork);
    }

    dwork[0] = ns0;
    dwork[1] = nh0;
    dwork[2] = nss;
    dwork[3] = nhs;
    if (loop) {
        dwork[4] = ths / ((mxnorm > sfmin) ? mxnorm : sfmin);
    } else {
        dwork[4] = thresh;
    }

    return;
}
