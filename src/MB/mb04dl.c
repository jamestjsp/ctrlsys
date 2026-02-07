/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 *
 * MB04DL - Balance a pair of N-by-N real matrices (A, B)
 *
 * Balances a real matrix pencil (A, B) by permuting to isolate eigenvalues
 * and applying diagonal equivalence transformations to equalize row/column
 * 1-norms. Optionally improves conditioning compared to LAPACK DGGBAL.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <string.h>
#include <stdbool.h>

void mb04dl(const char *job, const i32 n, const f64 thresh, f64 *a, const i32 lda,
            f64 *b, const i32 ldb, i32 *ilo, i32 *ihi, f64 *lscale, f64 *rscale,
            f64 *dwork, i32 *iwarn, i32 *info)
{
    const f64 HALF = 0.5;
    const f64 ONE = 1.0;
    const f64 TEN = 10.0;
    const f64 THREE = 3.0;
    const f64 ZERO = 0.0;
    const f64 MXGAIN = 100.0;
    const f64 SCLFAC = 10.0;

    bool evnorm, loop, lperm, lscal, stormn;
    i32 i, icab, iflow, ip1, ir, irab, it, iter, ith;
    i32 j, jc, jp1, k, kount, ks, kw1, kw2, kw3, kw4;
    i32 kw5, kw6, kw7, l, lm1, lrab, lsfmax, lsfmin, m;
    i32 nr, nrp2;
    f64 ab, alpha, basl, beta, cab, cmax, coef, coef2;
    f64 coef5, cor, denom, eps, ew, ewc, gamma, gap;
    f64 minpro, minrat, mn, mx, mxcond, mxnorm, mxs;
    f64 na, na0, nas, nb, nb0, nbs, pgamma, prod, rab;
    f64 ratio, sfmax, sfmin, sum, t, ta, tb, tc, th;
    f64 th0, ths;
    f64 dum;

    i32 int0 = 0, int1 = 1;

    *info = 0;
    *iwarn = 0;
    lperm = (job[0] == 'P' || job[0] == 'p' || job[0] == 'B' || job[0] == 'b');
    lscal = (job[0] == 'S' || job[0] == 's' || job[0] == 'B' || job[0] == 'b');

    if (!lperm && !lscal && job[0] != 'N' && job[0] != 'n') {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -5;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -7;
    }
    if (*info != 0) {
        return;
    }

    *ilo = 1;
    *ihi = n;

    if (n == 0) {
        return;
    }

    if ((!lperm && !lscal) || n == 1) {
        dum = ONE;
        SLC_DCOPY(&n, &dum, &int0, lscale, &int1);
        SLC_DCOPY(&n, &dum, &int0, rscale, &int1);
        if (n == 1 && lscal) {
            na0 = fabs(a[0]);
            nb0 = fabs(b[0]);
            dwork[0] = na0;
            dwork[1] = nb0;
            dwork[2] = na0;
            dwork[3] = nb0;
            dwork[4] = thresh;
        }
        return;
    }

    k = 1;
    l = n;

    if (lperm) {
        /* Permute to isolate eigenvalues */

        /* Find row with one nonzero in columns 1 through L */
lbl_10:
        lm1 = l - 1;
        for (i = l; i >= 1; i--) {
            for (j = 1; j <= lm1; j++) {
                jp1 = j + 1;
                if (a[(i - 1) + (j - 1) * lda] != ZERO ||
                    b[(i - 1) + (j - 1) * ldb] != ZERO) {
                    goto lbl_30;
                }
            }
            j = l;
            goto lbl_50;

lbl_30:
            for (j = jp1; j <= l; j++) {
                if (a[(i - 1) + (j - 1) * lda] != ZERO ||
                    b[(i - 1) + (j - 1) * ldb] != ZERO) {
                    goto lbl_60_end;
                }
            }
            j = jp1 - 1;

lbl_50:
            m = l;
            iflow = 1;
            goto lbl_130;
lbl_60_end:
            ;
        }

        /* Find column with one nonzero in rows K through N */
lbl_70:
        for (j = k; j <= l; j++) {
            for (i = k; i <= lm1; i++) {
                ip1 = i + 1;
                if (a[(i - 1) + (j - 1) * lda] != ZERO ||
                    b[(i - 1) + (j - 1) * ldb] != ZERO) {
                    goto lbl_90;
                }
            }
            i = l;
            goto lbl_110;

lbl_90:
            for (i = ip1; i <= l; i++) {
                if (a[(i - 1) + (j - 1) * lda] != ZERO ||
                    b[(i - 1) + (j - 1) * ldb] != ZERO) {
                    goto lbl_120_end;
                }
            }
            i = ip1 - 1;

lbl_110:
            m = k;
            iflow = 2;
            goto lbl_130;
lbl_120_end:
            ;
        }
        goto lbl_140;

        /* Permute rows M and I */
lbl_130:
        lscale[m - 1] = (f64)i;
        if (i != m) {
            i32 len = n - k + 1;
            SLC_DSWAP(&len, &a[(i - 1) + (k - 1) * lda], &lda,
                      &a[(m - 1) + (k - 1) * lda], &lda);
            SLC_DSWAP(&len, &b[(i - 1) + (k - 1) * ldb], &ldb,
                      &b[(m - 1) + (k - 1) * ldb], &ldb);
        }

        /* Permute columns M and J */
        rscale[m - 1] = (f64)j;
        if (j != m) {
            SLC_DSWAP(&l, &a[(j - 1) * lda], &int1, &a[(m - 1) * lda], &int1);
            SLC_DSWAP(&l, &b[(j - 1) * ldb], &int1, &b[(m - 1) * ldb], &int1);
        }

        if (iflow == 1) {
            l = lm1;
            if (l != 1) {
                goto lbl_10;
            }
            rscale[0] = ONE;
            lscale[0] = ONE;
        } else {
            k = k + 1;
            goto lbl_70;
        }
    }

lbl_140:
    *ilo = k;
    *ihi = l;

    if (!lscal) {
        for (i = *ilo; i <= *ihi; i++) {
            lscale[i - 1] = ONE;
            rscale[i - 1] = ONE;
        }
        return;
    }

    nr = *ihi - *ilo + 1;

    /* Compute initial 1-norms */
    na0 = SLC_DLANGE("1", &nr, &nr, &a[(*ilo - 1) + (*ilo - 1) * lda], &lda, dwork);
    nb0 = SLC_DLANGE("1", &nr, &nr, &b[(*ilo - 1) + (*ilo - 1) * ldb], &ldb, dwork);

    if (*ilo == *ihi) {
        dwork[0] = na0;
        dwork[1] = nb0;
        dwork[2] = na0;
        dwork[3] = nb0;
        dwork[4] = thresh;
        lscale[*ilo - 1] = ONE;
        rscale[*ilo - 1] = ONE;
        return;
    }

    /* Initialize balancing and allocate work storage */
    kw1 = n;
    kw2 = kw1 + n;
    kw3 = kw2 + n;
    kw4 = kw3 + n;
    kw5 = kw4 + n;
    dum = ZERO;

    /* Prepare for scaling */
    sfmin = SLC_DLAMCH("S");
    sfmax = ONE / sfmin;
    basl = log10(SCLFAC);
    lsfmin = (i32)(log10(sfmin) / basl + ONE);
    lsfmax = (i32)(log10(sfmax) / basl);
    mxnorm = (na0 > nb0) ? na0 : nb0;
    loop = thresh < ZERO;

    if (loop) {
        /* Compute relative threshold */
        na = na0;
        nas = na0;
        nb = nb0;
        nbs = nb0;

        ith = (i32)thresh;
        mxs = mxnorm;
        mx = ZERO;
        mn = sfmax;
        if (ith >= -2) {
            if (na < nb) {
                ratio = (nb / na < sfmax) ? nb / na : sfmax;
            } else {
                ratio = (na / nb < sfmax) ? na / nb : sfmax;
            }
            minrat = ratio;
        } else if (ith <= -10) {
            mxcond = -thresh;
        } else {
            denom = (ONE > mxnorm) ? ONE : mxnorm;
            prod = (na / denom) * (nb / denom);
            minpro = prod;
        }
        stormn = false;
        evnorm = false;

        /* Find max order of magnitude of differences */
        for (j = *ilo; j <= *ihi; j++) {
            for (i = *ilo; i <= *ihi; i++) {
                if (i != j) {
                    ab = fabs(a[(i - 1) + (j - 1) * lda]);
                    if (ab != ZERO) {
                        mn = (mn < ab) ? mn : ab;
                    }
                    mx = (mx > ab) ? mx : ab;
                }
            }
        }

        for (j = *ilo; j <= *ihi; j++) {
            for (i = *ilo; i <= *ihi; i++) {
                if (i != j) {
                    ab = fabs(b[(i - 1) + (j - 1) * ldb]);
                    if (ab != ZERO) {
                        mn = (mn < ab) ? mn : ab;
                    }
                    mx = (mx > ab) ? mx : ab;
                }
            }
        }

        if (mx * sfmin <= mn) {
            gap = mx / mn;
        } else {
            gap = sfmax;
        }
        eps = SLC_DLAMCH("P");
        iter = (i32)log10(gap);
        i32 tmp = -(i32)log10(eps);
        iter = (iter < tmp) ? iter : tmp;
        iter = iter + 1;
        th = ((mn > mx * eps) ? mn : mx * eps) / ((mxnorm > sfmin) ? mxnorm : sfmin);
        ths = th;
        kw6 = kw5 + n + *ilo;
        kw7 = kw6 + n;
        SLC_DCOPY(&nr, &lscale[*ilo - 1], &int1, &dwork[kw6 - 1], &int1);
        SLC_DCOPY(&nr, &rscale[*ilo - 1], &int1, &dwork[kw7 - 1], &int1);

        /* Set max condition number of transformations */
        if (ith > -10) {
            mxcond = ONE / sqrt(eps);
        }
    } else {
        th = mxnorm * thresh;
        iter = 1;
        evnorm = true;
    }
    th0 = th;

    coef = ONE / (f64)(2 * nr);
    coef2 = coef * coef;
    coef5 = HALF * coef2;
    nrp2 = nr + 2;
    beta = ZERO;

    /* Main iteration loop */
    for (k = 1; k <= iter; k++) {
        /* Compute right side vector */
        i32 len6n = 6 * n;
        SLC_DCOPY(&len6n, &dum, &int0, dwork, &int1);
        SLC_DCOPY(&nr, &dum, &int0, &lscale[*ilo - 1], &int1);
        SLC_DCOPY(&nr, &dum, &int0, &rscale[*ilo - 1], &int1);

        for (i = *ilo; i <= *ihi; i++) {
            for (j = *ilo; j <= *ihi; j++) {
                ta = fabs(a[(i - 1) + (j - 1) * lda]);
                tb = fabs(b[(i - 1) + (j - 1) * ldb]);
                if (ta > th) {
                    ta = log10(ta) / basl;
                } else {
                    ta = ZERO;
                }
                if (tb > th) {
                    tb = log10(tb) / basl;
                } else {
                    tb = ZERO;
                }
                dwork[(i - 1) + kw4] = dwork[(i - 1) + kw4] - ta - tb;
                dwork[(j - 1) + kw5] = dwork[(j - 1) + kw5] - ta - tb;
            }
        }

        it = 1;

        /* Generalized conjugate gradient iteration */
lbl_220:
        gamma = SLC_DDOT(&nr, &dwork[*ilo - 1 + kw4], &int1, &dwork[*ilo - 1 + kw4], &int1) +
                SLC_DDOT(&nr, &dwork[*ilo - 1 + kw5], &int1, &dwork[*ilo - 1 + kw5], &int1);

        ew = ZERO;
        ewc = ZERO;
        for (i = *ilo; i <= *ihi; i++) {
            ew = ew + dwork[(i - 1) + kw4];
            ewc = ewc + dwork[(i - 1) + kw5];
        }

        gamma = coef * gamma - coef2 * (ew * ew + ewc * ewc) -
                coef5 * (ew - ewc) * (ew - ewc);
        if (gamma == ZERO) {
            goto lbl_300;
        }
        if (it != 1) {
            beta = gamma / pgamma;
        }
        t = coef5 * (ewc - THREE * ew);
        tc = coef5 * (ew - THREE * ewc);

        SLC_DSCAL(&nr, &beta, &dwork[*ilo - 1], &int1);
        SLC_DSCAL(&nr, &beta, &dwork[*ilo - 1 + kw1], &int1);

        SLC_DAXPY(&nr, &coef, &dwork[*ilo - 1 + kw4], &int1, &dwork[*ilo - 1 + kw1], &int1);
        SLC_DAXPY(&nr, &coef, &dwork[*ilo - 1 + kw5], &int1, &dwork[*ilo - 1], &int1);

        for (j = *ilo; j <= *ihi; j++) {
            dwork[(j - 1)] = dwork[(j - 1)] + tc;
            dwork[(j - 1) + kw1] = dwork[(j - 1) + kw1] + t;
        }

        /* Apply matrix to vector */
        for (i = *ilo; i <= *ihi; i++) {
            kount = 0;
            sum = ZERO;
            for (j = *ilo; j <= *ihi; j++) {
                ks = kount;
                if (a[(i - 1) + (j - 1) * lda] != ZERO) {
                    kount = kount + 1;
                }
                if (b[(i - 1) + (j - 1) * ldb] != ZERO) {
                    kount = kount + 1;
                }
                sum = sum + (f64)(kount - ks) * dwork[(j - 1)];
            }
            dwork[(i - 1) + kw2] = (f64)kount * dwork[(i - 1) + kw1] + sum;
        }

        for (j = *ilo; j <= *ihi; j++) {
            kount = 0;
            sum = ZERO;
            for (i = *ilo; i <= *ihi; i++) {
                ks = kount;
                if (a[(i - 1) + (j - 1) * lda] != ZERO) {
                    kount = kount + 1;
                }
                if (b[(i - 1) + (j - 1) * ldb] != ZERO) {
                    kount = kount + 1;
                }
                sum = sum + (f64)(kount - ks) * dwork[(i - 1) + kw1];
            }
            dwork[(j - 1) + kw3] = (f64)kount * dwork[(j - 1)] + sum;
        }

        sum = SLC_DDOT(&nr, &dwork[*ilo - 1 + kw1], &int1, &dwork[*ilo - 1 + kw2], &int1) +
              SLC_DDOT(&nr, &dwork[*ilo - 1], &int1, &dwork[*ilo - 1 + kw3], &int1);
        alpha = gamma / sum;

        /* Determine correction to current iteration */
        cmax = ZERO;
        for (i = *ilo; i <= *ihi; i++) {
            cor = alpha * dwork[(i - 1) + kw1];
            if (fabs(cor) > cmax) {
                cmax = fabs(cor);
            }
            lscale[i - 1] = lscale[i - 1] + cor;
            cor = alpha * dwork[(i - 1)];
            if (fabs(cor) > cmax) {
                cmax = fabs(cor);
            }
            rscale[i - 1] = rscale[i - 1] + cor;
        }

        if (cmax >= HALF) {
            f64 nalpha = -alpha;
            SLC_DAXPY(&nr, &nalpha, &dwork[*ilo - 1 + kw2], &int1, &dwork[*ilo - 1 + kw4], &int1);
            SLC_DAXPY(&nr, &nalpha, &dwork[*ilo - 1 + kw3], &int1, &dwork[*ilo - 1 + kw5], &int1);

            pgamma = gamma;
            it = it + 1;
            if (it <= nrp2) {
                goto lbl_220;
            }
        }

        /* End conjugate gradient iteration */
lbl_300:
        /* Compute diagonal scaling matrices */
        for (i = *ilo; i <= *ihi; i++) {
            i32 len1 = n - *ilo + 1;
            irab = SLC_IDAMAX(&len1, &a[(i - 1) + (*ilo - 1) * lda], &lda);
            rab = fabs(a[(i - 1) + (*ilo + irab - 2) * lda]);
            irab = SLC_IDAMAX(&len1, &b[(i - 1) + (*ilo - 1) * ldb], &ldb);
            f64 rabB = fabs(b[(i - 1) + (*ilo + irab - 2) * ldb]);
            rab = (rab > rabB) ? rab : rabB;
            lrab = (i32)(log10(rab + sfmin) / basl + ONE);
            ir = (i32)(lscale[i - 1] + ((lscale[i - 1] >= 0) ? HALF : -HALF));
            ir = (ir > lsfmin) ? ir : lsfmin;
            ir = (ir < lsfmax) ? ir : lsfmax;
            ir = (ir < lsfmax - lrab) ? ir : lsfmax - lrab;
            lscale[i - 1] = pow(SCLFAC, (f64)ir);

            i32 ihi_val = *ihi;
            icab = SLC_IDAMAX(&ihi_val, &a[(i - 1) * lda], &int1);
            cab = fabs(a[(icab - 1) + (i - 1) * lda]);
            icab = SLC_IDAMAX(&ihi_val, &b[(i - 1) * ldb], &int1);
            f64 cabB = fabs(b[(icab - 1) + (i - 1) * ldb]);
            cab = (cab > cabB) ? cab : cabB;
            lrab = (i32)(log10(cab + sfmin) / basl + ONE);
            jc = (i32)(rscale[i - 1] + ((rscale[i - 1] >= 0) ? HALF : -HALF));
            jc = (jc > lsfmin) ? jc : lsfmin;
            jc = (jc < lsfmax) ? jc : lsfmax;
            jc = (jc < lsfmax - lrab) ? jc : lsfmax - lrab;
            rscale[i - 1] = pow(SCLFAC, (f64)jc);
        }

        /* Check if all scaling factors are 1 */
        for (i = *ilo; i <= *ihi; i++) {
            if (lscale[i - 1] != ONE || rscale[i - 1] != ONE) {
                goto lbl_330;
            }
        }

        /* All factors are 1 */
        nas = na0;
        nbs = nb0;
        ths = th0;
        goto lbl_460;

lbl_330:
        if (loop) {
            if (ith <= -10) {
                /* Compute reciprocal condition of transformations */
                ir = SLC_IDAMAX(&nr, &lscale[*ilo - 1], &int1);
                jc = SLC_IDAMAX(&nr, &rscale[*ilo - 1], &int1);
                t = lscale[*ilo + ir - 2];
                mn = t;
                for (i = *ilo; i <= *ihi; i++) {
                    if (lscale[i - 1] < mn) {
                        mn = lscale[i - 1];
                    }
                }
                t = mn / t;
                ta = rscale[*ilo + jc - 2];
                mn = ta;
                for (i = *ilo; i <= *ihi; i++) {
                    if (rscale[i - 1] < mn) {
                        mn = rscale[i - 1];
                    }
                }
                t = (t < mn / ta) ? t : mn / ta;
                if (t < ONE / mxcond) {
                    th = th * TEN;
                    goto lbl_400_end;
                } else {
                    ths = th;
                    evnorm = true;
                    goto lbl_430;
                }
            }

            /* Compute 1-norms of scaled submatrices without actually scaling */
            na = ZERO;
            for (j = *ilo; j <= *ihi; j++) {
                t = ZERO;
                for (i = *ilo; i <= *ihi; i++) {
                    t = t + fabs(a[(i - 1) + (j - 1) * lda]) * lscale[i - 1] * rscale[j - 1];
                }
                if (t > na) {
                    na = t;
                }
            }

            nb = ZERO;
            for (j = *ilo; j <= *ihi; j++) {
                t = ZERO;
                for (i = *ilo; i <= *ihi; i++) {
                    t = t + fabs(b[(i - 1) + (j - 1) * ldb]) * lscale[i - 1] * rscale[j - 1];
                }
                if (t > nb) {
                    nb = t;
                }
            }

            if (ith >= -4 && ith < -2) {
                prod = (na / denom) * (nb / denom);
                if (minpro > prod) {
                    minpro = prod;
                    stormn = true;
                    SLC_DCOPY(&nr, &lscale[*ilo - 1], &int1, &dwork[kw6 - 1], &int1);
                    SLC_DCOPY(&nr, &rscale[*ilo - 1], &int1, &dwork[kw7 - 1], &int1);
                    nas = na;
                    nbs = nb;
                    ths = th;
                }
            } else if (ith >= -2) {
                if (na < nb) {
                    ratio = (nb / na < sfmax) ? nb / na : sfmax;
                } else {
                    ratio = (na / nb < sfmax) ? na / nb : sfmax;
                }
                if (minrat > ratio) {
                    minrat = ratio;
                    stormn = true;
                    SLC_DCOPY(&nr, &lscale[*ilo - 1], &int1, &dwork[kw6 - 1], &int1);
                    SLC_DCOPY(&nr, &rscale[*ilo - 1], &int1, &dwork[kw7 - 1], &int1);
                    mxs = (na > nb) ? na : nb;
                    nas = na;
                    nbs = nb;
                    ths = th;
                }
            }
            th = th * TEN;
        }
lbl_400_end:
        ;
    }

    /* Prepare for scaling */
    if (loop) {
        if (ith <= -10) {
            /* Could not find well conditioned transformations */
            dum = ONE;
            SLC_DCOPY(&nr, &dum, &int0, &lscale[*ilo - 1], &int1);
            SLC_DCOPY(&nr, &dum, &int0, &rscale[*ilo - 1], &int1);
            *iwarn = 1;
            goto lbl_460;
        }

        /* Check if scaling might reduce accuracy */
        if ((mxnorm < mxs && mxnorm < mxs / MXGAIN && ith == -2) || ith == -4) {
            ir = SLC_IDAMAX(&nr, &dwork[kw6 - 1], &int1);
            jc = SLC_IDAMAX(&nr, &dwork[kw7 - 1], &int1);
            t = dwork[kw6 + ir - 2];
            mn = t;
            for (i = kw6; i <= kw6 + nr - 1; i++) {
                if (dwork[i - 1] < mn) {
                    mn = dwork[i - 1];
                }
            }
            t = mn / t;
            ta = dwork[kw7 + jc - 2];
            mn = ta;
            for (i = kw7; i <= kw7 + nr - 1; i++) {
                if (dwork[i - 1] < mn) {
                    mn = dwork[i - 1];
                }
            }
            t = (t < mn / ta) ? t : mn / ta;
            if (t < ONE / mxcond) {
                dum = ONE;
                SLC_DCOPY(&nr, &dum, &int0, &lscale[*ilo - 1], &int1);
                SLC_DCOPY(&nr, &dum, &int0, &rscale[*ilo - 1], &int1);
                *iwarn = 1;
                nas = na0;
                nbs = nb0;
                ths = th0;
                goto lbl_460;
            }
        }
        if (stormn) {
            SLC_DCOPY(&nr, &dwork[kw6 - 1], &int1, &lscale[*ilo - 1], &int1);
            SLC_DCOPY(&nr, &dwork[kw7 - 1], &int1, &rscale[*ilo - 1], &int1);
        } else {
            nas = na;
            nbs = nb;
            ths = th;
        }
    }

lbl_430:
    /* Row scaling */
    for (i = *ilo; i <= *ihi; i++) {
        i32 len = n - *ilo + 1;
        SLC_DSCAL(&len, &lscale[i - 1], &a[(i - 1) + (*ilo - 1) * lda], &lda);
        SLC_DSCAL(&len, &lscale[i - 1], &b[(i - 1) + (*ilo - 1) * ldb], &ldb);
    }

    /* Column scaling */
    for (j = *ilo; j <= *ihi; j++) {
        i32 ihi_val = *ihi;
        SLC_DSCAL(&ihi_val, &rscale[j - 1], &a[(j - 1) * lda], &int1);
        SLC_DSCAL(&ihi_val, &rscale[j - 1], &b[(j - 1) * ldb], &int1);
    }

    /* Set DWORK(1:5) */
lbl_460:
    if (evnorm) {
        nas = SLC_DLANGE("1", &nr, &nr, &a[(*ilo - 1) + (*ilo - 1) * lda], &lda, dwork);
        nbs = SLC_DLANGE("1", &nr, &nr, &b[(*ilo - 1) + (*ilo - 1) * ldb], &ldb, dwork);
    }

    dwork[0] = na0;
    dwork[1] = nb0;
    dwork[2] = nas;
    dwork[3] = nbs;
    if (loop) {
        dwork[4] = ths / ((mxnorm > sfmin) ? mxnorm : sfmin);
    } else {
        dwork[4] = thresh;
    }
}
