/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>

void tg01ad(
    const char* job, const i32 l, const i32 n, const i32 m, const i32 p,
    const f64 thresh,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* lscale, f64* rscale,
    f64* dwork,
    i32* info
)
{
    const f64 half = 0.5, one = 1.0, zero = 0.0;
    const f64 sclfac = 10.0, three = 3.0;

    bool withb, withc;
    i32 i, ir, it, j, jc, kount, kw1, kw2, kw3, kw4, kw5;
    i32 lcab, lrab, lsfmax, lsfmin, nrp2;
    f64 alpha, basl, beta, cab, cmax, coef, coef2, coef5, cor, eps;
    f64 ew, ewc, gamma, pgamma, rab, sfmax, sfmin, sum, t, ta, tb, tc, te;
    f64 dum;
    i32 int1 = 1;

    *info = 0;
    withb = (job[0] == 'A' || job[0] == 'a' || job[0] == 'B' || job[0] == 'b');
    withc = (job[0] == 'A' || job[0] == 'a' || job[0] == 'C' || job[0] == 'c');

    if (!withb && !withc && job[0] != 'N' && job[0] != 'n') {
        *info = -1;
    } else if (l < 0) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (p < 0) {
        *info = -5;
    } else if (thresh < zero) {
        *info = -6;
    } else if (lda < ((1 > l) ? 1 : l)) {
        *info = -8;
    } else if (lde < ((1 > l) ? 1 : l)) {
        *info = -10;
    } else if (ldb < 1 || (m > 0 && ldb < l)) {
        *info = -12;
    } else if (ldc < ((1 > p) ? 1 : p)) {
        *info = -14;
    }

    if (*info != 0) {
        return;
    }

    if (l == 0 || n == 0) {
        dum = one;
        if (l > 0) {
            for (i = 0; i < l; i++) {
                lscale[i] = dum;
            }
        } else if (n > 0) {
            for (j = 0; j < n; j++) {
                rscale[j] = dum;
            }
        }
        return;
    }

    kw1 = n;
    kw2 = kw1 + l;
    kw3 = kw2 + l;
    kw4 = kw3 + n;
    kw5 = kw4 + l;

    dum = zero;
    for (i = 0; i < l; i++) {
        lscale[i] = zero;
    }
    for (j = 0; j < n; j++) {
        rscale[j] = zero;
    }
    for (i = 0; i < 3 * (l + n); i++) {
        dwork[i] = zero;
    }

    basl = log10(sclfac);
    for (i = 0; i < l; i++) {
        for (j = 0; j < n; j++) {
            te = fabs(e[i + j * lde]);
            ta = fabs(a[i + j * lda]);
            if (ta > thresh) {
                ta = log10(ta) / basl;
            } else {
                ta = zero;
            }
            if (te > thresh) {
                te = log10(te) / basl;
            } else {
                te = zero;
            }
            dwork[i + kw4] = dwork[i + kw4] - ta - te;
            dwork[j + kw5] = dwork[j + kw5] - ta - te;
        }
    }

    if (m == 0) {
        withb = false;
        tb = zero;
    }
    if (p == 0) {
        withc = false;
        tc = zero;
    }

    if (withb) {
        for (i = 0; i < l; i++) {
            i32 idx = 0;
            f64 maxval = fabs(b[i]);
            for (j = 1; j < m; j++) {
                if (fabs(b[i + j * ldb]) > maxval) {
                    maxval = fabs(b[i + j * ldb]);
                    idx = j;
                }
            }
            tb = fabs(b[i + idx * ldb]);
            if (tb > thresh) {
                tb = log10(tb) / basl;
                dwork[i + kw4] = dwork[i + kw4] - tb;
            }
        }
    }

    if (withc) {
        for (j = 0; j < n; j++) {
            i32 idx = 0;
            f64 maxval = fabs(c[j * ldc]);
            for (i = 1; i < p; i++) {
                if (fabs(c[i + j * ldc]) > maxval) {
                    maxval = fabs(c[i + j * ldc]);
                    idx = i;
                }
            }
            tc = fabs(c[idx + j * ldc]);
            if (tc > thresh) {
                tc = log10(tc) / basl;
                dwork[j + kw5] = dwork[j + kw5] - tc;
            }
        }
    }

    coef = one / (f64)(l + n);
    coef2 = coef * coef;
    coef5 = half * coef2;
    nrp2 = (l > n) ? l : n;
    nrp2 = nrp2 + 2;
    beta = zero;
    eps = SLC_DLAMCH("Precision");
    it = 1;

    while (1) {
        gamma = zero;
        for (i = 0; i < l; i++) {
            gamma += dwork[i + kw4] * dwork[i + kw4];
        }
        for (j = 0; j < n; j++) {
            gamma += dwork[j + kw5] * dwork[j + kw5];
        }

        ew = zero;
        for (i = 0; i < l; i++) {
            ew += dwork[i + kw4];
        }

        ewc = zero;
        for (j = 0; j < n; j++) {
            ewc += dwork[j + kw5];
        }

        gamma = coef * gamma - coef2 * (ew * ew + ewc * ewc) -
                coef5 * (ew - ewc) * (ew - ewc);
        if (fabs(gamma) <= eps) {
            break;
        }
        if (it != 1) {
            beta = gamma / pgamma;
        }
        t  = coef5 * (ewc - three * ew);
        tc = coef5 * (ew - three * ewc);

        for (i = 0; i < n + l; i++) {
            dwork[i] = beta * dwork[i];
        }

        for (i = 0; i < l; i++) {
            dwork[i + kw1] += coef * dwork[i + kw4];
        }
        for (j = 0; j < n; j++) {
            dwork[j] += coef * dwork[j + kw5];
        }

        for (j = 0; j < n; j++) {
            dwork[j] = dwork[j] + tc;
        }

        for (i = 0; i < l; i++) {
            dwork[i + kw1] = dwork[i + kw1] + t;
        }

        for (i = 0; i < l; i++) {
            kount = 0;
            sum = zero;
            for (j = 0; j < n; j++) {
                if (fabs(a[i + j * lda]) > thresh) {
                    kount++;
                    sum += dwork[j];
                }
                if (fabs(e[i + j * lde]) > thresh) {
                    kount++;
                    sum += dwork[j];
                }
            }
            if (withb) {
                i32 idx = 0;
                f64 maxval = fabs(b[i]);
                for (j = 1; j < m; j++) {
                    if (fabs(b[i + j * ldb]) > maxval) {
                        maxval = fabs(b[i + j * ldb]);
                        idx = j;
                    }
                }
                if (fabs(b[i + idx * ldb]) > thresh) {
                    kount++;
                }
            }
            dwork[i + kw2] = (f64)kount * dwork[i + kw1] + sum;
        }

        for (j = 0; j < n; j++) {
            kount = 0;
            sum = zero;
            for (i = 0; i < l; i++) {
                if (fabs(a[i + j * lda]) > thresh) {
                    kount++;
                    sum += dwork[i + kw1];
                }
                if (fabs(e[i + j * lde]) > thresh) {
                    kount++;
                    sum += dwork[i + kw1];
                }
            }
            if (withc) {
                i32 idx = 0;
                f64 maxval = fabs(c[j * ldc]);
                for (i = 1; i < p; i++) {
                    if (fabs(c[i + j * ldc]) > maxval) {
                        maxval = fabs(c[i + j * ldc]);
                        idx = i;
                    }
                }
                if (fabs(c[idx + j * ldc]) > thresh) {
                    kount++;
                }
            }
            dwork[j + kw3] = (f64)kount * dwork[j] + sum;
        }

        sum = zero;
        for (i = 0; i < l; i++) {
            sum += dwork[i + kw1] * dwork[i + kw2];
        }
        for (j = 0; j < n; j++) {
            sum += dwork[j] * dwork[j + kw3];
        }
        alpha = gamma / sum;

        cmax = zero;
        for (i = 0; i < l; i++) {
            cor = alpha * dwork[i + kw1];
            if (fabs(cor) > cmax) {
                cmax = fabs(cor);
            }
            lscale[i] = lscale[i] + cor;
        }

        for (j = 0; j < n; j++) {
            cor = alpha * dwork[j];
            if (fabs(cor) > cmax) {
                cmax = fabs(cor);
            }
            rscale[j] = rscale[j] + cor;
        }
        if (cmax < half) {
            break;
        }

        for (i = 0; i < l; i++) {
            dwork[i + kw4] -= alpha * dwork[i + kw2];
        }
        for (j = 0; j < n; j++) {
            dwork[j + kw5] -= alpha * dwork[j + kw3];
        }

        pgamma = gamma;
        it++;
        if (it > nrp2) {
            break;
        }
    }

    sfmin = SLC_DLAMCH("Safe minimum");
    sfmax = one / sfmin;
    lsfmin = (i32)(log10(sfmin) / basl + one);
    lsfmax = (i32)(log10(sfmax) / basl);

    for (i = 0; i < l; i++) {
        i32 idx = 0;
        f64 maxval = fabs(a[i]);
        for (j = 1; j < n; j++) {
            if (fabs(a[i + j * lda]) > maxval) {
                maxval = fabs(a[i + j * lda]);
                idx = j;
            }
        }
        rab = fabs(a[i + idx * lda]);

        idx = 0;
        maxval = fabs(e[i]);
        for (j = 1; j < n; j++) {
            if (fabs(e[i + j * lde]) > maxval) {
                maxval = fabs(e[i + j * lde]);
                idx = j;
            }
        }
        if (fabs(e[i + idx * lde]) > rab) {
            rab = fabs(e[i + idx * lde]);
        }

        if (withb) {
            idx = 0;
            maxval = fabs(b[i]);
            for (j = 1; j < m; j++) {
                if (fabs(b[i + j * ldb]) > maxval) {
                    maxval = fabs(b[i + j * ldb]);
                    idx = j;
                }
            }
            if (fabs(b[i + idx * ldb]) > rab) {
                rab = fabs(b[i + idx * ldb]);
            }
        }

        lrab = (i32)(log10(rab + sfmin) / basl + one);
        ir = (i32)(lscale[i] + ((lscale[i] >= zero) ? half : -half));
        ir = (ir > lsfmin) ? ir : lsfmin;
        ir = (ir < lsfmax) ? ir : lsfmax;
        ir = (ir < lsfmax - lrab) ? ir : (lsfmax - lrab);
        lscale[i] = pow(sclfac, (f64)ir);
    }

    for (j = 0; j < n; j++) {
        i32 idx = 0;
        f64 maxval = fabs(a[j * lda]);
        for (i = 1; i < l; i++) {
            if (fabs(a[i + j * lda]) > maxval) {
                maxval = fabs(a[i + j * lda]);
                idx = i;
            }
        }
        cab = fabs(a[idx + j * lda]);

        idx = 0;
        maxval = fabs(e[j * lde]);
        for (i = 1; i < l; i++) {
            if (fabs(e[i + j * lde]) > maxval) {
                maxval = fabs(e[i + j * lde]);
                idx = i;
            }
        }
        if (fabs(e[idx + j * lde]) > cab) {
            cab = fabs(e[idx + j * lde]);
        }

        if (withc) {
            idx = 0;
            maxval = fabs(c[j * ldc]);
            for (i = 1; i < p; i++) {
                if (fabs(c[i + j * ldc]) > maxval) {
                    maxval = fabs(c[i + j * ldc]);
                    idx = i;
                }
            }
            if (fabs(c[idx + j * ldc]) > cab) {
                cab = fabs(c[idx + j * ldc]);
            }
        }

        lcab = (i32)(log10(cab + sfmin) / basl + one);
        jc = (i32)(rscale[j] + ((rscale[j] >= zero) ? half : -half));
        jc = (jc > lsfmin) ? jc : lsfmin;
        jc = (jc < lsfmax) ? jc : lsfmax;
        jc = (jc < lsfmax - lcab) ? jc : (lsfmax - lcab);
        rscale[j] = pow(sclfac, (f64)jc);
    }

    for (i = 0; i < l; i++) {
        SLC_DSCAL(&n, &lscale[i], &a[i], &lda);
        SLC_DSCAL(&n, &lscale[i], &e[i], &lde);
        if (withb) {
            SLC_DSCAL(&m, &lscale[i], &b[i], &ldb);
        }
    }

    for (j = 0; j < n; j++) {
        SLC_DSCAL(&l, &rscale[j], &a[j * lda], &int1);
        SLC_DSCAL(&l, &rscale[j], &e[j * lde], &int1);
        if (withc) {
            SLC_DSCAL(&p, &rscale[j], &c[j * ldc], &int1);
        }
    }
}
