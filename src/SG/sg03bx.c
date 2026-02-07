/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>
#include <complex.h>

void sg03bx(
    const char* dico, const char* trans,
    const f64* a, const i32 lda,
    const f64* e, const i32 lde,
    const f64* b, const i32 ldb,
    f64* u, const i32 ldu,
    f64* scale,
    f64* m1, const i32 ldm1,
    f64* m2, const i32 ldm2,
    i32* info
)
{
    const f64 one = 1.0, two = 2.0, zero = 0.0, safety = 1.0e2;

    bool istrns, iscont, estd = false;
    f64 estd_cl = 1.0, estd_sl = 0.0;
    f64 estd_bcg = 1.0, estd_bsg = 0.0;
    i32 ct, int1 = 1, int2 = 2;
    f64 a11, a12, a21, a22, ai11, ai12, ai21, ai22;
    f64 alpha, ar11, ar12, ar21, ar22, b11, b12i, b12r;
    f64 betai, betar, bi11, bi12, bi21, bi22, bignum;
    f64 br11, br12, br21, br22, c, cl, cq, cqb, cqbi;
    f64 cqu, cqui = 0.0, cz, e11, e12, e22, ei12, ei21, er11;
    f64 er12, er22, eps, lami, lamr, li, lr, m1i12;
    f64 m1r12, m2i12, m2r12, m2r22, m2s, mi, mr, mx, p;
    f64 s, scale1, scale2, si, siq, siqb, siqu, siz, sl;
    f64 smlnum, sqtwo, sr, srq, srqb, srqu, srz, t, tmp;
    f64 ui12, ui22, ur11, ur12, ur22, v, vi12, vr12;
    f64 vr22, w, xr, xi, yr, yi;

    f64 as[4], es[4], d[2], dwork[10], et[2], ev[2];
    i32 iwork[7];

    double complex x, zs, m3[1], m3c[2];

    istrns = (*trans == 'T' || *trans == 't');
    iscont = (*dico == 'C' || *dico == 'c');

    sqtwo = sqrt(two);
    eps = SLC_DLAMCH("P");
    smlnum = SLC_DLAMCH("S") / eps;
    bignum = one / smlnum;
    SLC_DLABAD(&smlnum, &bignum);

    iwork[1] = 1;
    iwork[2] = 0;
    iwork[3] = 2;
    iwork[4] = 0;
    ev[0] = one;
    ev[1] = zero;

    *info = 0;
    *scale = one;

    as[0] = a[0];
    as[1] = a[1];
    as[2] = a[0 + lda];
    as[3] = a[1 + lda];
    es[0] = e[0];
    es[1] = zero;
    es[2] = e[0 + lde];
    es[3] = e[1 + lde];
    br11 = b[0];
    br12 = b[0 + ldb];
    br22 = b[1 + ldb];

    if (istrns) {
        v = as[0];
        as[0] = as[3];
        as[3] = v;
        v = es[0];
        es[0] = es[3];
        es[3] = v;
        v = br11;
        br11 = br22;
        br22 = v;
    }

    ct = 0;
lab10:
    ct++;
    p = fmax(eps * fmax(fabs(es[0]), fmax(fabs(es[2]), fabs(es[3]))), smlnum);
    if (fmin(fabs(es[0]), fabs(es[3])) < p) {
        *info = 2;
        return;
    }
    f64 safmin_eps_safety = smlnum * eps * safety;
    SLC_DLAG2(as, &int2, es, &int2, &safmin_eps_safety, &scale1, &scale2, &lamr, &w, &lami);
    if (lami <= zero) {
        *info = 2;
        return;
    }

    if (es[2] != zero) {
        SLC_DLASV2(&es[0], &es[2], &es[3], &e22, &e11, &sr, &c, &sl, &cl);

        if (e11 < zero) {
            c = -c;
            sr = -sr;
            e11 = -e11;
            e22 = -e22;
        }

        s = cl * as[0] + sl * as[1];
        t = cl * as[2] + sl * as[3];
        v = cl * as[1] - sl * as[0];
        w = cl * as[3] - sl * as[2];

        as[0] = s * c + t * sr;
        as[1] = v * c + w * sr;
        as[2] = t * c - s * sr;
        as[3] = w * c - v * sr;

        // B' = B * Right^T, then QR to restore upper triangular
        // Right^T = [[c, -sr], [sr, c]]
        {
            f64 b11_new = br11 * c + br12 * sr;
            f64 b12_new = -br11 * sr + br12 * c;
            f64 b21_new = br22 * sr;
            f64 b22_new = br22 * c;
            f64 cg, sg, r_val;
            SLC_DLARTG(&b11_new, &b21_new, &cg, &sg, &r_val);
            br11 = r_val;
            br12 = cg * b12_new + sg * b22_new;
            br22 = -sg * b12_new + cg * b22_new;
            if (br22 < zero) {
                br22 = -br22;
                br12 = -br12;
                cg = -cg;
                sg = -sg;
            }
            estd_bcg = cg;
            estd_bsg = sg;
        }

        estd = true;
        estd_cl = cl;
        estd_sl = sl;

        es[0] = e11;
        es[1] = zero;
        es[2] = zero;

        if (e22 < zero) {
            es[3] = -e22;
            as[2] = -as[2];
            as[3] = -as[3];
            br12 = -br12;
        } else {
            es[3] = e22;
        }

        SLC_DLAG2(as, &int2, es, &int2, &safmin_eps_safety, &scale1, &scale2, &lamr, &w, &lami);

        if (lami == zero) {
            if (ct == 1) {
                goto lab10;
            } else {
                *info = 2;
                return;
            }
        }
    }

    a11 = as[0];
    a21 = as[1];
    a12 = as[2];
    a22 = as[3];
    e11 = es[0];
    e22 = es[3];
    sg03br(scale1 * a11 - e11 * lamr, -e11 * lami, scale1 * a21, zero,
           &cq, &srq, &siq, &lr, &li);

    ar11 = cq * a11 + srq * a21;
    ar21 = cq * a21 - srq * a11;
    ar12 = cq * a12 + srq * a22;
    ar22 = cq * a22 - srq * a12;
    ai11 = siq * a21;
    ai21 = siq * a11;
    ai12 = siq * a22;
    ai22 = siq * a12;

    ei21 = siq * e11;
    ei12 = siq * e22;
    tmp = srq * e11;
    e11 = cq * e11;
    e12 = srq * e22;

    sg03br(cq * e22, zero, tmp, -ei21, &cz, &srz, &siz, &lr, &li);

    er11 = e11 * cz + e12 * srz + ei12 * siz;
    er12 = e12 * cz - e11 * srz;
    ei12 = ei12 * cz - e11 * siz;
    er22 = lr;

    if (er11 < zero)
        er11 = -er11;

    a11 = ar11;
    a12 = ar12;
    tmp = ai11;
    ar11 = a12 * srz + a11 * cz + ai12 * siz;
    ai11 = ai12 * srz + tmp * cz - a12 * siz;
    ar12 = a12 * cz + tmp * siz - a11 * srz;
    ai12 = ai12 * cz - a11 * siz - tmp * srz;
    ar22 = ar22 * cz + ai21 * siz - ar21 * srz;
    ai22 = ai22 * cz - ar21 * siz - ai21 * srz;

    b11 = br11;
    bi11 = -br12 * siz;
    bi21 = -br22 * siz;
    bi12 = -b11 * siz;
    br11 = br12 * srz + b11 * cz;
    br21 = br22 * srz;
    br12 = br12 * cz - b11 * srz;
    br22 = br22 * cz;

    sg03br(br11, bi11, br21, bi21, &cqb, &srqb, &siqb, &lr, &li);
    v = br12;
    t = bi12;
    br12 = srqb * br22 + cqb * v;
    bi12 = siqb * br22 + cqb * t;
    br22 = cqb * br22 - srqb * v - siqb * t;
    bi22 = siqb * v - srqb * t;
    br11 = lr;
    bi11 = li;

    if (li != zero) {
        v = SLC_DLAPY2(&br11, &bi11);
        SLC_DLADIV(&v, &zero, &br11, &bi11, &xr, &xi);
        br11 = v;
        t = xr * br12 - xi * bi12;
        bi12 = xr * bi12 + xi * br12;
        br12 = t;

        cqbi = xi * cqb;
        cqb = xr * cqb;
        t = xr * srqb - xi * siqb;
        siqb = xr * siqb + xi * srqb;
        srqb = t;
    }

    if (bi22 != zero) {
        v = SLC_DLAPY2(&br22, &bi22);
        if (v >= fmax(eps * fmax(br11, SLC_DLAPY2(&br12, &bi12)), smlnum)) {
            SLC_DLADIV(&v, &zero, &br22, &bi22, &xr, &xi);
            br22 = v;
        } else {
            br22 = zero;
        }
    } else if (br22 < zero) {
        br22 = -br22;
    }

    if (iscont) {
        v = -ar11;
        if (v <= zero) {
            *info = 3;
            return;
        }
        v = sqrt(v) * sqrt(er11);
        t = (br11 * smlnum) / sqtwo;
        if (t > v) {
            scale1 = v / t;
            *scale = scale1 * (*scale);
            br11 = scale1 * br11;
            br12 = scale1 * br12;
            bi12 = scale1 * bi12;
            br22 = scale1 * br22;
        }
        v = v * sqtwo;
        ur11 = br11 / v;

        mx = fmax(fabs(ar11), fmax(fabs(ai11), v));
        if (er11 > mx * smlnum) {
            mr = ar11 / er11;
            mi = -ai11 / er11;
            m2s = v / er11;
            xr = m2s * br12;
            xi = m2s * bi12;
            if (ur11 != zero) {
                xr = xr + ur11 * (ar12 + mr * er12 - mi * ei12);
                xi = xi + ur11 * (ai12 + mr * ei12 + mi * er12);
            }
            yr = ar22 + mr * er22;
            yi = ai22 + mi * er22;
        } else {
            xr = br12 * v;
            xi = bi12 * v;
            if (ur11 != zero) {
                xr = xr + ur11 * (er11 * ar12 + ar11 * er12 + ai11 * ei12);
                xi = xi + ur11 * (er11 * ai12 + ar11 * ei12 - ai11 * er12);
            }
            yr = er11 * ar22 + ar11 * er22;
            yi = er11 * ai22 - ai11 * er22;
        }
        t = SLC_DLAPY2(&xr, &xi) * smlnum;
        w = SLC_DLAPY2(&yr, &yi);
        if (t > w) {
            scale1 = w / t;
            *scale = scale1 * (*scale);
            br11 = scale1 * br11;
            br12 = scale1 * br12;
            br22 = scale1 * br22;
            bi12 = scale1 * bi12;
            ur11 = scale1 * ur11;
            xr = scale1 * xr;
            xi = scale1 * xi;
        }
        f64 nyr = -yr, nyi = -yi;
        SLC_DLADIV(&xr, &xi, &nyr, &nyi, &ur12, &ui12);

        vr12 = ur11 * er12 + ur12 * er22;
        vi12 = ur11 * ei12 + ui12 * er22;
        if (er11 > mx * smlnum) {
            yr = br12 - m2s * vr12;
            yi = bi12 - m2s * vi12;
        } else {
            xr = vr12 * v;
            xi = -vi12 * v;
            t = SLC_DLAPY2(&xr, &xi) * smlnum;
            if (t > er11) {
                scale1 = er11 / t;
                *scale = scale1 * (*scale);
                br11 = scale1 * br11;
                br12 = scale1 * br12;
                bi12 = scale1 * bi12;
                br22 = scale1 * br22;
                ur11 = scale1 * ur11;
                ur12 = scale1 * ur12;
                ui12 = scale1 * ui12;
                xr = scale1 * xr;
                xi = scale1 * xi;
            }
            yr = br12 - xr / er11;
            yi = -bi12 - xi / er11;
        }
        sg03br(br22, zero, yr, yi, &c, &sr, &si, &lr, &li);
        v = -ar22;
        if (v <= zero) {
            *info = 3;
            return;
        }
        v = sqrt(v) * sqrt(er22);
        t = (lr * smlnum) / sqtwo;
        if (t > v) {
            scale1 = v / t;
            *scale = scale1 * (*scale);
            br11 = scale1 * br11;
            br12 = scale1 * br12;
            br22 = scale1 * br22;
            bi12 = scale1 * bi12;
            ur11 = scale1 * ur11;
            ur12 = scale1 * ur12;
            ui12 = scale1 * ui12;
            lr = scale1 * lr;
        }
        v = v * sqtwo;
        ur22 = lr / v;

        betar = ar11 / er11;
        betai = ai11 / er11;
        alpha = sqrt(-betar) * sqtwo;

        vr22 = ur22 * er22;
        if (vr22 != zero) {
            m2r22 = br22 / vr22;
            m2r12 = (br12 - alpha * vr12) / vr22;
            m2i12 = (bi12 - alpha * vi12) / vr22;
            m1r12 = -alpha * m2r12;
            m1i12 = -alpha * m2i12;
        } else {
            m1r12 = zero;
            m1i12 = zero;
            m2r12 = zero;
            m2r22 = alpha;
            m2i12 = zero;
        }

    } else {
        v = er11;
        t = SLC_DLAPY2(&ar11, &ai11);
        if (v <= t) {
            *info = 3;
            return;
        }
        t = t / v;
        v = sqrt(one - t) * sqrt(one + t) * v;
        t = br11 * smlnum;
        if (t > v) {
            scale1 = v / t;
            *scale = scale1 * (*scale);
            br11 = scale1 * br11;
            br12 = scale1 * br12;
            br22 = scale1 * br22;
            bi12 = scale1 * bi12;
        }
        ur11 = br11 / v;

        mx = fmax(fabs(ar11), fmax(fabs(ai11), v));
        if (er11 > mx * smlnum) {
            mr = ar11 / er11;
            mi = -ai11 / er11;
            m2s = v / er11;
            xr = m2s * br12;
            xi = m2s * bi12;
            if (ur11 != zero) {
                xr = xr + ur11 * (mr * ar12 - mi * ai12 - er12);
                xi = xi + ur11 * (mr * ai12 + mi * ar12 - ei12);
            }
            yr = mi * ai22 - mr * ar22 + er22;
            yi = -mr * ai22 - mi * ar22;
        } else {
            xr = -br12 * v;
            xi = -bi12 * v;
            if (ur11 != zero) {
                xr = xr + ur11 * (er11 * er12 - ar11 * ar12 - ai11 * ai12);
                xi = xi + ur11 * (er11 * ei12 - ar11 * ai12 + ai11 * ar12);
            }
            yr = ar11 * ar22 + ai11 * ai22 - er11 * er22;
            yi = ar11 * ai22 - ai11 * ar22;
        }
        t = SLC_DLAPY2(&xr, &xi) * smlnum;
        w = SLC_DLAPY2(&yr, &yi);
        if (t > w) {
            scale1 = w / t;
            *scale = scale1 * (*scale);
            br11 = scale1 * br11;
            br12 = scale1 * br12;
            br22 = scale1 * br22;
            bi12 = scale1 * bi12;
            ur11 = scale1 * ur11;
            xr = scale1 * xr;
            xi = scale1 * xi;
        }
        SLC_DLADIV(&xr, &xi, &yr, &yi, &ur12, &ui12);

        xr = ur11 * ar12 + ur12 * ar22 - ui12 * ai22;
        xi = ur11 * ai12 + ur12 * ai22 + ui12 * ar22;

        if (er11 > mx * smlnum) {
            x = -m2s * (mr + mi * I);
            m3[0] = x;
            SLC_ZLARFG(&int1, &x, m3, &int1, &zs);
            d[0] = mr * mr + mi * mi;
            d[1] = m2s * m2s;
            et[0] = creal(x);

            SLC_ZSTEIN(&int2, d, et, &int1, ev, &iwork[1], &iwork[3], m3c, &int2,
                       dwork, &iwork[5], iwork, info);
            if (*info != 0) {
                *info = 4;
                return;
            }

            v = creal(m3c[0]) * (one - creal(zs));
            w = -creal(m3c[0]) * cimag(zs);
            t = creal(m3c[1]);
            yr = v * br12 + w * bi12 + t * xr;
            yi = v * bi12 - w * br12 + t * xi;

            sg03br(br22, zero, yr, yi, &c, &sr, &si, &lr, &li);
        } else {
            t = SLC_DLAPY2(&ar22, &ai22);
            if (er22 <= t) {
                *info = 3;
                return;
            }
            yr = ur11 * er12 + ur12 * er22;
            yi = ur11 * ei12 + ui12 * er22;
            v = SLC_DLAPY2(&br12, &bi12);
            w = SLC_DLAPY2(&xr, &xi);
            t = SLC_DLAPY2(&yr, &yi);
            v = SLC_DLAPY3(&v, &br22, &w);
            if (v <= t) {
                *info = 3;
                return;
            }
            t = t / v;
            lr = sqrt(one - t) * sqrt(one + t) * v;
        }

        v = er22;
        t = SLC_DLAPY2(&ar22, &ai22);
        if (v <= t) {
            *info = 3;
            return;
        }
        t = t / v;
        v = sqrt(one - t) * sqrt(one + t) * v;
        t = lr * smlnum;
        if (v <= t) {
            scale1 = v / t;
            *scale = scale1 * (*scale);
            br11 = scale1 * br11;
            br12 = scale1 * br12;
            br22 = scale1 * br22;
            bi12 = scale1 * bi12;
            ur11 = scale1 * ur11;
            ur12 = scale1 * ur12;
            ui12 = scale1 * ui12;
            lr = scale1 * lr;
        }
        ur22 = lr / v;

        b11 = br11 / er11;
        t = er11 * er22;
        b12r = (er11 * br12 - br11 * er12) / t;
        b12i = (er11 * bi12 - br11 * ei12) / t;

        betar = ar11 / er11;
        betai = ai11 / er11;
        v = SLC_DLAPY2(&betar, &betai);
        alpha = sqrt(one - v) * sqrt(one + v);

        xr = (ai11 * ei12 - ar11 * er12) / t + ar12 / er22;
        xi = (ar11 * ei12 + ai11 * er12) / t - ai12 / er22;
        xr = -two * betai * b12i - b11 * xr;
        xi = -two * betai * b12r - b11 * xi;
        v = one + (betai - betar) * (betai + betar);
        w = -two * betai * betar;
        SLC_DLADIV(&xr, &xi, &v, &w, &yr, &yi);
        if (yr != zero || yi != zero) {
            m1r12 = -alpha * yr / ur22;
            m1i12 = alpha * yi / ur22;
            m2r12 = (yr * betar - yi * betai) / ur22;
            m2i12 = -(yi * betar + yr * betai) / ur22;
            m2r22 = br22 / er22 / ur22;
        } else {
            m1r12 = zero;
            m1i12 = zero;
            m2r12 = zero;
            m2i12 = zero;
            m2r22 = alpha;
        }
    }

    vr12 = ur12 * cq + ur11 * srq;
    vi12 = ui12 * cq + ur11 * siq;
    vr22 = ur22 * cq;

    sg03br(ur11 * cq - ur12 * srq - ui12 * siq, ur12 * siq - ui12 * srq,
           -ur22 * srq, ur22 * siq, &cqu, &srqu, &siqu, &lr, &li);
    u[0] = lr;
    u[1] = zero;
    u[0 + ldu] = cqu * vr12 + srqu * vr22;
    ui12 = cqu * vi12 + siqu * vr22;
    u[1 + ldu] = cqu * vr22 - srqu * vr12 - siqu * vi12;
    ui22 = siqu * vr12 - srqu * vi12;
    if (li != zero) {
        v = SLC_DLAPY2(&lr, &li);
        SLC_DLADIV(&v, &zero, &lr, &li, &xr, &xi);
        cqui = xi * cqu;
        cqu = xr * cqu;
        t = xr * srqu - xi * siqu;
        siqu = xr * siqu + xi * srqu;
        srqu = t;

        u[0 + ldu] = xr * u[0 + ldu] - xi * ui12;
        u[0] = v;
    }

    u[1 + ldu] = SLC_DLAPY2(&u[1 + ldu], &ui22);

    v = betar;
    t = (cqu * srqu + cqui * siqu) * m1r12 +
        (cqu * siqu - cqui * srqu) * m1i12;

    m1[0] = v + t;
    m1[1 + ldm1] = v - t;
    m1[0 + ldm1] = m1r12 * (cqu - cqui) * (cqu + cqui) +
                   two * (betai * (siqu * cqu + srqu * cqui) - m1i12 * cqui * cqu);
    m1[1] = siqu * (m1r12 * siqu - two * betai * cqu - m1i12 * srqu) -
            srqu * (m1r12 * srqu + two * betai * cqui + m1i12 * siqu);

    v = m2r12 * cqu - m2i12 * cqui - alpha * srqu;
    w = m2r12 * cqui + m2i12 * cqu - alpha * siqu;
    m2[0] = cqb * (alpha * cqu + m2r12 * srqu + m2i12 * siqu) +
            cqbi * (m2i12 * srqu - m2r12 * siqu - alpha * cqui) -
            m2r22 * (srqb * srqu + siqb * siqu);
    m2[1] = zero;
    m2[0 + ldm2] = cqb * v + cqbi * w - m2r22 * (srqb * cqu - siqb * cqui);
    m2[1 + ldm2] = srqb * v + siqb * w + m2r22 * (cqb * cqu - cqbi * cqui);

    // Back-transform through E standardization rotation.
    // U_orig = QR(U_std * Left), M1_orig = Qg*M1*Qg^T, M2_orig = Q_br^T*M2*Qg^T
    // where Qg is the Givens from QR(U_std * Left).
    if (estd) {
        f64 u11 = u[0], u12 = u[0 + ldu], u22 = u[1 + ldu];
        f64 t11 = u11 * estd_cl - u12 * estd_sl;
        f64 t12 = u11 * estd_sl + u12 * estd_cl;
        f64 t21 = -u22 * estd_sl;
        f64 t22 = u22 * estd_cl;
        f64 cg, sg, r_val;
        SLC_DLARTG(&t11, &t21, &cg, &sg, &r_val);
        u[0] = r_val;
        u[0 + ldu] = cg * t12 + sg * t22;
        u[1 + ldu] = -sg * t12 + cg * t22;
        if (u[1 + ldu] < zero) {
            u[1 + ldu] = -u[1 + ldu];
            u[0 + ldu] = -u[0 + ldu];
            cg = -cg;
            sg = -sg;
        }
        u[1] = zero;

        f64 g11 = m1[0], g12 = m1[0 + ldm1], g21 = m1[1], g22 = m1[1 + ldm1];
        f64 r11 = cg * g11 + sg * g21;
        f64 r12 = cg * g12 + sg * g22;
        f64 r21 = -sg * g11 + cg * g21;
        f64 r22 = -sg * g12 + cg * g22;
        m1[0]          = r11 * cg + r12 * sg;
        m1[0 + ldm1]   = -r11 * sg + r12 * cg;
        m1[1]          = r21 * cg + r22 * sg;
        m1[1 + ldm1]   = -r21 * sg + r22 * cg;

        g11 = m2[0]; g12 = m2[0 + ldm2]; g21 = m2[1]; g22 = m2[1 + ldm2];
        r11 = estd_bcg * g11 - estd_bsg * g21;
        r12 = estd_bcg * g12 - estd_bsg * g22;
        r21 = estd_bsg * g11 + estd_bcg * g21;
        r22 = estd_bsg * g12 + estd_bcg * g22;
        m2[0]          = r11 * cg + r12 * sg;
        m2[0 + ldm2]   = -r11 * sg + r12 * cg;
        m2[1]          = r21 * cg + r22 * sg;
        m2[1 + ldm2]   = -r21 * sg + r22 * cg;
    }

    if (istrns) {
        v = u[0];
        u[0] = u[1 + ldu];
        u[1 + ldu] = v;
        v = m1[0];
        m1[0] = m1[1 + ldm1];
        m1[1 + ldm1] = v;
        v = m2[0];
        m2[0] = m2[1 + ldm2];
        m2[1 + ldm2] = v;
    }
    u[1] = zero;
}
