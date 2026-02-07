// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"

#include <math.h>
#include <stdbool.h>
#include <float.h>

void mb03wd(const char* job, const char* compz, i32 n, i32 p, i32 ilo, i32 ihi,
            i32 iloz, i32 ihiz, f64* h, i32 ldh1, i32 ldh2, f64* z, i32 ldz1,
            i32 ldz2, f64* wr, f64* wi, f64* dwork, i32 ldwork, i32* info) {
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 half = 0.5;
    const f64 dat1 = 0.75;
    const f64 dat2 = -0.4375;

    bool wantt = (*job == 'S' || *job == 's');
    bool initz = (*compz == 'I' || *compz == 'i');
    bool wantz = initz || (*compz == 'V' || *compz == 'v');

    *info = 0;
    i32 max_1_n = (1 > n) ? 1 : n;
    i32 min_ilo_n = (ilo < n) ? ilo : n;

    if (!wantt && !(*job == 'E' || *job == 'e')) {
        *info = -1;
    } else if (!wantz && !(*compz == 'N' || *compz == 'n')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (p < 1) {
        *info = -4;
    } else if (ilo < 1 || ilo > max_1_n) {
        *info = -5;
    } else if (ihi < min_ilo_n || ihi > n) {
        *info = -6;
    } else if (iloz < 1 || iloz > ilo) {
        *info = -7;
    } else if (ihiz < ihi || ihiz > n) {
        *info = -8;
    } else if (ldh1 < max_1_n) {
        *info = -10;
    } else if (ldh2 < max_1_n) {
        *info = -11;
    } else if (ldz1 < 1 || (wantz && ldz1 < n)) {
        *info = -13;
    } else if (ldz2 < 1 || (wantz && ldz2 < n)) {
        *info = -14;
    } else if (ldwork < ihi - ilo + p - 1) {
        *info = -18;
    }

    if (*info == 0) {
        if (ilo > 1) {
            i32 idx = (ilo - 1) + (ilo - 2) * ldh1;
            if (h[idx] != zero) {
                *info = -5;
            }
        } else if (ihi < n) {
            i32 idx = ihi + (ihi - 1) * ldh1;
            if (h[idx] != zero) {
                *info = -6;
            }
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        return;
    }

    i32 ldh12 = ldh1 * ldh2;
    i32 ldz12 = ldz1 * ldz2;

    if (initz) {
        for (i32 j = 0; j < p; j++) {
            f64* zj = z + j * ldz12;
            for (i32 jj = 0; jj < n; jj++) {
                for (i32 ii = 0; ii < n; ii++) {
                    zj[ii + jj * ldz1] = (ii == jj) ? one : zero;
                }
            }
        }
    }

    i32 nh = ihi - ilo + 1;

    if (nh == 1) {
        f64 hp00 = one;
        for (i32 j = 0; j < p; j++) {
            f64* hj = h + j * ldh12;
            i32 idx = (ilo - 1) + (ilo - 1) * ldh1;
            hp00 *= hj[idx];
        }
        wr[ilo - 1] = hp00;
        wi[ilo - 1] = zero;
        return;
    }

    f64 unfl = DBL_MIN;
    f64 ovfl = one / unfl;
    SLC_DLABAD(&unfl, &ovfl);
    f64 ulp = DBL_EPSILON;
    f64 smlnum = unfl * ((f64)nh / ulp);

    if (nh > 2) {
        i32 nh2 = nh - 2;
        for (i32 jj = 0; jj < nh2; jj++) {
            for (i32 ii = jj + 2; ii < nh; ii++) {
                i32 idx = (ilo - 1 + ii) + (ilo - 1 + jj) * ldh1;
                h[idx] = zero;
            }
        }
    }

    i32 dw_idx = nh - 1;
    f64 s = ulp * (f64)n;

    for (i32 j = 1; j < p; j++) {
        f64* hj = h + j * ldh12;
        for (i32 jj = 0; jj < nh - 1; jj++) {
            for (i32 ii = jj + 1; ii < nh; ii++) {
                i32 idx = (ilo - 1 + ii) + (ilo - 1 + jj) * ldh1;
                hj[idx] = zero;
            }
        }

        i32 nhh = nh;
        f64 norm = SLC_DLANTR("1", "U", "N", &nhh, &nhh,
                              &hj[(ilo - 1) + (ilo - 1) * ldh1], &ldh1, dwork);
        dwork[dw_idx] = s * norm;
        dw_idx++;
    }

    i32 i1 = 0, i2 = 0;
    if (wantt) {
        i1 = 0;
        i2 = n - 1;
    }

    i32 nz = 0;
    if (wantz) {
        nz = ihiz - iloz + 1;
    }

    i32 itn = 30 * nh;
    i32 i = ihi - 1;

    f64 hp22, hp12, hp11, hp00, hp01, hp02;
    f64 hh21, hh22, hh11, hh12, hh10;
    i32 its, k;

    while (i >= ilo - 1) {
        i32 l = ilo - 1;

        for (its = 0; its <= itn; its++) {
            hp22 = one;
            hp12 = zero;
            hp11 = one;
            hh21 = zero;
            hh22 = zero;

            if (i > l) {
                hp12 = zero;
                hp11 = one;

                for (i32 j = 1; j < p; j++) {
                    f64* hj = h + j * ldh12;
                    f64 h_ii = hj[i + i * ldh1];
                    f64 h_im1_i = hj[(i - 1) + i * ldh1];
                    f64 h_im1_im1 = hj[(i - 1) + (i - 1) * ldh1];

                    hp22 = hp22 * h_ii;
                    hp12 = hp11 * h_im1_i + hp12 * h_ii;
                    hp11 = hp11 * h_im1_im1;
                }

                f64 h1_i_im1 = h[i + (i - 1) * ldh1];
                f64 h1_i_i = h[i + i * ldh1];

                hh21 = h1_i_im1 * hp11;
                hh22 = h1_i_im1 * hp12 + h1_i_i * hp22;

                wr[i] = hh22;
                wi[i] = hh21;
            } else {
                for (i32 j = 0; j < p; j++) {
                    f64* hj = h + j * ldh12;
                    hp22 = hp22 * hj[i + i * ldh1];
                }
                wr[i] = hp22;
            }

            for (k = i; k >= l + 1; k--) {
                hp00 = one;
                hp01 = zero;

                if (k > l + 1) {
                    hp02 = zero;

                    for (i32 j = 1; j < p; j++) {
                        f64* hj = h + j * ldh12;
                        f64 h_km2_k = hj[(k - 2) + k * ldh1];
                        f64 h_km1_k = hj[(k - 1) + k * ldh1];
                        f64 h_k_k = hj[k + k * ldh1];
                        f64 h_km2_km1 = hj[(k - 2) + (k - 1) * ldh1];
                        f64 h_km1_km1 = hj[(k - 1) + (k - 1) * ldh1];
                        f64 h_km2_km2 = hj[(k - 2) + (k - 2) * ldh1];

                        hp02 = hp00 * h_km2_k + hp01 * h_km1_k + hp02 * h_k_k;
                        hp01 = hp00 * h_km2_km1 + hp01 * h_km1_km1;
                        hp00 = hp00 * h_km2_km2;
                    }

                    f64 h1_km1_km2 = h[(k - 1) + (k - 2) * ldh1];
                    f64 h1_km1_km1 = h[(k - 1) + (k - 1) * ldh1];
                    f64 h1_km1_k = h[(k - 1) + k * ldh1];

                    hh10 = h1_km1_km2 * hp00;
                    hh11 = h1_km1_km2 * hp01 + h1_km1_km1 * hp11;
                    hh12 = h1_km1_km2 * hp02 + h1_km1_km1 * hp12 + h1_km1_k * hp22;
                    wi[k - 1] = hh10;
                } else {
                    hh10 = zero;
                    f64 h1_km1_km1 = h[(k - 1) + (k - 1) * ldh1];
                    f64 h1_km1_k = h[(k - 1) + k * ldh1];
                    hh11 = h1_km1_km1 * hp11;
                    hh12 = h1_km1_km1 * hp12 + h1_km1_k * hp22;
                }

                wr[k - 1] = hh11;
                dwork[nh - 1 - i + k - 1] = hh12;

                f64 tst1 = fabs(hh11) + fabs(hh22);
                if (tst1 == zero) {
                    i32 len = i - l + 1;
                    tst1 = SLC_DLANHS("1", &len, &h[l + l * ldh1], &ldh1, dwork);
                }

                f64 threshold = ulp * tst1;
                if (threshold < smlnum) threshold = smlnum;
                if (fabs(hh21) <= threshold) {
                    break;
                }

                hp22 = hp11;
                hp11 = hp00;
                hp12 = hp01;
                hh22 = hh11;
                hh21 = hh10;
            }

            l = k;

            if (l > ilo - 1) {
                if (wantt) {
                    f64 h1_l_lm1 = h[l + (l - 1) * ldh1];
                    f64 h1_lm1_lm1 = h[(l - 1) + (l - 1) * ldh1];
                    f64 h1_l_l = h[l + l * ldh1];

                    f64 tst1 = fabs(h1_lm1_lm1) + fabs(h1_l_l);
                    if (tst1 == zero) {
                        i32 len = i - l + 1;
                        tst1 = SLC_DLANHS("1", &len, &h[l + l * ldh1], &ldh1, dwork);
                    }

                    f64 threshold = ulp * tst1;
                    if (threshold < smlnum) threshold = smlnum;

                    if (fabs(h1_l_lm1) > threshold) {
                        for (i32 kk = i; kk >= l; kk--) {
                            for (i32 j = 0; j < p - 1; j++) {
                                f64* hj = h + j * ldh12;
                                f64* hjp1 = h + (j + 1) * ldh12;

                                f64 v[3];
                                v[0] = hj[kk + (kk - 1) * ldh1];
                                i32 two = 2;
                                i32 int1 = 1;
                                f64 tau;
                                SLC_DLARFG(&two, &hj[kk + kk * ldh1], v, &int1, &tau);
                                hj[kk + (kk - 1) * ldh1] = zero;
                                v[1] = one;

                                i32 nrows_r = kk - i1;
                                SLC_DLARFX("R", &nrows_r, &two, v, &tau,
                                           &hj[i1 + (kk - 1) * ldh1], &ldh1, dwork);

                                i32 ncols_l = i2 - kk + 2;
                                SLC_DLARFX("L", &two, &ncols_l, v, &tau,
                                           &hjp1[(kk - 1) + (kk - 1) * ldh1], &ldh1, dwork);

                                if (wantz) {
                                    f64* zjp1 = z + (j + 1) * ldz12;
                                    SLC_DLARFX("R", &nz, &two, v, &tau,
                                               &zjp1[(iloz - 1) + (kk - 1) * ldz1], &ldz1, dwork);
                                }
                            }

                            if (kk < i) {
                                f64* hp = h + (p - 1) * ldh12;
                                f64 v[3];
                                v[0] = hp[(kk + 1) + kk * ldh1];
                                i32 two = 2;
                                i32 int1 = 1;
                                f64 tau;
                                SLC_DLARFG(&two, &hp[(kk + 1) + (kk + 1) * ldh1], v, &int1, &tau);
                                hp[(kk + 1) + kk * ldh1] = zero;
                                v[1] = one;

                                i32 nrows_r = kk - i1 + 1;
                                SLC_DLARFX("R", &nrows_r, &two, v, &tau,
                                           &hp[i1 + kk * ldh1], &ldh1, dwork);

                                i32 ncols_l = i2 - kk + 1;
                                SLC_DLARFX("L", &two, &ncols_l, v, &tau,
                                           &h[kk + kk * ldh1], &ldh1, dwork);

                                if (wantz) {
                                    f64* z1 = z;
                                    SLC_DLARFX("R", &nz, &two, v, &tau,
                                               &z1[(iloz - 1) + kk * ldz1], &ldz1, dwork);
                                }
                            }
                        }

                        f64* hp = h + (p - 1) * ldh12;
                        hp[l + (l - 1) * ldh1] = zero;
                    }
                    h[l + (l - 1) * ldh1] = zero;
                }
            }

            if (l >= i - 1) {
                goto converged;
            }

            if (!wantt) {
                i1 = l;
                i2 = i;
            }

            f64 h44, h33, h43h34;

            if (its == 10 || its == 20) {
                s = fabs(wi[i]) + fabs(wi[i - 1]);
                h44 = dat1 * s + wr[i];
                h33 = h44;
                h43h34 = dat2 * s * s;
            } else {
                h44 = wr[i];
                h33 = wr[i - 1];
                h43h34 = wi[i] * dwork[nh - 2];
                f64 disc = (h33 - h44) * half;
                disc = disc * disc + h43h34;
                if (disc > zero) {
                    disc = sqrt(disc);
                    f64 ave = half * (h33 + h44);
                    if (fabs(h33) - fabs(h44) > zero) {
                        h33 = h33 * h44 - h43h34;
                        h44 = h33 / (copysign(disc, ave) + ave);
                    } else {
                        h44 = copysign(disc, ave) + ave;
                    }
                    h33 = h44;
                    h43h34 = zero;
                }
            }

            i32 m;
            f64 v[3];
            for (m = i - 2; m >= l; m--) {
                f64 h11 = wr[m];
                f64 h12 = dwork[nh - 1 - i + m];
                f64 h21 = wi[m + 1];
                f64 h22 = wr[m + 1];
                f64 h44s = h44 - h11;
                f64 h33s = h33 - h11;
                f64 v1 = (fabs(h21) < DBL_EPSILON) ? h12 : (h33s * h44s - h43h34) / h21 + h12;
                f64 v2 = h22 - h11 - h33s - h44s;
                f64 v3 = wi[m + 2];
                s = fabs(v1) + fabs(v2) + fabs(v3);
                v1 = v1 / s;
                v2 = v2 / s;
                v3 = v3 / s;
                v[0] = v1;
                v[1] = v2;
                v[2] = v3;
                if (m == l) {
                    break;
                }
                f64 tst1 = fabs(v1) * (fabs(wr[m - 1]) + fabs(h11) + fabs(h22));
                if (fabs(wi[m]) * (fabs(v2) + fabs(v3)) <= ulp * tst1) {
                    break;
                }
            }

            for (k = m; k <= i - 1; k++) {
                i32 nr = (3 < i - k + 1) ? 3 : (i - k + 1);
                i32 nrow = ((k + nr < i) ? (k + nr) : i) - i1 + 1;

                if (k > m) {
                    i32 int1 = 1;
                    SLC_DCOPY(&nr, &h[k + (k - 1) * ldh1], &int1, v, &int1);
                }

                f64 tau;
                i32 int1 = 1;
                SLC_DLARFG(&nr, v, &v[1], &int1, &tau);

                if (k > m) {
                    h[k + (k - 1) * ldh1] = v[0];
                    h[k + 1 + (k - 1) * ldh1] = zero;
                    if (k < i - 1) {
                        h[k + 2 + (k - 1) * ldh1] = zero;
                    }
                } else if (m > l) {
                    h[k + (k - 1) * ldh1] = -h[k + (k - 1) * ldh1];
                }

                i32 ncols = i2 - k + 1;
                SLC_MB04PY('L', nr, ncols, &v[1], tau, &h[k + k * ldh1], ldh1, dwork);

                f64* hp = h + (p - 1) * ldh12;
                SLC_MB04PY('R', nrow, nr, &v[1], tau, &hp[i1 + k * ldh1], ldh1, dwork);

                if (wantz) {
                    f64* z1 = z;
                    SLC_MB04PY('R', nz, nr, &v[1], tau, &z1[(iloz - 1) + k * ldz1], ldz1, dwork);
                }

                for (i32 j = p - 1; j >= 1; j--) {
                    f64* hj = h + j * ldh12;
                    f64* hjm1 = h + (j - 1) * ldh12;

                    i32 nr1 = nr - 1;
                    SLC_DCOPY(&nr1, &hj[k + 1 + k * ldh1], &int1, v, &int1);
                    SLC_DLARFG(&nr, &hj[k + k * ldh1], v, &int1, &tau);
                    hj[k + 1 + k * ldh1] = zero;
                    if (nr == 3) {
                        hj[k + 2 + k * ldh1] = zero;
                    }

                    i32 ncols_l = i2 - k;
                    SLC_MB04PY('L', nr, ncols_l, v, tau, &hj[k + (k + 1) * ldh1], ldh1, dwork);

                    SLC_MB04PY('R', nrow, nr, v, tau, &hjm1[i1 + k * ldh1], ldh1, dwork);

                    if (wantz) {
                        f64* zj = z + j * ldz12;
                        SLC_MB04PY('R', nz, nr, v, tau, &zj[(iloz - 1) + k * ldz1], ldz1, dwork);
                    }

                    if (nr == 3) {
                        v[0] = hj[k + 2 + (k + 1) * ldh1];
                        i32 two = 2;
                        SLC_DLARFG(&two, &hj[k + 1 + (k + 1) * ldh1], v, &int1, &tau);
                        hj[k + 2 + (k + 1) * ldh1] = zero;

                        i32 ncols_l2 = i2 - k - 1;
                        SLC_MB04PY('L', two, ncols_l2, v, tau, &hj[k + 1 + (k + 2) * ldh1], ldh1, dwork);

                        SLC_MB04PY('R', nrow, two, v, tau, &hjm1[i1 + (k + 1) * ldh1], ldh1, dwork);

                        if (wantz) {
                            f64* zj = z + j * ldz12;
                            SLC_MB04PY('R', nz, two, v, tau, &zj[(iloz - 1) + (k + 1) * ldz1], ldz1, dwork);
                        }
                    }
                }
            }
        }

        *info = i + 1;
        return;

    converged:
        if (l == i) {
            wi[i] = zero;
        } else if (l == i - 1) {
            hp22 = one;
            hp12 = zero;
            hp11 = one;

            if (wantt) {
                for (i32 j = 1; j < p; j++) {
                    f64* hj = h + j * ldh12;
                    f64 h_i_i = hj[i + i * ldh1];
                    f64 h_im1_i = hj[(i - 1) + i * ldh1];
                    f64 h_im1_im1 = hj[(i - 1) + (i - 1) * ldh1];

                    hp22 = hp22 * h_i_i;
                    hp12 = hp11 * h_im1_i + hp12 * h_i_i;
                    hp11 = hp11 * h_im1_im1;
                }

                hh21 = h[i + (i - 1) * ldh1] * hp11;
                hh22 = h[i + (i - 1) * ldh1] * hp12 + h[i + i * ldh1] * hp22;
                hh11 = h[(i - 1) + (i - 1) * ldh1] * hp11;
                hh12 = h[(i - 1) + (i - 1) * ldh1] * hp12 + h[(i - 1) + i * ldh1] * hp22;
            } else {
                hh11 = wr[i - 1];
                hh12 = dwork[nh - 2];
                hh21 = wi[i];
                hh22 = wr[i];
            }

            f64 cs, sn;
            SLC_DLANV2(&hh11, &hh12, &hh21, &hh22, &wr[i - 1], &wi[i - 1],
                       &wr[i], &wi[i], &cs, &sn);

            if (wantt) {
                i32 jmin = 0;
                i32 jmax = 0;

                for (i32 j = 1; j < p; j++) {
                    f64* hj = h + j * ldh12;
                    f64 h_im1_im1 = hj[(i - 1) + (i - 1) * ldh1];
                    f64 h_i_i = hj[i + i * ldh1];

                    if (jmin == 0) {
                        if (fabs(h_im1_im1) <= dwork[nh - 1 + j - 1]) {
                            jmin = j + 1;
                        }
                    }
                    if (fabs(h_i_i) <= dwork[nh - 1 + j - 1]) {
                        jmax = j + 1;
                    }
                }

                if (jmin != 0 && jmax != 0) {
                    if (jmin - 1 <= p - jmax + 1) {
                        jmax = 0;
                    } else {
                        jmin = 0;
                    }
                }

                if (jmin != 0) {
                    for (i32 j = 0; j < jmin - 1; j++) {
                        f64* hj = h + j * ldh12;
                        f64* hjp1 = h + (j + 1) * ldh12;

                        f64 v[3];
                        v[0] = hj[i + (i - 1) * ldh1];
                        i32 two = 2;
                        i32 int1 = 1;
                        f64 tau;
                        SLC_DLARFG(&two, &hj[i + i * ldh1], v, &int1, &tau);
                        hj[i + (i - 1) * ldh1] = zero;
                        v[1] = one;

                        i32 nrows_r = i - i1;
                        SLC_DLARFX("R", &nrows_r, &two, v, &tau,
                                   &hj[i1 + (i - 1) * ldh1], &ldh1, dwork);

                        i32 ncols_l = i2 - i + 2;
                        SLC_DLARFX("L", &two, &ncols_l, v, &tau,
                                   &hjp1[(i - 1) + (i - 1) * ldh1], &ldh1, dwork);

                        if (wantz) {
                            f64* zjp1 = z + (j + 1) * ldz12;
                            SLC_DLARFX("R", &nz, &two, v, &tau,
                                       &zjp1[(iloz - 1) + (i - 1) * ldz1], &ldz1, dwork);
                        }
                    }

                    f64* hjmin = h + (jmin - 1) * ldh12;
                    hjmin[i + (i - 1) * ldh1] = zero;
                } else {
                    if (jmax > 0 && wi[i - 1] == zero) {
                        f64 tau;
                        SLC_DLARTG(&h[(i - 1) + (i - 1) * ldh1], &h[i + (i - 1) * ldh1],
                                   &cs, &sn, &tau);
                    }

                    i32 ncols_rot = i2 - i + 2;
                    i32 int1 = 1;
                    SLC_DROT(&ncols_rot, &h[(i - 1) + (i - 1) * ldh1], &ldh1,
                             &h[i + (i - 1) * ldh1], &ldh1, &cs, &sn);

                    i32 nrows_rot = i - i1 + 1;
                    f64* hp = h + (p - 1) * ldh12;
                    SLC_DROT(&nrows_rot, &hp[i1 + (i - 1) * ldh1], &int1,
                             &hp[i1 + i * ldh1], &int1, &cs, &sn);

                    if (wantz) {
                        f64* z1 = z;
                        SLC_DROT(&nz, &z1[(iloz - 1) + (i - 1) * ldz1], &int1,
                                 &z1[(iloz - 1) + i * ldz1], &int1, &cs, &sn);
                    }

                    i32 jmax_start = (2 > jmax + 1) ? 2 : (jmax + 1);
                    for (i32 j = p - 1; j >= jmax_start - 1; j--) {
                        f64* hj = h + j * ldh12;
                        f64* hjm1 = h + (j - 1) * ldh12;

                        f64 v[3];
                        v[0] = hj[i + (i - 1) * ldh1];
                        i32 two = 2;
                        f64 tau;
                        SLC_DLARFG(&two, &hj[(i - 1) + (i - 1) * ldh1], v, &int1, &tau);
                        hj[i + (i - 1) * ldh1] = zero;

                        i32 ncols_l = i2 - i + 1;
                        SLC_MB04PY('L', two, ncols_l, v, tau, &hj[(i - 1) + i * ldh1], ldh1, dwork);

                        i32 nrows_r = i - i1 + 1;
                        SLC_MB04PY('R', nrows_r, two, v, tau, &hjm1[i1 + (i - 1) * ldh1], ldh1, dwork);

                        if (wantz) {
                            f64* zj = z + j * ldz12;
                            SLC_MB04PY('R', nz, two, v, tau, &zj[(iloz - 1) + (i - 1) * ldz1], ldz1, dwork);
                        }
                    }

                    if (jmax > 0) {
                        h[i + (i - 1) * ldh1] = zero;
                        f64* hjmax = h + (jmax - 1) * ldh12;
                        hjmax[i + (i - 1) * ldh1] = zero;
                    } else {
                        if (hh21 == zero) {
                            h[i + (i - 1) * ldh1] = zero;
                        }
                    }
                }
            }
        }

        itn = itn - its;
        i = l - 2;
    }
}
