/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * MB03KB - Swap pairs of adjacent diagonal blocks in generalized periodic Schur form
 *
 * Reorders the diagonal blocks of a formal matrix product
 * T22_K^S(K) * T22_K-1^S(K-1) * ... * T22_1^S(1) of length K
 * in generalized periodic Schur form such that pairs of adjacent
 * diagonal blocks of sizes 1 and/or 2 are swapped.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdlib.h>

void mb03kb(const char* compq, const i32* whichq, const bool ws,
            const i32 k, const i32 nc, const i32 kschur,
            const i32 j1, const i32 n1, const i32 n2,
            const i32* n, const i32* ni, const i32* s,
            f64* t, const i32* ldt, const i32* ixt,
            f64* q, const i32* ldq, const i32* ixq,
            const f64* tol, i32* iwork, f64* dwork,
            const i32 ldwork, i32* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    bool fill21, fill43, fillin, specq, wantq, wantql;
    i32 i, i11, i12, i21, i22, ia, ib, ic, ii, indf1, indf2, indtau, indtt;
    i32 indv1, indv2, indvf, indvp1, indxc, indxv, ip1, ipp, iq, is, it, it2;
    i32 itau1, itau2, itauf, itauf1, itauf2, itaup1, iv1p1, iv2p1;
    i32 j2, j3, j4, l, ldwke, ltau, ltau1, ltau2, ltt, minwrk, mn, nd, nd2;
    i32 tau, tau1, tau1p1, tau2, tau2p1, tt, v, v1, v2, vloc, vloc1, vloc2, w, we;
    i32 a_idx, b_idx, c_idx;
    f64 dnrm, dtau1, dtau2, eps = 0.0, scaloc, smlnum = 0.0, strong, tauloc, thresh;
    f64 tmp, tmp1, tmp2, v_1, v_2, v_3, w_2, w_3, x_11, x_12, x_21, x_22;
    f64 taus[2], temp[16], tempm1[16];

    i32 int1 = 1, int2 = 2, int3 = 3, int4 = 4;
    f64 neg_one = -1.0;

    *info = 0;

    if (ldwork == -1) {
        mn = 0;
        for (i = 0; i < k; i++) {
            if (n[i] > mn) mn = n[i];
        }
        if (mn <= 10) mn = 0;

        if (n1 == 1 && n2 == 1) {
            we = k * 3;
            mn = k * 10 + mn;
        } else if (n1 == 1 && n2 == 2) {
            we = k * 7;
            mn = k * 25 + mn;
        } else if (n1 == 2 && n2 == 1) {
            we = k * 7;
            mn = k * 23 + mn;
        } else if (n1 == 2 && n2 == 2) {
            we = k * 12;
            mn = k * 42 + mn;
        } else {
            we = 0;
        }

        i32 ldw_query = -1;
        i32 info_ke = 0;
        mb03ke(false, false, -1, k, n1, n2, eps, smlnum, s,
               dwork, dwork, dwork, &scaloc, dwork, ldw_query, &info_ke);

        i32 opt1 = (i32)dwork[0] + we;
        minwrk = (opt1 > mn) ? opt1 : mn;
        dwork[0] = (f64)minwrk;
        return;
    }

    eps = tol[1];
    smlnum = tol[2];

    j2 = j1 + n1;
    i11 = 0;
    i21 = i11 + k;
    i12 = i21 + k;
    i22 = i12 + k;
    mn = 0;

    for (i = 0; i < k; i++) {
        if (n[i] > mn) mn = n[i];
        ip1 = (i + 1) % k;
        if (s[i] == 1) {
            ii = ixt[i] - 1 + ni[i] * ldt[i] + ni[ip1];
        } else {
            ii = ixt[i] - 1 + ni[ip1] * ldt[i] + ni[i];
        }
        iwork[i11 + i] = ii + (j1 - 1) * ldt[i] + j1 - 1;
        iwork[i21 + i] = iwork[i11 + i] + n1;
        iwork[i12 + i] = iwork[i11 + i] + n1 * ldt[i];
        iwork[i22 + i] = iwork[i12 + i] + n1;
    }

    if (mn <= 10) mn = 0;

    a_idx = 0;
    if (n1 == 1 && n2 == 1) {
        b_idx = a_idx + k;
        c_idx = b_idx + k;
        tau = c_idx + k;
        v = tau + k;
        tt = v + k * 2;
        w = tt + k * 4;
        we = tau;
        mn = mn + k * 10;
        ldwke = k * 5 - 4;
    } else if (n1 == 1 && n2 == 2) {
        b_idx = a_idx + k;
        c_idx = b_idx + k * 4;
        tau1 = c_idx + k * 2;
        v1 = tau1 + k;
        tau2 = v1 + k * 2;
        v2 = tau2 + k;
        tt = v2 + k * 2;
        ltau = tt + k * 9;
        vloc = ltau + k;
        w = vloc + k * 2;
        we = tau1;
        mn = mn + k * 25;
        ldwke = k * 18 - 13;
    } else if (n1 == 2 && n2 == 1) {
        b_idx = a_idx + k * 4;
        c_idx = b_idx + k;
        tau = c_idx + k * 2;
        v = tau + k;
        tt = v + k * 3;
        ltau = tt + k * 9;
        vloc = ltau + k;
        w = vloc + k * 2;
        we = tau;
        mn = mn + k * 23;
        ldwke = k * 18 - 13;
    } else if (n1 == 2 && n2 == 2) {
        b_idx = a_idx + k * 4;
        c_idx = b_idx + k * 4;
        tau1 = c_idx + k * 4;
        v1 = tau1 + k;
        tau2 = v1 + k * 3;
        v2 = tau2 + k;
        tt = v2 + k * 3;
        ltau1 = tt + k * 16;
        vloc1 = ltau1 + k;
        ltau2 = vloc1 + k * 2;
        vloc2 = ltau2 + k;
        w = vloc2 + k * 2;
        we = tau1;
        mn = mn + k * 42;
        ldwke = k * 68 - 49;
    } else {
        b_idx = a_idx;
        c_idx = a_idx;
        tau = a_idx;
        v = a_idx;
        tt = a_idx;
        w = a_idx;
        we = a_idx;
        ldwke = 0;
        tau1 = 0; v1 = 0; tau2 = 0; v2 = 0;
        ltau = 0; vloc = 0;
        ltau1 = 0; vloc1 = 0; ltau2 = 0; vloc2 = 0;
    }

    minwrk = (ldwke + we > mn) ? (ldwke + we) : mn;

    if (ldwork < minwrk) {
        dwork[0] = (f64)minwrk;
        *info = -22;
        i32 neg_info = 22;
        SLC_XERBLA("MB03KB", &neg_info);
        return;
    } else if (nc <= 1 || n1 <= 0 || n2 <= 0 || n1 > nc || j2 > nc || j2 + n2 - 1 > nc) {
        return;
    }

    wantq = (compq[0] == 'U' || compq[0] == 'u');
    specq = (compq[0] == 'W' || compq[0] == 'w');

    j2 = j1 + 1;
    j3 = j1 + 2;
    j4 = j1 + 3;

    ia = a_idx;
    ib = b_idx;
    ic = c_idx;
    nd = n1 + n2;
    nd2 = nd * nd;

    dnrm = ZERO;

    for (i = 0; i < k; i++) {
        it = iwork[i11 + i];
        is = iwork[i12 + i];
        iq = iwork[i22 + i];

        tmp = SLC_DLANTR("F", "U", "N", &nd, &nd, &t[it], &ldt[i], dwork);
        if (i == kschur - 1) {
            if (n1 == 2) {
                tmp = SLC_DLAPY2(&t[it + 1], &tmp);
            }
            if (n2 == 2) {
                tmp = SLC_DLAPY2(&t[iq + 1], &tmp);
            }
        }
        dnrm = SLC_DLAPY2(&dnrm, &tmp);
        tmp = (tmp > smlnum) ? tmp : smlnum;

        if (n1 == 1) {
            dwork[ia] = t[it] / tmp;
            dwork[ic] = t[is] / tmp;
            if (n2 == 1) {
                dwork[ib] = t[iq] / tmp;
            } else {
                SLC_DLACPY("A", &n2, &n2, &t[iq], &ldt[i], &dwork[ib], &n2);
                SLC_DLASCL("G", &int1, &int1, &tmp, &ONE, &n2, &n2, &dwork[ib], &n2, info);
                dwork[ic + 1] = t[is + ldt[i]] / tmp;
            }
        } else {
            SLC_DLACPY("A", &n1, &n1, &t[it], &ldt[i], &dwork[ia], &n1);
            SLC_DLASCL("G", &int1, &int1, &tmp, &ONE, &n1, &n1, &dwork[ia], &n1, info);
            if (n2 == 1) {
                dwork[ib] = t[iq] / tmp;
                dwork[ic] = t[is] / tmp;
                dwork[ic + 1] = t[is + 1] / tmp;
            } else {
                SLC_DLACPY("A", &n2, &n2, &t[iq], &ldt[i], &dwork[ib], &n2);
                SLC_DLASCL("G", &int1, &int1, &tmp, &ONE, &n2, &n2, &dwork[ib], &n2, info);
                SLC_DLACPY("A", &n1, &n2, &t[is], &ldt[i], &dwork[ic], &n1);
                SLC_DLASCL("G", &int1, &int1, &tmp, &ONE, &n1, &n2, &dwork[ic], &n1, info);
            }
        }
        ia = ia + n1 * n1;
        ib = ib + n2 * n2;
        ic = ic + n1 * n2;
    }

    thresh = tol[0] * eps * dnrm;
    if (thresh < smlnum) thresh = smlnum;

    mb03ke(false, false, -1, k, n1, n2, eps, smlnum, s,
           &dwork[a_idx], &dwork[b_idx], &dwork[c_idx], &scaloc,
           &dwork[we], ldwork - we + 1, info);

    l = n1 + n1 + n2 - 2;

    if (l == 1) {
        indxc = c_idx;
        indxv = v;

        for (indtau = tau; indtau < tau + k; indtau++) {
            x_11 = dwork[indxc];
            dwork[indxv] = x_11;
            dwork[indxv + 1] = scaloc;
            SLC_DLARFG(&int2, &dwork[indxv], &dwork[indxv + 1], &int1, &dwork[indtau]);
            dwork[indxv] = ONE;

            tauloc = dwork[indtau];
            tmp = scaloc * (ONE - tauloc) + tauloc * dwork[indxv + 1] * x_11;
            if (fabs(tmp) > thresh) {
                *info = 1;
                return;
            }
            indxc = indxc + 1;
            indxv = indxv + 2;
        }

        if (ws) {
            indtau = tau;
            indxv = v;
            indtt = tt;

            for (i = 0; i < k; i++) {
                ip1 = (i + 1) % k;
                indvp1 = v + ip1 * 2;
                itaup1 = tau + ip1;
                SLC_DLACPY("A", &int2, &int2, &t[iwork[i11 + i]], &ldt[i], temp, &int2);
                SLC_DLACPY("A", &int2, &int2, temp, &int2, &dwork[indtt], &int2);

                if (s[i] == 1) {
                    SLC_DLARFX("L", &int2, &int2, &dwork[indvp1], &dwork[itaup1],
                              &dwork[indtt], &int2, &dwork[w]);
                    SLC_DLARFX("R", &int2, &int2, &dwork[indxv], &dwork[indtau],
                              &dwork[indtt], &int2, &dwork[w]);
                } else {
                    SLC_DLARFX("R", &int2, &int2, &dwork[indvp1], &dwork[itaup1],
                              &dwork[indtt], &int2, &dwork[w]);
                    SLC_DLARFX("L", &int2, &int2, &dwork[indxv], &dwork[indtau],
                              &dwork[indtt], &int2, &dwork[w]);
                }

                for (ii = 0; ii < 16; ii++) tempm1[ii] = dwork[indtt + (ii < 4 ? ii : 0)];
                SLC_DLACPY("A", &int2, &int2, &dwork[indtt], &int2, tempm1, &int2);

                if (s[i] == 1) {
                    SLC_DLARFX("L", &int2, &int2, &dwork[indvp1], &dwork[itaup1],
                              tempm1, &int2, &dwork[w]);
                    SLC_DLARFX("R", &int2, &int2, &dwork[indxv], &dwork[indtau],
                              tempm1, &int2, &dwork[w]);
                } else {
                    SLC_DLARFX("R", &int2, &int2, &dwork[indvp1], &dwork[itaup1],
                              tempm1, &int2, &dwork[w]);
                    SLC_DLARFX("L", &int2, &int2, &dwork[indxv], &dwork[indtau],
                              tempm1, &int2, &dwork[w]);
                }
                SLC_DAXPY(&nd2, &neg_one, temp, &int1, tempm1, &int1);
                strong = SLC_DLANGE("F", &int2, &int2, tempm1, &int2, dwork);
                if (strong > thresh) {
                    *info = 1;
                    return;
                }

                indtau = indtau + 1;
                indxv = indxv + 2;
                indtt = indtt + 4;
            }
        }

        indtau = tau;
        indxv = v;

        for (i = 0; i < k; i++) {
            ip1 = (i + 1) % k;
            indvp1 = v + ip1 * 2;
            itaup1 = tau + ip1;

            i32 ip1_1 = ip1 + 1;
            it = iwork[i11 + i] - j1 + 1;
            i32 ncols, nrows;

            if (s[i] == 1) {
                ncols = n[i] - j1 + 1;
                SLC_DLARFX("L", &int2, &ncols, &dwork[indvp1], &dwork[itaup1],
                          &t[iwork[i11 + i]], &ldt[i], &dwork[w]);
                nrows = ni[ip1_1 - 1] + j2;
                SLC_DLARFX("R", &nrows, &int2, &dwork[indxv], &dwork[indtau],
                          &t[it - ni[ip1_1 - 1]], &ldt[i], &dwork[w]);
            } else {
                ncols = n[ip1_1 - 1] - j1 + 1;
                SLC_DLARFX("L", &int2, &ncols, &dwork[indxv], &dwork[indtau],
                          &t[iwork[i11 + i]], &ldt[i], &dwork[w]);
                nrows = ni[i] + j2;
                SLC_DLARFX("R", &nrows, &int2, &dwork[indvp1], &dwork[itaup1],
                          &t[it - ni[i]], &ldt[i], &dwork[w]);
            }

            t[iwork[i21 + i]] = ZERO;

            wantql = wantq;
            if (specq) wantql = (whichq[i] != 0);
            if (wantql) {
                iq = ixq[i] - 1 + (j1 - 1) * ldq[i];
                i32 nrows_q = n[i];
                SLC_DLARFX("R", &nrows_q, &int2, &dwork[indxv], &dwork[indtau],
                          &q[iq], &ldq[i], &dwork[w]);
            }

            indtau = indtau + 1;
            indxv = indxv + 2;
        }
    } else if (l == 2) {
        itau2 = tau2;
        indxc = c_idx;
        indv1 = v1;
        indv2 = v2;

        for (itau1 = tau1; itau1 < tau1 + k; itau1++) {
            x_11 = dwork[indxc];
            x_12 = dwork[indxc + 1];
            dwork[indv1] = x_11;
            dwork[indv1 + 1] = scaloc;
            SLC_DLARFG(&int2, &dwork[indv1], &dwork[indv1 + 1], &int1, &dwork[itau1]);
            dwork[indv1] = ONE;

            dwork[indv2] = x_12;
            dwork[indv2 + 1] = ZERO;
            SLC_DLARFX("L", &int2, &int1, &dwork[indv1], &dwork[itau1],
                      &dwork[indv2], &int2, &dwork[w]);
            dwork[indv2] = dwork[indv2 + 1];
            dwork[indv2 + 1] = scaloc;
            SLC_DLARFG(&int2, &dwork[indv2], &dwork[indv2 + 1], &int1, &dwork[itau2]);
            dwork[indv2] = ONE;

            taus[0] = dwork[itau1];
            taus[1] = dwork[itau2];
            v_1 = dwork[indv1 + 1];
            tmp1 = scaloc * (ONE - taus[0]) + taus[0] * v_1 * x_11;
            tmp2 = -(scaloc * taus[0] * v_1 + x_11 * (ONE - taus[0] * v_1 * v_1)) *
                   (ONE - taus[1]) + taus[1] * dwork[indv2 + 1] * x_12;
            if (SLC_DLAPY2(&tmp1, &tmp2) > thresh) {
                *info = 1;
                return;
            }

            itau2 = itau2 + 1;
            indxc = indxc + 2;
            indv1 = indv1 + 2;
            indv2 = indv2 + 2;
        }

        itau1 = tau1;
        itau2 = tau2;
        indv1 = v1;
        indv2 = v2;
        indtt = tt;
        ltt = 3;

        for (i = 0; i < k; i++) {
            ip1 = (i + 1) % k;
            iv1p1 = v1 + ip1 * 2;
            iv2p1 = v2 + ip1 * 2;
            tau1p1 = tau1 + ip1;
            tau2p1 = tau2 + ip1;
            SLC_DLACPY("A", &int3, &int3, &t[iwork[i11 + i]], &ldt[i], &dwork[indtt], &int3);

            if (s[i] == 1) {
                SLC_DLARFX("L", &int2, &int3, &dwork[iv1p1], &dwork[tau1p1],
                          &dwork[indtt], &int3, &dwork[w]);
                SLC_DLARFX("L", &int2, &int3, &dwork[iv2p1], &dwork[tau2p1],
                          &dwork[indtt + 1], &int3, &dwork[w]);
                SLC_DLARFX("R", &int3, &int2, &dwork[indv1], &dwork[itau1],
                          &dwork[indtt], &int3, &dwork[w]);
                SLC_DLARFX("R", &int3, &int2, &dwork[indv2], &dwork[itau2],
                          &dwork[indtt + 3], &int3, &dwork[w]);
            } else {
                SLC_DLARFX("R", &int3, &int2, &dwork[iv1p1], &dwork[tau1p1],
                          &dwork[indtt], &int3, &dwork[w]);
                SLC_DLARFX("R", &int3, &int2, &dwork[iv2p1], &dwork[tau2p1],
                          &dwork[indtt + 3], &int3, &dwork[w]);
                SLC_DLARFX("L", &int2, &int3, &dwork[indv1], &dwork[itau1],
                          &dwork[indtt], &int3, &dwork[w]);
                SLC_DLARFX("L", &int2, &int3, &dwork[indv2], &dwork[itau2],
                          &dwork[indtt + 1], &int3, &dwork[w]);
            }

            itau1 = itau1 + 1;
            itau2 = itau2 + 1;
            indv1 = indv1 + 2;
            indv2 = indv2 + 2;
            indtt = indtt + 9;
        }

        fillin = false;
        indtt = tt + 1;
        for (i = 0; i < k; i++) {
            if (i != kschur - 1 && fabs(dwork[indtt]) > thresh) fillin = true;
            indtt = indtt + 9;
        }

        if (fillin) {
            mb03kc(k, kschur, ltt, 1, s, &dwork[tt], ltt, &dwork[vloc], &dwork[ltau]);
        }

        if (ws) {
            itau1 = tau1;
            itau2 = tau2;
            itauf = ltau;
            indv1 = v1;
            indv2 = v2;
            indvf = vloc;
            indtt = tt;

            for (i = 0; i < k; i++) {
                ip1 = (i + 1) % k;
                SLC_DLACPY("A", &int3, &int3, &dwork[indtt], &int3, tempm1, &int3);

                if (fillin) {
                    indvp1 = vloc + ip1 * 2;
                    itaup1 = ltau + ip1;

                    if (s[i] == 1) {
                        SLC_DLARFX("L", &int2, &int3, &dwork[indvp1], &dwork[itaup1],
                                  tempm1, &int3, &dwork[w]);
                        SLC_DLARFX("R", &int3, &int2, &dwork[indvf], &dwork[itauf],
                                  tempm1, &int3, &dwork[w]);
                    } else {
                        SLC_DLARFX("R", &int3, &int2, &dwork[indvp1], &dwork[itaup1],
                                  tempm1, &int3, &dwork[w]);
                        SLC_DLARFX("L", &int2, &int3, &dwork[indvf], &dwork[itauf],
                                  tempm1, &int3, &dwork[w]);
                    }
                }

                iv1p1 = v1 + ip1 * 2;
                iv2p1 = v2 + ip1 * 2;
                tau1p1 = tau1 + ip1;
                tau2p1 = tau2 + ip1;

                if (s[i] == 1) {
                    SLC_DLARFX("L", &int2, &int3, &dwork[iv2p1], &dwork[tau2p1],
                              &tempm1[1], &int3, &dwork[w]);
                    SLC_DLARFX("L", &int2, &int3, &dwork[iv1p1], &dwork[tau1p1],
                              tempm1, &int3, &dwork[w]);
                    SLC_DLARFX("R", &int3, &int2, &dwork[indv2], &dwork[itau2],
                              &tempm1[3], &int3, &dwork[w]);
                    SLC_DLARFX("R", &int3, &int2, &dwork[indv1], &dwork[itau1],
                              tempm1, &int3, &dwork[w]);
                } else {
                    SLC_DLARFX("R", &int3, &int2, &dwork[iv2p1], &dwork[tau2p1],
                              &tempm1[3], &int3, &dwork[w]);
                    SLC_DLARFX("R", &int3, &int2, &dwork[iv1p1], &dwork[tau1p1],
                              tempm1, &int3, &dwork[w]);
                    SLC_DLARFX("L", &int2, &int3, &dwork[indv2], &dwork[itau2],
                              &tempm1[1], &int3, &dwork[w]);
                    SLC_DLARFX("L", &int2, &int3, &dwork[indv1], &dwork[itau1],
                              tempm1, &int3, &dwork[w]);
                }

                SLC_DLACPY("A", &int3, &int3, &t[iwork[i11 + i]], &ldt[i], temp, &int3);
                SLC_DAXPY(&nd2, &neg_one, temp, &int1, tempm1, &int1);
                strong = SLC_DLANGE("F", &int3, &int3, tempm1, &int3, dwork);
                if (strong > thresh) {
                    *info = 1;
                    return;
                }

                itau1 = itau1 + 1;
                itau2 = itau2 + 1;
                itauf = itauf + 1;
                indv1 = indv1 + 2;
                indv2 = indv2 + 2;
                indvf = indvf + 2;
                indtt = indtt + 9;
            }
        }

        itau1 = tau1;
        itau2 = tau2;
        itauf = ltau;
        indv1 = v1;
        indv2 = v2;
        indvf = vloc;

        for (i = 0; i < k; i++) {
            ip1 = (i + 1) % k;
            it = iwork[i11 + i] - j1 + 1;

            iv1p1 = v1 + ip1 * 2;
            iv2p1 = v2 + ip1 * 2;
            tau1p1 = tau1 + ip1;
            tau2p1 = tau2 + ip1;
            i32 ip1_1 = ip1 + 1;

            i32 ncols, nrows;

            if (s[i] == 1) {
                it = it - ni[ip1_1 - 1];
                it2 = it + ldt[i];
                ncols = n[i] - j1 + 1;
                SLC_DLARFX("L", &int2, &ncols, &dwork[iv1p1], &dwork[tau1p1],
                          &t[iwork[i11 + i]], &ldt[i], &dwork[w]);
                SLC_DLARFX("L", &int2, &ncols, &dwork[iv2p1], &dwork[tau2p1],
                          &t[iwork[i21 + i]], &ldt[i], &dwork[w]);
                nrows = ni[ip1_1 - 1] + j3;
                SLC_DLARFX("R", &nrows, &int2, &dwork[indv1], &dwork[itau1],
                          &t[it], &ldt[i], &dwork[w]);
                SLC_DLARFX("R", &nrows, &int2, &dwork[indv2], &dwork[itau2],
                          &t[it2], &ldt[i], &dwork[w]);
            } else {
                it = it - ni[i];
                it2 = it + ldt[i];
                ncols = n[ip1_1 - 1] - j1 + 1;
                SLC_DLARFX("L", &int2, &ncols, &dwork[indv1], &dwork[itau1],
                          &t[iwork[i11 + i]], &ldt[i], &dwork[w]);
                SLC_DLARFX("L", &int2, &ncols, &dwork[indv2], &dwork[itau2],
                          &t[iwork[i21 + i]], &ldt[i], &dwork[w]);
                nrows = ni[i] + j3;
                SLC_DLARFX("R", &nrows, &int2, &dwork[iv1p1], &dwork[tau1p1],
                          &t[it], &ldt[i], &dwork[w]);
                SLC_DLARFX("R", &nrows, &int2, &dwork[iv2p1], &dwork[tau2p1],
                          &t[it2], &ldt[i], &dwork[w]);
            }

            wantql = wantq;
            if (specq) wantql = (whichq[i] != 0);
            if (wantql) {
                iq = ixq[i] - 1 + (j1 - 1) * ldq[i];
                i32 nrows_q = n[i];
                SLC_DLARFX("R", &nrows_q, &int2, &dwork[indv1], &dwork[itau1],
                          &q[iq], &ldq[i], &dwork[w]);
                iq = iq + ldq[i];
                SLC_DLARFX("R", &nrows_q, &int2, &dwork[indv2], &dwork[itau2],
                          &q[iq], &ldq[i], &dwork[w]);
            }

            if (fillin) {
                iv1p1 = vloc + ip1 * 2;
                tau1p1 = ltau + ip1;

                if (s[i] == 1) {
                    ncols = n[i] - j1 + 1;
                    SLC_DLARFX("L", &int2, &ncols, &dwork[iv1p1], &dwork[tau1p1],
                              &t[iwork[i11 + i]], &ldt[i], &dwork[w]);
                    nrows = ni[ip1_1 - 1] + j2;
                    SLC_DLARFX("R", &nrows, &int2, &dwork[indvf], &dwork[itauf],
                              &t[it], &ldt[i], &dwork[w]);
                } else {
                    nrows = ni[i] + j2;
                    SLC_DLARFX("R", &nrows, &int2, &dwork[iv1p1], &dwork[tau1p1],
                              &t[it], &ldt[i], &dwork[w]);
                    ncols = n[ip1_1 - 1] - j1 + 1;
                    SLC_DLARFX("L", &int2, &ncols, &dwork[indvf], &dwork[itauf],
                              &t[iwork[i11 + i]], &ldt[i], &dwork[w]);
                }

                wantql = wantq;
                if (specq) wantql = (whichq[i] != 0);
                if (wantql) {
                    iq = ixq[i] - 1 + (j1 - 1) * ldq[i];
                    i32 nrows_q = n[i];
                    SLC_DLARFX("R", &nrows_q, &int2, &dwork[indvf], &dwork[itauf],
                              &q[iq], &ldq[i], &dwork[w]);
                }
            }

            itau1 = itau1 + 1;
            itau2 = itau2 + 1;
            itauf = itauf + 1;
            indv1 = indv1 + 2;
            indv2 = indv2 + 2;
            indvf = indvf + 2;
        }

        for (i = 0; i < k; i++) {
            t[iwork[i21 + i] + 1] = ZERO;
            t[iwork[i22 + i] + 1] = ZERO;
            if (i != kschur - 1) t[iwork[i21 + i]] = ZERO;
        }
    } else if (l == 3) {
        indxc = c_idx;
        indxv = v;

        for (indtau = tau; indtau < tau + k; indtau++) {
            x_11 = dwork[indxc];
            x_21 = dwork[indxc + 1];
            dwork[indxv] = x_11;
            dwork[indxv + 1] = x_21;
            dwork[indxv + 2] = scaloc;
            SLC_DLARFG(&int3, &dwork[indxv], &dwork[indxv + 1], &int1, &dwork[indtau]);
            dwork[indxv] = ONE;

            v_2 = dwork[indxv + 2];
            tauloc = dwork[indtau];
            tmp1 = scaloc * (ONE - tauloc) + tauloc * v_2 * x_11;
            tmp2 = tauloc * (v_2 * x_21 - scaloc * dwork[indxv + 1]);
            if (SLC_DLAPY2(&tmp1, &tmp2) > thresh) {
                *info = 1;
                return;
            }

            indxc = indxc + 2;
            indxv = indxv + 3;
        }

        indtau = tau;
        indxv = v;
        indtt = tt;
        ltt = 3;

        for (i = 0; i < k; i++) {
            ip1 = (i + 1) % k;
            indvp1 = v + ip1 * 3;
            itaup1 = tau + ip1;
            SLC_DLACPY("A", &int3, &int3, &t[iwork[i11 + i]], &ldt[i], &dwork[indtt], &int3);

            if (s[i] == 1) {
                SLC_DLARFX("L", &int3, &int3, &dwork[indvp1], &dwork[itaup1],
                          &dwork[indtt], &int3, &dwork[w]);
                SLC_DLARFX("R", &int3, &int3, &dwork[indxv], &dwork[indtau],
                          &dwork[indtt], &int3, &dwork[w]);
            } else {
                SLC_DLARFX("R", &int3, &int3, &dwork[indvp1], &dwork[itaup1],
                          &dwork[indtt], &int3, &dwork[w]);
                SLC_DLARFX("L", &int3, &int3, &dwork[indxv], &dwork[indtau],
                          &dwork[indtt], &int3, &dwork[w]);
            }

            indtau = indtau + 1;
            indxv = indxv + 3;
            indtt = indtt + 9;
        }

        fillin = false;
        indtt = tt + 5;
        for (i = 0; i < k; i++) {
            if (i != kschur - 1 && fabs(dwork[indtt]) > thresh) fillin = true;
            indtt = indtt + 9;
        }

        if (fillin) {
            mb03kc(k, kschur, ltt, 2, s, &dwork[tt], ltt, &dwork[vloc], &dwork[ltau]);
        }

        if (ws) {
            indtau = tau;
            indxv = v;
            itauf = ltau;
            indvf = vloc;
            indtt = tt;

            for (i = 0; i < k; i++) {
                ip1 = (i + 1) % k;
                SLC_DLACPY("A", &int3, &int3, &dwork[indtt], &int3, tempm1, &int3);

                if (fillin) {
                    indvp1 = vloc + ip1 * 2;
                    itaup1 = ltau + ip1;

                    if (s[i] == 1) {
                        SLC_DLARFX("L", &int2, &int2, &dwork[indvp1], &dwork[itaup1],
                                  &tempm1[4], &int3, &dwork[w]);
                        SLC_DLARFX("R", &int3, &int2, &dwork[indvf], &dwork[itauf],
                                  &tempm1[3], &int3, &dwork[w]);
                    } else {
                        SLC_DLARFX("R", &int3, &int2, &dwork[indvp1], &dwork[itaup1],
                                  &tempm1[3], &int3, &dwork[w]);
                        SLC_DLARFX("L", &int2, &int2, &dwork[indvf], &dwork[itauf],
                                  &tempm1[4], &int3, &dwork[w]);
                    }
                }

                indvp1 = v + ip1 * 3;
                itaup1 = tau + ip1;

                if (s[i] == 1) {
                    SLC_DLARFX("L", &int3, &int3, &dwork[indvp1], &dwork[itaup1],
                              tempm1, &int3, &dwork[w]);
                    SLC_DLARFX("R", &int3, &int3, &dwork[indxv], &dwork[indtau],
                              tempm1, &int3, &dwork[w]);
                } else {
                    SLC_DLARFX("R", &int3, &int3, &dwork[indvp1], &dwork[itaup1],
                              tempm1, &int3, &dwork[w]);
                    SLC_DLARFX("L", &int3, &int3, &dwork[indxv], &dwork[indtau],
                              tempm1, &int3, &dwork[w]);
                }

                SLC_DLACPY("A", &int3, &int3, &t[iwork[i11 + i]], &ldt[i], temp, &int3);
                SLC_DAXPY(&nd2, &neg_one, temp, &int1, tempm1, &int1);
                strong = SLC_DLANGE("F", &int3, &int3, tempm1, &int3, dwork);
                if (strong > thresh) {
                    *info = 1;
                    return;
                }

                indtau = indtau + 1;
                indxv = indxv + 3;
                itauf = itauf + 1;
                indvf = indvf + 2;
                indtt = indtt + 9;
            }
        }

        indtau = tau;
        indxv = v;
        itauf = ltau;
        indvf = vloc;

        for (i = 0; i < k; i++) {
            ip1 = (i + 1) % k;
            it = iwork[i11 + i] - j1 + 1;

            indvp1 = v + ip1 * 3;
            itaup1 = tau + ip1;
            i32 ip1_1 = ip1 + 1;

            i32 ncols, nrows;

            if (s[i] == 1) {
                it = it - ni[ip1_1 - 1];
                ncols = n[i] - j1 + 1;
                SLC_DLARFX("L", &int3, &ncols, &dwork[indvp1], &dwork[itaup1],
                          &t[iwork[i11 + i]], &ldt[i], &dwork[w]);
                nrows = ni[ip1_1 - 1] + j3;
                SLC_DLARFX("R", &nrows, &int3, &dwork[indxv], &dwork[indtau],
                          &t[it], &ldt[i], &dwork[w]);
            } else {
                it = it - ni[i];
                ncols = n[ip1_1 - 1] - j1 + 1;
                SLC_DLARFX("L", &int3, &ncols, &dwork[indxv], &dwork[indtau],
                          &t[iwork[i11 + i]], &ldt[i], &dwork[w]);
                nrows = ni[i] + j3;
                SLC_DLARFX("R", &nrows, &int3, &dwork[indvp1], &dwork[itaup1],
                          &t[it], &ldt[i], &dwork[w]);
            }

            wantql = wantq;
            if (specq) wantql = (whichq[i] != 0);
            if (wantql) {
                iq = ixq[i] - 1 + (j1 - 1) * ldq[i];
                i32 nrows_q = n[i];
                SLC_DLARFX("R", &nrows_q, &int3, &dwork[indxv], &dwork[indtau],
                          &q[iq], &ldq[i], &dwork[w]);
            }

            if (fillin) {
                indvp1 = vloc + ip1 * 2;
                itaup1 = ltau + ip1;
                it2 = it + ldt[i];

                if (s[i] == 1) {
                    ncols = n[i] - j1;
                    SLC_DLARFX("L", &int2, &ncols, &dwork[indvp1], &dwork[itaup1],
                              &t[it2 + j1], &ldt[i], &dwork[w]);
                    nrows = ni[ip1_1 - 1] + j3;
                    SLC_DLARFX("R", &nrows, &int2, &dwork[indvf], &dwork[itauf],
                              &t[it2], &ldt[i], &dwork[w]);
                } else {
                    ncols = n[ip1_1 - 1] - j1;
                    SLC_DLARFX("L", &int2, &ncols, &dwork[indvf], &dwork[itauf],
                              &t[it2 + j1], &ldt[i], &dwork[w]);
                    nrows = ni[i] + j3;
                    SLC_DLARFX("R", &nrows, &int2, &dwork[indvp1], &dwork[itaup1],
                              &t[it2], &ldt[i], &dwork[w]);
                }

                wantql = wantq;
                if (specq) wantql = (whichq[i] != 0);
                if (wantql) {
                    iq = ixq[i] - 1 + j1 * ldq[i];
                    i32 nrows_q = n[i];
                    SLC_DLARFX("R", &nrows_q, &int2, &dwork[indvf], &dwork[itauf],
                              &q[iq], &ldq[i], &dwork[w]);
                }
            }

            indtau = indtau + 1;
            indxv = indxv + 3;
            itauf = itauf + 1;
            indvf = indvf + 2;
        }

        for (i = 0; i < k; i++) {
            it = iwork[i11 + i] + 1;
            t[it] = ZERO;
            t[it + 1] = ZERO;
            if (i != kschur - 1) t[it + ldt[i] + 1] = ZERO;
        }
    } else if (l == 4) {
        indxc = c_idx;
        itau2 = tau2;
        indv1 = v1;
        indv2 = v2;

        for (itau1 = tau1; itau1 < tau1 + k; itau1++) {
            x_11 = dwork[indxc];
            x_21 = dwork[indxc + 1];
            x_12 = dwork[indxc + 2];
            x_22 = dwork[indxc + 3];

            dwork[indv1] = x_11;
            dwork[indv1 + 1] = x_21;
            dwork[indv1 + 2] = scaloc;
            SLC_DLARFG(&int3, &dwork[indv1], &dwork[indv1 + 1], &int1, &dwork[itau1]);
            dwork[indv1] = ONE;

            dwork[indv2] = x_12;
            dwork[indv2 + 1] = x_22;
            dwork[indv2 + 2] = ZERO;
            SLC_DLARFX("L", &int3, &int1, &dwork[indv1], &dwork[itau1],
                      &dwork[indv2], &int3, &dwork[w]);
            dwork[indv2] = dwork[indv2 + 1];
            dwork[indv2 + 1] = dwork[indv2 + 2];
            dwork[indv2 + 2] = scaloc;
            SLC_DLARFG(&int3, &dwork[indv2], &dwork[indv2 + 1], &int1, &dwork[itau2]);
            dwork[indv2] = ONE;

            v_2 = dwork[indv1 + 1];
            v_3 = dwork[indv1 + 2];
            w_2 = dwork[indv2 + 1];
            w_3 = dwork[indv2 + 2];
            dtau1 = dwork[itau1];
            dtau2 = dwork[itau2];

            temp[0] = scaloc * (ONE - dtau1) + x_11 * dtau1 * v_3;
            temp[2] = scaloc * (dtau2 * w_2 * dtau1 * v_3 - dtau1 * v_2 * (ONE - dtau2)) -
                      x_11 * (-dtau1 * v_2 * v_3 * (ONE - dtau2) -
                      (ONE - dtau1 * v_3 * v_3) * dtau2 * w_2) + x_12 * dtau2 * w_3;
            temp[1] = -scaloc * dtau1 * v_2 + x_21 * dtau1 * v_3;
            temp[3] = scaloc * ((ONE - dtau1 * v_2 * v_2) * (ONE - dtau2) +
                      dtau1 * v_2 * v_3 * dtau2 * w_2) -
                      x_21 * (-dtau1 * v_2 * v_3 * (ONE - dtau2) -
                      (ONE - dtau1 * v_3 * v_3) * dtau2 * w_2) + x_22 * dtau2 * w_3;

            if (SLC_DLANGE("F", &int2, &int2, temp, &int2, dwork) > thresh) {
                *info = 1;
                return;
            }

            indxc = indxc + 4;
            itau2 = itau2 + 1;
            indv1 = indv1 + 3;
            indv2 = indv2 + 3;
        }

        itau1 = tau1;
        itau2 = tau2;
        indv1 = v1;
        indv2 = v2;
        indtt = tt;
        ltt = 4;

        for (i = 0; i < k; i++) {
            ip1 = (i + 1) % k;
            iv1p1 = v1 + ip1 * 3;
            iv2p1 = v2 + ip1 * 3;
            tau1p1 = tau1 + ip1;
            tau2p1 = tau2 + ip1;
            SLC_DLACPY("A", &int4, &int4, &t[iwork[i11 + i]], &ldt[i], &dwork[indtt], &int4);

            if (s[i] == 1) {
                SLC_DLARFX("L", &int3, &int4, &dwork[iv1p1], &dwork[tau1p1],
                          &dwork[indtt], &int4, &dwork[w]);
                SLC_DLARFX("L", &int3, &int4, &dwork[iv2p1], &dwork[tau2p1],
                          &dwork[indtt + 1], &int4, &dwork[w]);
                SLC_DLARFX("R", &int4, &int3, &dwork[indv1], &dwork[itau1],
                          &dwork[indtt], &int4, &dwork[w]);
                SLC_DLARFX("R", &int4, &int3, &dwork[indv2], &dwork[itau2],
                          &dwork[indtt + 4], &int4, &dwork[w]);
            } else {
                SLC_DLARFX("R", &int4, &int3, &dwork[iv1p1], &dwork[tau1p1],
                          &dwork[indtt], &int4, &dwork[w]);
                SLC_DLARFX("R", &int4, &int3, &dwork[iv2p1], &dwork[tau2p1],
                          &dwork[indtt + 4], &int4, &dwork[w]);
                SLC_DLARFX("L", &int3, &int4, &dwork[indv1], &dwork[itau1],
                          &dwork[indtt], &int4, &dwork[w]);
                SLC_DLARFX("L", &int3, &int4, &dwork[indv2], &dwork[itau2],
                          &dwork[indtt + 1], &int4, &dwork[w]);
            }

            itau1 = itau1 + 1;
            itau2 = itau2 + 1;
            indv1 = indv1 + 3;
            indv2 = indv2 + 3;
            indtt = indtt + 16;
        }

        fillin = false;
        fill21 = false;
        indtt = tt + 1;
        for (i = 0; i < k; i++) {
            if (i != kschur - 1 && fabs(dwork[indtt]) > thresh) {
                fillin = true;
                fill21 = true;
            }
            indtt = indtt + 16;
        }

        if (fillin) {
            mb03kc(k, kschur, ltt, 1, s, &dwork[tt], ltt, &dwork[vloc1], &dwork[ltau1]);
        }

        fillin = false;
        fill43 = false;
        indtt = tt + 11;
        for (i = 0; i < k; i++) {
            if (i != kschur - 1 && fabs(dwork[indtt]) > eps) {
                fillin = true;
                fill43 = true;
            }
            indtt = indtt + 16;
        }

        if (fillin) {
            mb03kc(k, kschur, ltt, 3, s, &dwork[tt], ltt, &dwork[vloc2], &dwork[ltau2]);
        }

        fillin = fill21 || fill43;

        if (ws) {
            itau1 = tau1;
            itau2 = tau2;
            indv1 = v1;
            indv2 = v2;
            indtt = tt;

            itauf1 = ltau1;
            itauf2 = ltau2;
            indf1 = vloc1;
            indf2 = vloc2;

            for (i = 0; i < k; i++) {
                ip1 = (i + 1) % k;
                SLC_DLACPY("A", &int4, &int4, &dwork[indtt], &int4, tempm1, &int4);

                if (fillin) {
                    iv1p1 = vloc1 + ip1 * 2;
                    iv2p1 = vloc2 + ip1 * 2;
                    tau1p1 = ltau1 + ip1;
                    tau2p1 = ltau2 + ip1;

                    if (fill21) {
                        if (s[i] == 1) {
                            SLC_DLARFX("L", &int2, &int4, &dwork[iv1p1], &dwork[tau1p1],
                                      tempm1, &int4, &dwork[w]);
                            SLC_DLARFX("R", &int2, &int2, &dwork[indf1], &dwork[itauf1],
                                      tempm1, &int4, &dwork[w]);
                        } else {
                            SLC_DLARFX("R", &int2, &int2, &dwork[iv1p1], &dwork[tau1p1],
                                      tempm1, &int4, &dwork[w]);
                            SLC_DLARFX("L", &int2, &int4, &dwork[indf1], &dwork[itauf1],
                                      tempm1, &int4, &dwork[w]);
                        }
                    }

                    if (fill43) {
                        if (s[i] == 1) {
                            SLC_DLARFX("L", &int2, &int2, &dwork[iv2p1], &dwork[tau2p1],
                                      &tempm1[10], &int4, &dwork[w]);
                            SLC_DLARFX("R", &int4, &int2, &dwork[indf2], &dwork[itauf2],
                                      &tempm1[8], &int4, &dwork[w]);
                        } else {
                            SLC_DLARFX("R", &int4, &int2, &dwork[iv2p1], &dwork[tau2p1],
                                      &tempm1[8], &int4, &dwork[w]);
                            SLC_DLARFX("L", &int2, &int2, &dwork[indf2], &dwork[itauf2],
                                      &tempm1[10], &int4, &dwork[w]);
                        }
                    }
                }

                iv1p1 = v1 + ip1 * 3;
                iv2p1 = v2 + ip1 * 3;
                tau1p1 = tau1 + ip1;
                tau2p1 = tau2 + ip1;

                if (s[i] == 1) {
                    SLC_DLARFX("L", &int3, &int4, &dwork[iv2p1], &dwork[tau2p1],
                              &tempm1[1], &int4, &dwork[w]);
                    SLC_DLARFX("L", &int3, &int4, &dwork[iv1p1], &dwork[tau1p1],
                              tempm1, &int4, &dwork[w]);
                    SLC_DLARFX("R", &int4, &int3, &dwork[indv2], &dwork[itau2],
                              &tempm1[4], &int4, &dwork[w]);
                    SLC_DLARFX("R", &int4, &int3, &dwork[indv1], &dwork[itau1],
                              tempm1, &int4, &dwork[w]);
                } else {
                    SLC_DLARFX("R", &int4, &int3, &dwork[iv2p1], &dwork[tau2p1],
                              &tempm1[4], &int4, &dwork[w]);
                    SLC_DLARFX("R", &int4, &int3, &dwork[iv1p1], &dwork[tau1p1],
                              tempm1, &int4, &dwork[w]);
                    SLC_DLARFX("L", &int3, &int4, &dwork[indv2], &dwork[itau2],
                              &tempm1[1], &int4, &dwork[w]);
                    SLC_DLARFX("L", &int3, &int4, &dwork[indv1], &dwork[itau1],
                              tempm1, &int4, &dwork[w]);
                }

                SLC_DLACPY("A", &int4, &int4, &t[iwork[i11 + i]], &ldt[i], temp, &int4);
                SLC_DAXPY(&nd2, &neg_one, temp, &int1, tempm1, &int1);
                strong = SLC_DLANGE("F", &int4, &int4, tempm1, &int4, dwork);
                if (strong > thresh) {
                    *info = 1;
                    return;
                }

                itau1 = itau1 + 1;
                itau2 = itau2 + 1;
                indv1 = indv1 + 3;
                indv2 = indv2 + 3;
                indtt = indtt + 16;

                if (fillin) {
                    itauf1 = itauf1 + 1;
                    itauf2 = itauf2 + 1;
                    indf1 = indf1 + 2;
                    indf2 = indf2 + 2;
                }
            }
        }

        itau1 = tau1;
        itau2 = tau2;
        indv1 = v1;
        indv2 = v2;

        if (fillin) {
            itauf1 = ltau1;
            itauf2 = ltau2;
            indf1 = vloc1;
            indf2 = vloc2;
        }

        for (i = 0; i < k; i++) {
            ip1 = (i + 1) % k;
            ipp = ip1 + 1;
            it = iwork[i11 + i] - j1 + 1;

            iv1p1 = v1 + ip1 * 3;
            iv2p1 = v2 + ip1 * 3;
            tau1p1 = tau1 + ip1;
            tau2p1 = tau2 + ip1;

            i32 ncols, nrows;

            if (s[i] == 1) {
                it = it - ni[ipp - 1];
                it2 = it + ldt[i];
                ncols = n[i] - j1 + 1;
                SLC_DLARFX("L", &int3, &ncols, &dwork[iv1p1], &dwork[tau1p1],
                          &t[iwork[i11 + i]], &ldt[i], &dwork[w]);
                SLC_DLARFX("L", &int3, &ncols, &dwork[iv2p1], &dwork[tau2p1],
                          &t[iwork[i11 + i] + 1], &ldt[i], &dwork[w]);
                nrows = ni[ipp - 1] + j4;
                SLC_DLARFX("R", &nrows, &int3, &dwork[indv1], &dwork[itau1],
                          &t[it], &ldt[i], &dwork[w]);
                SLC_DLARFX("R", &nrows, &int3, &dwork[indv2], &dwork[itau2],
                          &t[it2], &ldt[i], &dwork[w]);
            } else {
                it = it - ni[i];
                it2 = it + ldt[i];
                nrows = ni[i] + j4;
                SLC_DLARFX("R", &nrows, &int3, &dwork[iv1p1], &dwork[tau1p1],
                          &t[it], &ldt[i], &dwork[w]);
                SLC_DLARFX("R", &nrows, &int3, &dwork[iv2p1], &dwork[tau2p1],
                          &t[it2], &ldt[i], &dwork[w]);
                ncols = n[ipp - 1] - j1 + 1;
                SLC_DLARFX("L", &int3, &ncols, &dwork[indv1], &dwork[itau1],
                          &t[iwork[i11 + i]], &ldt[i], &dwork[w]);
                SLC_DLARFX("L", &int3, &ncols, &dwork[indv2], &dwork[itau2],
                          &t[iwork[i11 + i] + 1], &ldt[i], &dwork[w]);
            }

            wantql = wantq;
            if (specq) wantql = (whichq[i] != 0);
            if (wantql) {
                iq = ixq[i] - 1 + (j1 - 1) * ldq[i];
                i32 nrows_q = n[i];
                SLC_DLARFX("R", &nrows_q, &int3, &dwork[indv1], &dwork[itau1],
                          &q[iq], &ldq[i], &dwork[w]);
                iq = iq + ldq[i];
                SLC_DLARFX("R", &nrows_q, &int3, &dwork[indv2], &dwork[itau2],
                          &q[iq], &ldq[i], &dwork[w]);
            }

            if (fillin) {
                iv1p1 = vloc1 + ip1 * 2;
                iv2p1 = vloc2 + ip1 * 2;
                tau1p1 = ltau1 + ip1;
                tau2p1 = ltau2 + ip1;

                if (fill21) {
                    if (s[i] == 1) {
                        ncols = n[i] - j1 + 1;
                        SLC_DLARFX("L", &int2, &ncols, &dwork[iv1p1], &dwork[tau1p1],
                                  &t[iwork[i11 + i]], &ldt[i], &dwork[w]);
                        nrows = ni[ipp - 1] + j2;
                        SLC_DLARFX("R", &nrows, &int2, &dwork[indf1], &dwork[itauf1],
                                  &t[it], &ldt[i], &dwork[w]);
                    } else {
                        nrows = ni[i] + j2;
                        SLC_DLARFX("R", &nrows, &int2, &dwork[iv1p1], &dwork[tau1p1],
                                  &t[it], &ldt[i], &dwork[w]);
                        ncols = n[ipp - 1] - j1 + 1;
                        SLC_DLARFX("L", &int2, &ncols, &dwork[indf1], &dwork[itauf1],
                                  &t[iwork[i11 + i]], &ldt[i], &dwork[w]);
                    }
                }

                if (fill43) {
                    it = iwork[i22 + i];
                    it2 = it2 + ldt[i];

                    if (s[i] == 1) {
                        ncols = n[i] - j2;
                        SLC_DLARFX("L", &int2, &ncols, &dwork[iv2p1], &dwork[tau2p1],
                                  &t[it], &ldt[i], &dwork[w]);
                        nrows = ni[ipp - 1] + j4;
                        SLC_DLARFX("R", &nrows, &int2, &dwork[indf2], &dwork[itauf2],
                                  &t[it2], &ldt[i], &dwork[w]);
                    } else {
                        nrows = ni[i] + j4;
                        SLC_DLARFX("R", &nrows, &int2, &dwork[iv2p1], &dwork[tau2p1],
                                  &t[it2], &ldt[i], &dwork[w]);
                        ncols = n[ipp - 1] - j2;
                        SLC_DLARFX("L", &int2, &ncols, &dwork[indf2], &dwork[itauf2],
                                  &t[it], &ldt[i], &dwork[w]);
                    }
                }

                wantql = wantq;
                if (specq) wantql = (whichq[i] != 0);
                if (wantql) {
                    if (fill21) {
                        iq = ixq[i] - 1 + (j1 - 1) * ldq[i];
                        i32 nrows_q = n[i];
                        SLC_DLARFX("R", &nrows_q, &int2, &dwork[indf1], &dwork[itauf1],
                                  &q[iq], &ldq[i], &dwork[w]);
                    }
                    if (fill43) {
                        iq = ixq[i] - 1 + j2 * ldq[i];
                        i32 nrows_q = n[i];
                        SLC_DLARFX("R", &nrows_q, &int2, &dwork[indf2], &dwork[itauf2],
                                  &q[iq], &ldq[i], &dwork[w]);
                    }
                }
            }

            itau1 = itau1 + 1;
            itau2 = itau2 + 1;
            indv1 = indv1 + 3;
            indv2 = indv2 + 3;

            if (fillin) {
                itauf1 = itauf1 + 1;
                itauf2 = itauf2 + 1;
                indf1 = indf1 + 2;
                indf2 = indf2 + 2;
            }
        }

        for (i = 0; i < k; i++) {
            it = iwork[i21 + i];
            t[it] = ZERO;
            t[it + 1] = ZERO;
            it = it + ldt[i];
            t[it] = ZERO;
            t[it + 1] = ZERO;
            if (i != kschur - 1) {
                t[iwork[i11 + i] + 1] = ZERO;
                t[iwork[i22 + i] + 1] = ZERO;
            }
        }
    }

    dwork[0] = (f64)minwrk;
}
