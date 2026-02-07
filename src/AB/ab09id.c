/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * AB09ID - Frequency-weighted model reduction based on balancing techniques
 *
 * Computes a reduced order model (Ar,Br,Cr,Dr) for an original state-space
 * representation (A,B,C,D) by using frequency weighted square-root or
 * balancing-free square-root Balance & Truncate (B&T) or Singular
 * Perturbation Approximation (SPA) model reduction methods.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

static inline bool lsame(char ca, char cb) {
    return (ca == cb) || (ca == cb + 32) || (ca == cb - 32);
}

void ab09id(
    const char* dico, const char* jobc, const char* jobo, const char* job,
    const char* weight, const char* equil, const char* ordsel,
    const i32 n, const i32 m, const i32 p, const i32 nv, const i32 pv,
    const i32 nw, const i32 mw, i32* nr, const f64 alpha, const f64 alphac,
    const f64 alphao, f64* a, const i32 lda, f64* b, const i32 ldb,
    f64* c, const i32 ldc, f64* d, const i32 ldd,
    f64* av, const i32 ldav, f64* bv, const i32 ldbv,
    f64* cv, const i32 ldcv, const f64* dv, const i32 lddv,
    f64* aw, const i32 ldaw, f64* bw, const i32 ldbw,
    f64* cw, const i32 ldcw, const f64* dw, const i32 lddw,
    i32* ns, f64* hsv, const f64 tol1, const f64 tol2,
    i32* iwarn, i32* info
)
{
    const f64 c100 = 100.0;
    const f64 one = 1.0;
    const f64 zero = 0.0;

    bool discr, bta, spa, scale, fixord, leftw, rightw, frwght;
    i32 iwarnl, kbr, kbv, kbw, kcr, kcv, kcw, kdr, kdv;
    i32 ki, kl, kt, kti, ku, kw, lcf, ldw, lw, nmr;
    i32 nn, nnq, nnr, nnv, nnw, nra, nu, nu1, nvr, nwr, ppv, wrkopt;
    f64 alpwrk, maxred, scalec, scaleo;

    *info = 0;
    *iwarn = 0;

    discr = lsame(dico[0], 'D');
    bta = lsame(job[0], 'B') || lsame(job[0], 'F');
    spa = lsame(job[0], 'S') || lsame(job[0], 'P');
    scale = lsame(equil[0], 'S');
    fixord = lsame(ordsel[0], 'F');
    leftw = lsame(weight[0], 'L') || lsame(weight[0], 'B');
    rightw = lsame(weight[0], 'R') || lsame(weight[0], 'B');
    frwght = leftw || rightw;

    lw = 1;
    nn = n * n;
    nnv = n + nv;
    nnw = n + nw;
    ppv = (p > pv) ? p : pv;

    if (leftw && pv > 0) {
        i32 term = nnv * (nnv + ((nnv > pv) ? nnv : pv) + 5);
        lw = (lw > term) ? lw : term;
    } else {
        i32 term = n * (p + 5);
        lw = (lw > term) ? lw : term;
    }

    if (rightw && mw > 0) {
        i32 term = nnw * (nnw + ((nnw > mw) ? nnw : mw) + 5);
        lw = (lw > term) ? lw : term;
    } else {
        i32 term = n * (m + 5);
        lw = (lw > term) ? lw : term;
    }
    lw = 2 * nn + ((lw > (2 * nn + 5 * n)) ? lw : (2 * nn + 5 * n));
    {
        i32 term = n * ((m > p) ? m : p);
        lw = (lw > term) ? lw : term;
    }

    if (leftw && nv > 0) {
        i32 nv5 = nv * (nv + 5);
        i32 pv2 = pv * (pv + 2);
        i32 p4 = 4 * ppv;
        i32 maxterm = (nv5 > pv2) ? nv5 : pv2;
        maxterm = (maxterm > p4) ? maxterm : p4;
        lcf = pv * (nv + pv) + pv * nv + maxterm;
        if (pv == p) {
            i32 term1 = nv + ((nv > 3 * p) ? nv : 3 * p);
            i32 term = (lcf > term1) ? lcf : term1;
            lw = (lw > term) ? lw : term;
        } else {
            i32 ppv3 = (nv > 3 * ppv) ? nv : 3 * ppv;
            i32 term = ppv * (2 * nv + ppv) + ((lcf > (nv + ppv3)) ? lcf : (nv + ppv3));
            lw = (lw > term) ? lw : term;
        }
    }

    if (rightw && nw > 0) {
        i32 m_mw = (m > mw) ? m : mw;
        if (mw == m) {
            i32 term = nw + ((nw > 3 * m) ? nw : 3 * m);
            lw = (lw > term) ? lw : term;
        } else {
            i32 mw3 = (nw > 3 * m_mw) ? nw : 3 * m_mw;
            i32 term = 2 * nw * m_mw + nw + mw3;
            lw = (lw > term) ? lw : term;
        }
        i32 nw5 = nw * (nw + 5);
        i32 mw2 = mw * (mw + 2);
        i32 mw4 = 4 * mw;
        i32 m4 = 4 * m;
        i32 maxterm = (nw5 > mw2) ? nw5 : mw2;
        maxterm = (maxterm > mw4) ? maxterm : mw4;
        maxterm = (maxterm > m4) ? maxterm : m4;
        i32 term = mw * (nw + mw) + maxterm;
        lw = (lw > term) ? lw : term;
    }

    if (!(lsame(dico[0], 'C') || discr)) {
        *info = -1;
    } else if (!(lsame(jobc[0], 'S') || lsame(jobc[0], 'E'))) {
        *info = -2;
    } else if (!(lsame(jobo[0], 'S') || lsame(jobo[0], 'E'))) {
        *info = -3;
    } else if (!(bta || spa)) {
        *info = -4;
    } else if (!(frwght || lsame(weight[0], 'N'))) {
        *info = -5;
    } else if (!(scale || lsame(equil[0], 'N'))) {
        *info = -6;
    } else if (!(fixord || lsame(ordsel[0], 'A'))) {
        *info = -7;
    } else if (n < 0) {
        *info = -8;
    } else if (m < 0) {
        *info = -9;
    } else if (p < 0) {
        *info = -10;
    } else if (nv < 0) {
        *info = -11;
    } else if (pv < 0) {
        *info = -12;
    } else if (nw < 0) {
        *info = -13;
    } else if (mw < 0) {
        *info = -14;
    } else if (fixord && (*nr < 0 || *nr > n)) {
        *info = -15;
    } else if ((discr && (alpha < zero || alpha > one)) ||
               (!discr && alpha > zero)) {
        *info = -16;
    } else if (fabs(alphac) > one) {
        *info = -17;
    } else if (fabs(alphao) > one) {
        *info = -18;
    } else if (lda < ((1 > n) ? 1 : n)) {
        *info = -20;
    } else if (ldb < ((1 > n) ? 1 : n)) {
        *info = -22;
    } else if (ldc < ((1 > p) ? 1 : p)) {
        *info = -24;
    } else if (ldd < ((1 > p) ? 1 : p)) {
        *info = -26;
    } else if (ldav < 1 || (leftw && ldav < nv)) {
        *info = -28;
    } else if (ldbv < 1 || (leftw && ldbv < nv)) {
        *info = -30;
    } else if (ldcv < 1 || (leftw && ldcv < pv)) {
        *info = -32;
    } else if (lddv < 1 || (leftw && lddv < pv)) {
        *info = -34;
    } else if (ldaw < 1 || (rightw && ldaw < nw)) {
        *info = -36;
    } else if (ldbw < 1 || (rightw && ldbw < nw)) {
        *info = -38;
    } else if (ldcw < 1 || (rightw && ldcw < m)) {
        *info = -40;
    } else if (lddw < 1 || (rightw && lddw < m)) {
        *info = -42;
    } else if (tol2 > zero && !fixord && tol2 > tol1) {
        *info = -46;
    }

    if (*info != 0) {
        return;
    }

    i32 minval = (n < m) ? n : m;
    minval = (minval < p) ? minval : p;
    if (minval == 0) {
        *nr = 0;
        *ns = 0;
        return;
    }

    i32 ldwork = lw + 3 * n + 100;
    f64* dwork = (f64*)calloc((size_t)ldwork, sizeof(f64));
    if (!dwork) {
        *info = -49;
        return;
    }

    i32 liwrk1 = 0;
    if (lsame(job[0], 'F')) liwrk1 = n;
    else if (spa) liwrk1 = 2 * n;

    i32 liwrk2 = 0;
    if (leftw && nv > 0) {
        liwrk2 = nv + ((p > pv) ? p : pv);
    }

    i32 liwrk3 = 0;
    if (rightw && nw > 0) {
        liwrk3 = nw + ((m > mw) ? m : mw);
    }

    i32 liwork = 3;
    if (liwrk1 > liwork) liwork = liwrk1;
    if (liwrk2 > liwork) liwork = liwrk2;
    if (liwrk3 > liwork) liwork = liwrk3;

    i32* iwork = (i32*)calloc((size_t)liwork, sizeof(i32));
    if (!iwork) {
        free(dwork);
        *info = -47;
        return;
    }

    if (scale) {
        maxred = c100;
        i32 info_tb01id;
        tb01id("All", n, m, p, &maxred, a, lda, b, ldb, c, ldc, dwork, &info_tb01id);
    }

    alpwrk = alpha;
    if (discr) {
        if (alpha == one) alpwrk = one - sqrt(SLC_DLAMCH("E"));
    } else {
        if (alpha == zero) alpwrk = -sqrt(SLC_DLAMCH("E"));
    }

    ku = 0;
    kl = ku + nn;
    ki = kl + n;
    kw = ki + n;

    i32 info_tb01kd;
    tb01kd(dico, "Unstable", "General", n, m, p, alpwrk, a, lda, b, ldb, c, ldc,
           &nu, &dwork[ku], n, &dwork[kl], &dwork[ki], &dwork[kw], ldwork - kw, &info_tb01kd);

    if (info_tb01kd != 0) {
        if (info_tb01kd != 3) {
            *info = 1;
        } else {
            *info = 2;
        }
        free(dwork);
        free(iwork);
        return;
    }

    wrkopt = (i32)dwork[kw] + kw;

    iwarnl = 0;
    *ns = n - nu;
    if (fixord) {
        nra = (*nr - nu > 0) ? *nr - nu : 0;
        if (*nr < nu) iwarnl = 3;
    } else {
        nra = 0;
    }

    if (*ns == 0) {
        *nr = nu;
        iwork[0] = 0;
        iwork[1] = nv;
        iwork[2] = nw;
        free(dwork);
        free(iwork);
        return;
    }

    nvr = nv;
    if (leftw && nv > 0) {
        i32 info_tb01pd;
        if (pv == p) {
            kw = 0;
            tb01pd("Minimal", "Scale", nv, p, pv, av, ldav, bv, ldbv, cv, ldcv,
                   &nvr, zero, iwork, dwork, ldwork, &info_tb01pd);
            wrkopt = ((i32)dwork[kw] + kw > wrkopt) ? (i32)dwork[kw] + kw : wrkopt;

            kbr = 0;
            kdr = kbr + pv * nvr;
            kw = kdr + pv * pv;

            i32 info_sb08cd;
            sb08cd(dico, nvr, p, pv, av, ldav, bv, ldbv, cv, ldcv, (f64*)dv, lddv,
                   &nnq, &nnr, &dwork[kbr], (1 > nvr) ? 1 : nvr, &dwork[kdr], pv,
                   zero, &dwork[kw], ldwork - kw, iwarn, &info_sb08cd);

            if (info_sb08cd != 0) {
                *info = info_sb08cd + 2;
                free(dwork);
                free(iwork);
                return;
            }
            nvr = nnq;
            wrkopt = ((i32)dwork[kw] + kw > wrkopt) ? (i32)dwork[kw] + kw : wrkopt;
            if (*iwarn > 0) *iwarn = 10 + *iwarn;
        } else {
            ldw = (p > pv) ? p : pv;
            kbv = 0;
            kcv = kbv + nv * ldw;
            kw = kcv + nv * ldw;

            SLC_DLACPY("Full", &nv, &p, bv, &ldbv, &dwork[kbv], &nv);
            SLC_DLACPY("Full", &pv, &nv, cv, &ldcv, &dwork[kcv], &ldw);

            tb01pd("Minimal", "Scale", nv, p, pv, av, ldav, &dwork[kbv], nv,
                   &dwork[kcv], ldw, &nvr, zero, iwork, &dwork[kw], ldwork - kw, &info_tb01pd);

            kdv = kw;
            kbr = kdv + ldw * ldw;
            kdr = kbr + pv * nvr;
            kw = kdr + pv * pv;

            SLC_DLACPY("Full", &pv, &p, dv, &lddv, &dwork[kdv], &ldw);

            i32 info_sb08cd;
            i32 nvr_ld = (1 > nvr) ? 1 : nvr;
            sb08cd(dico, nvr, p, pv, av, ldav, &dwork[kbv], nv, &dwork[kcv], ldw,
                   &dwork[kdv], ldw, &nnq, &nnr, &dwork[kbr], nvr_ld, &dwork[kdr], pv,
                   zero, &dwork[kw], ldwork - kw, iwarn, &info_sb08cd);

            if (info_sb08cd != 0) {
                *info = info_sb08cd + 2;
                free(dwork);
                free(iwork);
                return;
            }

            SLC_DLACPY("Full", &nvr, &p, &dwork[kbv], &nv, bv, &ldbv);
            SLC_DLACPY("Full", &pv, &nvr, &dwork[kcv], &ldw, cv, &ldcv);

            nvr = nnq;
            wrkopt = ((i32)dwork[kw] + kw > wrkopt) ? (i32)dwork[kw] + kw : wrkopt;
            if (*iwarn > 0) *iwarn = 10 + *iwarn;
        }
    }

    nwr = nw;
    if (rightw && nw > 0) {
        i32 info_tb01pd;
        if (m == mw) {
            kw = 0;
            tb01pd("Minimal", "Scale", nw, mw, m, aw, ldaw, bw, ldbw, cw, ldcw,
                   &nwr, zero, iwork, dwork, ldwork, &info_tb01pd);
            wrkopt = ((i32)dwork[kw] + kw > wrkopt) ? (i32)dwork[kw] + kw : wrkopt;
        } else {
            ldw = (m > mw) ? m : mw;
            kbw = 0;
            kcw = kbw + nw * ldw;
            kw = kcw + nw * ldw;

            SLC_DLACPY("Full", &nw, &mw, bw, &ldbw, &dwork[kbw], &nw);
            SLC_DLACPY("Full", &m, &nw, cw, &ldcw, &dwork[kcw], &ldw);

            tb01pd("Minimal", "Scale", nw, mw, m, aw, ldaw, &dwork[kbw], nw,
                   &dwork[kcw], ldw, &nwr, zero, iwork, &dwork[kw], ldwork - kw, &info_tb01pd);

            SLC_DLACPY("Full", &nwr, &mw, &dwork[kbw], &nw, bw, &ldbw);
            SLC_DLACPY("Full", &m, &nwr, &dwork[kcw], &ldw, cw, &ldcw);
            wrkopt = ((i32)dwork[kw] + kw > wrkopt) ? (i32)dwork[kw] + kw : wrkopt;
        }
    }

    if (rightw && nwr > 0) {
        i32 info_sb08dd;
        ldw = (1 > mw) ? 1 : mw;
        kcr = 0;
        kdr = kcr + nwr * ldw;
        kw = kdr + mw * ldw;

        sb08dd(dico, nwr, mw, m, aw, ldaw, bw, ldbw, cw, ldcw, (f64*)dw, lddw,
               &nnq, &nnr, &dwork[kcr], ldw, &dwork[kdr], ldw,
               zero, &dwork[kw], ldwork - kw, iwarn, &info_sb08dd);

        if (info_sb08dd != 0) {
            *info = info_sb08dd + 5;
            free(dwork);
            free(iwork);
            return;
        }
        nwr = nnq;
        wrkopt = ((i32)dwork[kw] + kw > wrkopt) ? (i32)dwork[kw] + kw : wrkopt;
        if (*iwarn > 0) *iwarn = 10 + *iwarn;
    }

    nu1 = nu;

    kt = 0;
    kti = kt + nn;
    kw = kti + nn;

    i32 info_ab09iy;
    ab09iy(dico, jobc, jobo, weight, *ns, m, p, nvr, pv, nwr, mw,
           alphac, alphao, &a[nu1 + nu1 * lda], lda, &b[nu1], ldb, &c[nu1 * ldc], ldc,
           av, ldav, bv, ldbv, cv, ldcv, dv, lddv, aw, ldaw, bw, ldbw, cw, ldcw, dw, lddw,
           &scalec, &scaleo, &dwork[kti], n, &dwork[kt], n, &dwork[kw], ldwork - kw, &info_ab09iy);

    if (info_ab09iy != 0) {
        *info = 9;
        free(dwork);
        free(iwork);
        return;
    }
    wrkopt = ((i32)dwork[kw] + kw > wrkopt) ? (i32)dwork[kw] + kw : wrkopt;

    i32 info_ab09ix, iwarn_ab09ix;
    info_ab09ix = ab09ix(dico, job, "Schur", ordsel, *ns, m, p, &nra, scalec, scaleo,
           &a[nu1 + nu1 * lda], lda, &b[nu1], ldb, &c[nu1 * ldc], ldc, d, ldd,
           &dwork[kti], n, &dwork[kt], n, &nmr, hsv, tol1, tol2, iwork,
           &dwork[kw], ldwork - kw, &iwarn_ab09ix);

    *iwarn = (iwarn_ab09ix > iwarnl) ? iwarn_ab09ix : iwarnl;
    if (info_ab09ix != 0) {
        *info = 10;
        free(dwork);
        free(iwork);
        return;
    }

    *nr = nra + nu;

    iwork[0] = nmr;
    iwork[1] = nvr;
    iwork[2] = nwr;

    free(dwork);
    free(iwork);
}
