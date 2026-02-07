/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

/*
 * AB09JW - State-space representation of projection of G*W or G*conj(W)
 *
 * Constructs (A,BS,CS,DS) of the projection of G*W or G*conj(W) containing
 * the poles of G, from (A,B,C,D) and (AW-lambda*EW,BW,CW,DW).
 *
 * G is assumed stable with A in real Schur form.
 * G*W requires G and W have distinct poles.
 * G*conj(W) requires G and conj(W) have distinct poles.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

void ab09jw(
    const char* job, const char* dico, const char* jobew, const char* stbchk,
    const i32 n, const i32 m, const i32 p, const i32 nw, const i32 mw,
    const f64* a, const i32 lda,
    f64* b, const i32 ldb,
    const f64* c, const i32 ldc,
    f64* d, const i32 ldd,
    f64* aw, const i32 ldaw,
    f64* ew, const i32 ldew,
    f64* bw, const i32 ldbw,
    f64* cw, const i32 ldcw,
    const f64* dw, const i32 lddw,
    i32* iwork,
    f64* dwork, const i32 ldwork,
    i32* info
)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 mone = -1.0;
    const i32 int1 = 1;

    bool conjs = (job[0] == 'C' || job[0] == 'c');
    bool discr = (dico[0] == 'D' || dico[0] == 'd');
    bool unitew = (jobew[0] == 'I' || jobew[0] == 'i');
    bool stabck = (stbchk[0] == 'C' || stbchk[0] == 'c');

    f64 alpha, dif, scale = 1.0, tolinf, work;
    i32 ierr, kq, kz, kar, kai, kb, kw, kc, kf, ke;
    i32 ldw, ldwn, ldwm, ldwp, lw, sdim;
    i32 bwork[1];
    char stdom;
    char evtype;

    *info = 0;

    i32 lw_ia;
    if (discr && conjs) {
        lw_ia = 2 * nw;
    } else {
        lw_ia = 0;
    }

    i32 lw1_inner = lw_ia;
    if (n * mw > lw1_inner) lw1_inner = n * mw;
    if (p * mw > lw1_inner) lw1_inner = p * mw;

    i32 lw1_a = nw * (nw + 5);
    i32 lw1_b = nw * n + lw1_inner;
    i32 lw1 = (lw1_a > lw1_b) ? lw1_a : lw1_b;
    if (lw1 < 1) lw1 = 1;

    i32 lw2_inner1 = 11 * nw + 16;
    if (nw * m > lw2_inner1) lw2_inner1 = nw * m;
    if (mw * nw > lw2_inner1) lw2_inner1 = mw * nw;

    i32 lw2_inner2 = nw * n + n * n;
    if (mw * n > lw2_inner2) lw2_inner2 = mw * n;
    if (p * mw > lw2_inner2) lw2_inner2 = p * mw;

    i32 lw2_a = 2 * nw * nw + lw2_inner1;
    i32 lw2_b = nw * n + lw2_inner2;
    i32 lw2 = (lw2_a > lw2_b) ? lw2_a : lw2_b;
    if (lw2 < 1) lw2 = 1;

    if (unitew) {
        lw = lw1;
    } else {
        lw = lw2;
    }

    ldw = (nw > 1) ? nw : 1;
    ldwm = (mw > 1) ? mw : 1;
    ldwn = (n > 1) ? n : 1;
    ldwp = (p > 1) ? p : 1;

    if (job[0] != 'W' && job[0] != 'w' && job[0] != 'C' && job[0] != 'c') {
        *info = -1;
    } else if (dico[0] != 'C' && dico[0] != 'c' && dico[0] != 'D' && dico[0] != 'd') {
        *info = -2;
    } else if (jobew[0] != 'G' && jobew[0] != 'g' && jobew[0] != 'I' && jobew[0] != 'i') {
        *info = -3;
    } else if (stbchk[0] != 'N' && stbchk[0] != 'n' && stbchk[0] != 'C' && stbchk[0] != 'c') {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (m < 0) {
        *info = -6;
    } else if (p < 0) {
        *info = -7;
    } else if (nw < 0) {
        *info = -8;
    } else if (mw < 0) {
        *info = -9;
    } else if (lda < ldwn) {
        *info = -11;
    } else if (ldb < ldwn) {
        *info = -13;
    } else if (ldc < ldwp) {
        *info = -15;
    } else if (ldd < ldwp) {
        *info = -17;
    } else if (ldaw < ldw) {
        *info = -19;
    } else if (ldew < 1 || (!unitew && ldew < nw)) {
        *info = -21;
    } else if (ldbw < ldw) {
        *info = -23;
    } else if (!conjs && ldcw < ((m > 1) ? m : 1)) {
        *info = -25;
    } else if (conjs && ldcw < ldwm) {
        *info = -25;
    } else if (!conjs && lddw < ((m > 1) ? m : 1)) {
        *info = -27;
    } else if (conjs && lddw < ldwm) {
        *info = -27;
    } else if (ldwork < lw) {
        *info = -30;
    }

    if (*info != 0) {
        return;
    }

    if (m == 0) {
        SLC_DLASET("F", &n, &mw, &zero, &zero, b, &ldb);
        SLC_DLASET("F", &p, &mw, &zero, &zero, d, &ldd);
        dwork[0] = one;
        return;
    }

    if (discr) {
        alpha = one;
    } else {
        alpha = zero;
    }

    work = one;
    tolinf = SLC_DLAMCH("P");

    if (unitew) {
        if (nw > 0) {
            kw = nw * (nw + 2);
            if (conjs) {
                stdom = 'S';
                alpha = alpha + sqrt(tolinf);
                tb01wd(nw, m, mw, aw, ldaw, bw, ldbw, cw, ldcw,
                       dwork + 2 * nw, nw, dwork, dwork + nw,
                       dwork + kw, ldwork - kw, &ierr);
            } else {
                stdom = 'U';
                alpha = alpha - sqrt(tolinf);
                tb01wd(nw, mw, m, aw, ldaw, bw, ldbw, cw, ldcw,
                       dwork + 2 * nw, nw, dwork, dwork + nw,
                       dwork + kw, ldwork - kw, &ierr);
            }
            if (ierr != 0) {
                *info = 1;
                return;
            }
            if (stabck) {
                ierr = ab09jx(dico, &stdom, "S", nw, alpha, dwork,
                              dwork + nw, dwork, tolinf);
                if (ierr != 0) {
                    *info = 4;
                    return;
                }
            }
            if (work < dwork[kw] + (f64)kw) {
                work = dwork[kw] + (f64)kw;
            }
        }

        kw = nw * n;
        if (conjs) {
            {
                i32 mm = nw, nn = n, kk = m;
                SLC_DGEMM("N", "T", &mm, &nn, &kk, &mone, bw, &ldbw, b, &ldb,
                          &zero, dwork, &ldw);
            }

            if (discr) {
                char trana = 'N';
                char tranb = 'T';
                i32 isgn = -1;
                sb04py(trana, tranb, isgn, nw, n, aw, ldaw, a, lda,
                       dwork, ldw, &scale, dwork + kw, &ierr);
                if (ierr != 0) {
                    *info = 3;
                    return;
                }

                {
                    i32 mm = n, nn = mw, kk = m;
                    f64* work_kw = dwork + kw;
                    SLC_DGEMM("N", "T", &mm, &nn, &kk, &one, b, &ldb, dw, &lddw,
                              &zero, work_kw, &ldwn);
                    SLC_DLACPY("F", &mm, &nn, work_kw, &ldwn, b, &ldb);
                }

                {
                    i32 mm = p, nn = mw, kk = m;
                    f64* work_kw = dwork + kw;
                    SLC_DGEMM("N", "T", &mm, &nn, &kk, &one, d, &ldd, dw, &lddw,
                              &zero, work_kw, &ldwp);
                    SLC_DLACPY("F", &mm, &nn, work_kw, &ldwp, d, &ldd);
                }

                {
                    f64 scale_inv = one / scale;
                    i32 mm = n, nn = mw, kk = nw;
                    f64* work_kw = dwork + kw;
                    SLC_DGEMM("T", "T", &mm, &nn, &kk, &scale_inv, dwork, &ldw,
                              cw, &ldcw, &zero, work_kw, &ldwn);
                }

                {
                    i32 mm = n, nn = mw, kk = n;
                    f64* work_kw = dwork + kw;
                    SLC_DGEMM("N", "N", &mm, &nn, &kk, &one, a, &lda,
                              work_kw, &ldwn, &one, b, &ldb);
                }

                {
                    i32 mm = p, nn = mw, kk = n;
                    f64* work_kw = dwork + kw;
                    SLC_DGEMM("N", "N", &mm, &nn, &kk, &one, c, &ldc,
                              work_kw, &ldwn, &one, d, &ldd);
                }
            } else {
                if (n > 0) {
                    SLC_DTRSYL("N", "T", &int1, &nw, &n, aw, &ldaw, a, &lda,
                               dwork, &ldw, &scale, &ierr);
                    if (ierr != 0) {
                        *info = 3;
                        return;
                    }
                }

                {
                    i32 mm = n, nn = mw, kk = m;
                    f64* work_kw = dwork + kw;
                    SLC_DGEMM("N", "T", &mm, &nn, &kk, &one, b, &ldb, dw, &lddw,
                              &zero, work_kw, &ldwn);
                    SLC_DLACPY("F", &mm, &nn, work_kw, &ldwn, b, &ldb);
                }

                {
                    f64 scale_inv = one / scale;
                    i32 mm = n, nn = mw, kk = nw;
                    SLC_DGEMM("T", "T", &mm, &nn, &kk, &scale_inv, dwork, &ldw,
                              cw, &ldcw, &one, b, &ldb);
                }

                {
                    i32 mm = p, nn = mw, kk = m;
                    f64* work_kw = dwork + kw;
                    SLC_DGEMM("N", "T", &mm, &nn, &kk, &one, d, &ldd, dw, &lddw,
                              &zero, work_kw, &ldwp);
                    SLC_DLACPY("F", &mm, &nn, work_kw, &ldwp, d, &ldd);
                }
            }
        } else {
            {
                i32 mm = n, nn = nw, kk = m;
                SLC_DGEMM("N", "N", &mm, &nn, &kk, &one, b, &ldb, cw, &ldcw,
                          &zero, dwork, &ldwn);
            }

            if (n > 0) {
                i32 neg1 = -1;
                SLC_DTRSYL("N", "N", &neg1, &n, &nw, a, &lda, aw, &ldaw,
                           dwork, &ldwn, &scale, &ierr);
                if (ierr != 0) {
                    *info = 3;
                    return;
                }
            }

            {
                i32 mm = n, nn = mw, kk = m;
                f64* work_kw = dwork + kw;
                SLC_DGEMM("N", "N", &mm, &nn, &kk, &one, b, &ldb, dw, &lddw,
                          &zero, work_kw, &ldwn);
                SLC_DLACPY("F", &mm, &nn, work_kw, &ldwn, b, &ldb);
            }

            {
                f64 scale_inv = one / scale;
                i32 mm = n, nn = mw, kk = nw;
                SLC_DGEMM("N", "N", &mm, &nn, &kk, &scale_inv, dwork, &ldwn,
                          bw, &ldbw, &one, b, &ldb);
            }

            {
                i32 mm = p, nn = mw, kk = m;
                f64* work_kw = dwork + kw;
                SLC_DGEMM("N", "N", &mm, &nn, &kk, &one, d, &ldd, dw, &lddw,
                          &zero, work_kw, &ldwp);
                SLC_DLACPY("F", &mm, &nn, work_kw, &ldwp, d, &ldd);
            }
        }
    } else {
        if (nw > 0) {
            {
                i32 nn = nw;
                tolinf = tolinf * SLC_DLANGE("1", &nn, &nn, ew, &ldew, dwork);
            }

            kq = 0;
            kz = kq + nw * nw;
            kar = kz + nw * nw;
            kai = kar + nw;
            kb = kai + nw;
            kw = kb + nw;

            if (conjs) {
                stdom = 'S';
                alpha = alpha + sqrt(tolinf);

                for (i32 i = 0; i < nw - 1; i++) {
                    i32 len = nw - i - 1;
                    SLC_DSWAP(&len, &aw[(i + 1) + i * ldaw], &int1, &aw[i + (i + 1) * ldaw], &ldaw);
                    SLC_DSWAP(&len, &ew[(i + 1) + i * ldew], &int1, &ew[i + (i + 1) * ldew], &ldew);
                }

                if (discr) {
                    evtype = 'R';
                    SLC_DGGES("V", "V", "N", delctg, &nw, ew, &ldew, aw, &ldaw,
                              &sdim, dwork + kar, dwork + kai, dwork + kb,
                              dwork + kq, &ldw, dwork + kz, &ldw,
                              dwork + kw, &(i32){ldwork - kw}, bwork, &ierr);
                } else {
                    evtype = 'G';
                    SLC_DGGES("V", "V", "N", delctg, &nw, aw, &ldaw, ew, &ldew,
                              &sdim, dwork + kar, dwork + kai, dwork + kb,
                              dwork + kq, &ldw, dwork + kz, &ldw,
                              dwork + kw, &(i32){ldwork - kw}, bwork, &ierr);
                }
                if (ierr != 0) {
                    *info = 1;
                    return;
                }
                if (stabck) {
                    ierr = ab09jx(dico, &stdom, &evtype, nw, alpha,
                                  dwork + kar, dwork + kai, dwork + kb, tolinf);
                    if (ierr != 0) {
                        *info = 4;
                        return;
                    }
                }
                if (work < dwork[kw] + (f64)kw) {
                    work = dwork[kw] + (f64)kw;
                }

                kw = kar;
                {
                    i32 mm = nw, nn = m;
                    SLC_DLACPY("F", &mm, &nn, bw, &ldbw, dwork + kw, &ldw);
                    SLC_DGEMM("T", "N", &mm, &nn, &mm, &one, dwork + kz, &ldw,
                              dwork + kw, &ldw, &zero, bw, &ldbw);
                }
                {
                    i32 mm = mw, nn = nw;
                    SLC_DLACPY("F", &mm, &nn, cw, &ldcw, dwork + kw, &ldwm);
                    SLC_DGEMM("N", "N", &mm, &nn, &nn, &one, dwork + kw, &ldwm,
                              dwork + kq, &ldw, &zero, cw, &ldcw);
                }
            } else {
                stdom = 'U';
                evtype = 'G';
                alpha = alpha - sqrt(tolinf);
                SLC_DGGES("V", "V", "N", delctg, &nw, aw, &ldaw, ew, &ldew,
                          &sdim, dwork + kar, dwork + kai, dwork + kb,
                          dwork + kq, &ldw, dwork + kz, &ldw,
                          dwork + kw, &(i32){ldwork - kw}, bwork, &ierr);
                if (ierr != 0) {
                    *info = 1;
                    return;
                }
                if (stabck) {
                    ierr = ab09jx(dico, &stdom, &evtype, nw, alpha,
                                  dwork + kar, dwork + kai, dwork + kb, tolinf);
                    if (ierr != 0) {
                        *info = 4;
                        return;
                    }
                }
                if (work < dwork[kw] + (f64)kw) {
                    work = dwork[kw] + (f64)kw;
                }

                kw = kar;
                {
                    i32 mm = nw, nn = mw;
                    SLC_DLACPY("F", &mm, &nn, bw, &ldbw, dwork + kw, &ldw);
                    SLC_DGEMM("T", "N", &mm, &nn, &mm, &one, dwork + kq, &ldw,
                              dwork + kw, &ldw, &zero, bw, &ldbw);
                }
                {
                    i32 mm = m, nn = nw;
                    SLC_DLACPY("F", &mm, &nn, cw, &ldcw, dwork + kw, &m);
                    SLC_DGEMM("N", "N", &mm, &nn, &nn, &one, dwork + kw, &m,
                              dwork + kz, &ldw, &zero, cw, &ldcw);
                }
            }
            {
                i32 maxmw = (m > mw) ? m : mw;
                f64 temp = (f64)(2 * nw * nw + nw * maxmw);
                if (work < temp) work = temp;
            }
        }

        kc = 0;
        kf = kc + nw * n;
        ke = kf + nw * n;
        kw = ke + n * n;

        {
            i32 mm = n, nn = nw;
            SLC_DLASET("F", &mm, &nn, &zero, &zero, dwork + kf, &ldwn);
        }

        if (conjs) {
            {
                i32 mm = n, nn = nw, kk = m;
                SLC_DGEMM("N", "T", &mm, &nn, &kk, &one, b, &ldb, bw, &ldbw,
                          &zero, dwork + kc, &ldwn);
            }

            if (discr) {
                if (n > 0) {
                    SLC_DLASET("F", &n, &n, &zero, &one, dwork + ke, &ldwn);
                    SLC_DTGSYL("N", &(i32){0}, &n, &nw, a, &lda, ew, &ldew,
                               dwork + kc, &ldwn, dwork + ke, &ldwn, aw,
                               &ldaw, dwork + kf, &ldwn, &scale, &dif,
                               dwork + kw, &(i32){ldwork - kw}, iwork, &ierr);
                    if (ierr != 0) {
                        *info = 2;
                        return;
                    }
                }

                kw = kf;
                {
                    i32 mm = n, nn = mw, kk = m;
                    SLC_DGEMM("N", "T", &mm, &nn, &kk, &one, b, &ldb, dw, &lddw,
                              &zero, dwork + kw, &ldwn);
                    SLC_DLACPY("F", &mm, &nn, dwork + kw, &ldwn, b, &ldb);
                }
                {
                    i32 mm = p, nn = mw, kk = m;
                    SLC_DGEMM("N", "T", &mm, &nn, &kk, &one, d, &ldd, dw, &lddw,
                              &zero, dwork + kw, &ldwp);
                    SLC_DLACPY("F", &mm, &nn, dwork + kw, &ldwp, d, &ldd);
                }
                {
                    f64 neg_scale_inv = -one / scale;
                    i32 mm = n, nn = mw, kk = nw;
                    SLC_DGEMM("N", "T", &mm, &nn, &kk, &neg_scale_inv,
                              dwork + kc, &ldwn, cw, &ldcw, &zero,
                              dwork + kw, &ldwn);
                }
                {
                    i32 mm = n, nn = mw, kk = n;
                    SLC_DGEMM("N", "N", &mm, &nn, &kk, &one, a, &lda,
                              dwork + kw, &ldwn, &one, b, &ldb);
                }
                {
                    i32 mm = p, nn = mw, kk = n;
                    SLC_DGEMM("N", "N", &mm, &nn, &kk, &one, c, &ldc,
                              dwork + kw, &ldwn, &one, d, &ldd);
                }
            } else {
                if (n > 0) {
                    SLC_DLASET("F", &n, &n, &zero, &mone, dwork + ke, &ldwn);
                    SLC_DTGSYL("N", &(i32){0}, &n, &nw, a, &lda, aw, &ldaw,
                               dwork + kc, &ldwn, dwork + ke, &ldwn, ew,
                               &ldew, dwork + kf, &ldwn, &scale, &dif,
                               dwork + kw, &(i32){ldwork - kw}, iwork, &ierr);
                    if (ierr != 0) {
                        *info = 2;
                        return;
                    }
                }

                kw = kf;
                {
                    i32 mm = n, nn = mw, kk = m;
                    SLC_DGEMM("N", "T", &mm, &nn, &kk, &one, b, &ldb, dw, &lddw,
                              &zero, dwork + kw, &ldwn);
                    SLC_DLACPY("F", &mm, &nn, dwork + kw, &ldwn, b, &ldb);
                }
                {
                    f64 scale_inv = one / scale;
                    i32 mm = n, nn = mw, kk = nw;
                    SLC_DGEMM("N", "T", &mm, &nn, &kk, &scale_inv,
                              dwork + kf, &ldwn, cw, &ldcw, &one, b, &ldb);
                }
                {
                    i32 mm = p, nn = mw, kk = m;
                    SLC_DGEMM("N", "T", &mm, &nn, &kk, &one, d, &ldd, dw, &lddw,
                              &zero, dwork + kw, &ldwp);
                    SLC_DLACPY("F", &mm, &nn, dwork + kw, &ldwp, d, &ldd);
                }
            }
        } else {
            {
                i32 mm = n, nn = nw, kk = m;
                SLC_DGEMM("N", "N", &mm, &nn, &kk, &one, b, &ldb, cw, &ldcw,
                          &zero, dwork + kc, &ldwn);
            }

            if (n > 0) {
                SLC_DLASET("F", &n, &n, &zero, &one, dwork + ke, &ldwn);
                SLC_DTGSYL("N", &(i32){0}, &n, &nw, a, &lda, aw, &ldaw,
                           dwork + kc, &ldwn, dwork + ke, &ldwn, ew, &ldew,
                           dwork + kf, &ldwn, &scale, &dif, dwork + kw,
                           &(i32){ldwork - kw}, iwork, &ierr);
                if (ierr != 0) {
                    *info = 2;
                    return;
                }
            }

            kw = kf;
            {
                i32 mm = n, nn = mw, kk = m;
                SLC_DGEMM("N", "N", &mm, &nn, &kk, &one, b, &ldb, dw, &lddw,
                          &zero, dwork + kw, &ldwn);
                SLC_DLACPY("F", &mm, &nn, dwork + kw, &ldwn, b, &ldb);
            }
            {
                f64 scale_inv = one / scale;
                i32 mm = n, nn = mw, kk = nw;
                SLC_DGEMM("N", "N", &mm, &nn, &kk, &scale_inv,
                          dwork + kf, &ldwn, bw, &ldbw, &one, b, &ldb);
            }
            {
                i32 mm = p, nn = mw, kk = m;
                SLC_DGEMM("N", "N", &mm, &nn, &kk, &one, d, &ldd, dw, &lddw,
                          &zero, dwork + kw, &ldwp);
                SLC_DLACPY("F", &mm, &nn, dwork + kw, &ldwp, d, &ldd);
            }
        }
    }

    if (work < (f64)lw) work = (f64)lw;
    dwork[0] = work;
}
