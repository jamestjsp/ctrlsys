/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

/*
 * AB09JV - State-space representation of projection of V*G or conj(V)*G
 *
 * Constructs (A,BS,CS,DS) of the projection of V*G or conj(V)*G containing
 * the poles of G, from (A,B,C,D) and (AV-lambda*EV,BV,CV,DV).
 *
 * G is assumed stable with A in real Schur form.
 * V*G requires G and V have distinct poles.
 * conj(V)*G requires G and conj(V) have distinct poles.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

void ab09jv(
    const char* job, const char* dico, const char* jobev, const char* stbchk,
    const i32 n, const i32 m, const i32 p, const i32 nv, const i32 pv,
    const f64* a, const i32 lda,
    const f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* d, const i32 ldd,
    f64* av, const i32 ldav,
    f64* ev, const i32 ldev,
    f64* bv, const i32 ldbv,
    f64* cv, const i32 ldcv,
    const f64* dv, const i32 lddv,
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
    bool unitev = (jobev[0] == 'I' || jobev[0] == 'i');
    bool stabck = (stbchk[0] == 'C' || stbchk[0] == 'c');

    f64 alpha, dif, scale = 1.0, tolinf, work;
    i32 ierr, kq, kz, kar, kai, kb, kw, kc, kf, ke;
    i32 ldw, ldwn, lw, sdim;
    i32 bwork[1];
    char stdom;
    char evtype;

    *info = 0;

    i32 lw_ia;
    if (discr && conjs) {
        lw_ia = 2 * nv;
    } else {
        lw_ia = 0;
    }

    i32 lw1_inner = lw_ia;
    if (pv * n > lw1_inner) lw1_inner = pv * n;
    if (pv * m > lw1_inner) lw1_inner = pv * m;

    i32 lw1_a = nv * (nv + 5);
    i32 lw1_b = nv * n + lw1_inner;
    i32 lw1 = (lw1_a > lw1_b) ? lw1_a : lw1_b;
    if (lw1 < 1) lw1 = 1;

    i32 lw2_inner1 = 11 * nv + 16;
    if (p * nv > lw2_inner1) lw2_inner1 = p * nv;
    if (pv * nv > lw2_inner1) lw2_inner1 = pv * nv;

    i32 lw2_inner2 = nv * n + n * n;
    if (pv * n > lw2_inner2) lw2_inner2 = pv * n;
    if (pv * m > lw2_inner2) lw2_inner2 = pv * m;

    i32 lw2_a = 2 * nv * nv + lw2_inner1;
    i32 lw2_b = nv * n + lw2_inner2;
    i32 lw2 = (lw2_a > lw2_b) ? lw2_a : lw2_b;
    if (lw2 < 1) lw2 = 1;

    if (unitev) {
        lw = lw1;
    } else {
        lw = lw2;
    }

    ldwn = (n > 1) ? n : 1;
    ldw = (nv > 1) ? nv : 1;

    if (job[0] != 'V' && job[0] != 'v' && job[0] != 'C' && job[0] != 'c') {
        *info = -1;
    } else if (dico[0] != 'C' && dico[0] != 'c' && dico[0] != 'D' && dico[0] != 'd') {
        *info = -2;
    } else if (jobev[0] != 'G' && jobev[0] != 'g' && jobev[0] != 'I' && jobev[0] != 'i') {
        *info = -3;
    } else if (stbchk[0] != 'N' && stbchk[0] != 'n' && stbchk[0] != 'C' && stbchk[0] != 'c') {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (m < 0) {
        *info = -6;
    } else if (p < 0) {
        *info = -7;
    } else if (nv < 0) {
        *info = -8;
    } else if (pv < 0) {
        *info = -9;
    } else if (lda < ldwn) {
        *info = -11;
    } else if (ldb < ldwn) {
        *info = -13;
    } else {
        i32 ldc_min = 1;
        if (p > ldc_min) ldc_min = p;
        if (pv > ldc_min) ldc_min = pv;
        if (ldc < ldc_min) {
            *info = -15;
        }
    }
    if (*info == 0) {
        i32 ldd_min = 1;
        if (p > ldd_min) ldd_min = p;
        if (pv > ldd_min) ldd_min = pv;
        if (ldd < ldd_min) {
            *info = -17;
        }
    }
    if (*info == 0 && ldav < ldw) {
        *info = -19;
    }
    if (*info == 0) {
        if (ldev < 1 || (!unitev && ldev < nv)) {
            *info = -21;
        }
    }
    if (*info == 0 && ldbv < ldw) {
        *info = -23;
    }
    if (*info == 0) {
        if (!conjs && ldcv < ((pv > 1) ? pv : 1)) {
            *info = -25;
        } else if (conjs && ldcv < ((p > 1) ? p : 1)) {
            *info = -25;
        }
    }
    if (*info == 0) {
        if (!conjs && lddv < ((pv > 1) ? pv : 1)) {
            *info = -27;
        } else if (conjs && lddv < ((p > 1) ? p : 1)) {
            *info = -27;
        }
    }
    if (*info == 0 && ldwork < lw) {
        *info = -30;
    }

    if (*info != 0) {
        return;
    }

    if (p == 0 || pv == 0) {
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

    if (unitev) {
        if (nv > 0) {
            kw = nv * (nv + 2);
            if (conjs) {
                stdom = 'S';
                alpha = alpha + sqrt(tolinf);
                tb01wd(nv, pv, p, av, ldav, bv, ldbv, cv, ldcv,
                       dwork + 2 * nv, nv, dwork, dwork + nv,
                       dwork + kw, ldwork - kw, &ierr);
            } else {
                stdom = 'U';
                alpha = alpha - sqrt(tolinf);
                tb01wd(nv, p, pv, av, ldav, bv, ldbv, cv, ldcv,
                       dwork + 2 * nv, nv, dwork, dwork + nv,
                       dwork + kw, ldwork - kw, &ierr);
            }
            if (ierr != 0) {
                *info = 1;
                return;
            }
            if (stabck) {
                ierr = ab09jx(dico, &stdom, "S", nv, alpha, dwork,
                              dwork + nv, dwork, tolinf);
                if (ierr != 0) {
                    *info = 4;
                    return;
                }
            }
            if (work < dwork[kw] + (f64)kw) {
                work = dwork[kw] + (f64)kw;
            }
        }

        kw = nv * n;
        if (conjs) {
            {
                i32 mm = nv, nn = n, kk = p;
                SLC_DGEMM("T", "N", &mm, &nn, &kk, &mone, cv, &ldcv, c, &ldc,
                          &zero, dwork, &ldw);
            }

            if (discr) {
                char trana = 'T';
                char tranb = 'N';
                i32 isgn = -1;
                sb04py(trana, tranb, isgn, nv, n, av, ldav, a, lda,
                       dwork, ldw, &scale, dwork + kw, &ierr);
                if (ierr != 0) {
                    *info = 3;
                    return;
                }

                {
                    i32 mm = pv, nn = n, kk = p;
                    f64* work_kw = dwork + kw;
                    SLC_DGEMM("T", "N", &mm, &nn, &kk, &one, dv, &lddv, c, &ldc,
                              &zero, work_kw, &pv);
                    SLC_DLACPY("F", &mm, &nn, work_kw, &pv, c, &ldc);
                }

                {
                    i32 mm = pv, nn = m, kk = p;
                    f64* work_kw = dwork + kw;
                    SLC_DGEMM("T", "N", &mm, &nn, &kk, &one, dv, &lddv, d, &ldd,
                              &zero, work_kw, &pv);
                    SLC_DLACPY("F", &mm, &nn, work_kw, &pv, d, &ldd);
                }

                {
                    f64 scale_inv = one / scale;
                    i32 mm = pv, nn = n, kk = nv;
                    f64* work_kw = dwork + kw;
                    SLC_DGEMM("T", "N", &mm, &nn, &kk, &scale_inv, bv, &ldbv,
                              dwork, &ldw, &zero, work_kw, &pv);
                }

                {
                    i32 mm = pv, nn = n, kk = n;
                    f64* work_kw = dwork + kw;
                    SLC_DGEMM("N", "N", &mm, &nn, &kk, &one, work_kw, &pv,
                              a, &lda, &one, c, &ldc);
                }

                {
                    i32 mm = pv, nn = m, kk = n;
                    f64* work_kw = dwork + kw;
                    SLC_DGEMM("N", "N", &mm, &nn, &kk, &one, work_kw, &pv,
                              b, &ldb, &one, d, &ldd);
                }
            } else {
                if (n > 0) {
                    SLC_DTRSYL("T", "N", &int1, &nv, &n, av, &ldav, a, &lda,
                               dwork, &ldw, &scale, &ierr);
                    if (ierr != 0) {
                        *info = 3;
                        return;
                    }
                }

                {
                    i32 mm = pv, nn = n, kk = p;
                    f64* work_kw = dwork + kw;
                    SLC_DGEMM("T", "N", &mm, &nn, &kk, &one, dv, &lddv, c, &ldc,
                              &zero, work_kw, &pv);
                    SLC_DLACPY("F", &mm, &nn, work_kw, &pv, c, &ldc);
                }

                {
                    f64 scale_inv = one / scale;
                    i32 mm = pv, nn = n, kk = nv;
                    SLC_DGEMM("T", "N", &mm, &nn, &kk, &scale_inv, bv, &ldbv,
                              dwork, &ldw, &one, c, &ldc);
                }

                {
                    i32 mm = pv, nn = m, kk = p;
                    f64* work_kw = dwork + kw;
                    SLC_DGEMM("T", "N", &mm, &nn, &kk, &one, dv, &lddv, d, &ldd,
                              &zero, work_kw, &pv);
                    SLC_DLACPY("F", &mm, &nn, work_kw, &pv, d, &ldd);
                }
            }
        } else {
            {
                i32 mm = nv, nn = n, kk = p;
                SLC_DGEMM("N", "N", &mm, &nn, &kk, &mone, bv, &ldbv, c, &ldc,
                          &zero, dwork, &ldw);
            }

            if (n > 0) {
                i32 neg1 = -1;
                SLC_DTRSYL("N", "N", &neg1, &nv, &n, av, &ldav, a, &lda,
                           dwork, &ldw, &scale, &ierr);
                if (ierr != 0) {
                    *info = 3;
                    return;
                }
            }

            {
                i32 mm = pv, nn = n, kk = p;
                f64* work_kw = dwork + kw;
                SLC_DGEMM("N", "N", &mm, &nn, &kk, &one, dv, &lddv, c, &ldc,
                          &zero, work_kw, &pv);
                SLC_DLACPY("F", &mm, &nn, work_kw, &pv, c, &ldc);
            }

            {
                f64 scale_inv = one / scale;
                i32 mm = pv, nn = n, kk = nv;
                SLC_DGEMM("N", "N", &mm, &nn, &kk, &scale_inv, cv, &ldcv,
                          dwork, &ldw, &one, c, &ldc);
            }

            {
                i32 mm = pv, nn = m, kk = p;
                f64* work_kw = dwork + kw;
                SLC_DGEMM("N", "N", &mm, &nn, &kk, &one, dv, &lddv, d, &ldd,
                          &zero, work_kw, &pv);
                SLC_DLACPY("F", &mm, &nn, work_kw, &pv, d, &ldd);
            }
        }
    } else {
        if (nv > 0) {
            {
                i32 nn = nv;
                tolinf = tolinf * SLC_DLANGE("1", &nn, &nn, ev, &ldev, dwork);
            }

            kq = 0;
            kz = kq + nv * nv;
            kar = kz + nv * nv;
            kai = kar + nv;
            kb = kai + nv;
            kw = kb + nv;

            if (conjs) {
                stdom = 'S';
                alpha = alpha + sqrt(tolinf);

                for (i32 i = 0; i < nv - 1; i++) {
                    i32 len = nv - i - 1;
                    SLC_DSWAP(&len, &av[(i + 1) + i * ldav], &int1, &av[i + (i + 1) * ldav], &ldav);
                    SLC_DSWAP(&len, &ev[(i + 1) + i * ldev], &int1, &ev[i + (i + 1) * ldev], &ldev);
                }

                if (discr) {
                    evtype = 'R';
                    SLC_DGGES("V", "V", "N", delctg, &nv, ev, &ldev, av, &ldav,
                              &sdim, dwork + kar, dwork + kai, dwork + kb,
                              dwork + kq, &ldw, dwork + kz, &ldw,
                              dwork + kw, &(i32){ldwork - kw}, bwork, &ierr);
                } else {
                    evtype = 'G';
                    SLC_DGGES("V", "V", "N", delctg, &nv, av, &ldav, ev, &ldev,
                              &sdim, dwork + kar, dwork + kai, dwork + kb,
                              dwork + kq, &ldw, dwork + kz, &ldw,
                              dwork + kw, &(i32){ldwork - kw}, bwork, &ierr);
                }
                if (ierr != 0) {
                    *info = 1;
                    return;
                }
                if (stabck) {
                    ierr = ab09jx(dico, &stdom, &evtype, nv, alpha,
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
                    i32 mm = nv, nn = pv;
                    SLC_DLACPY("F", &mm, &nn, bv, &ldbv, dwork + kw, &ldw);
                    SLC_DGEMM("T", "N", &mm, &nn, &mm, &one, dwork + kz, &ldw,
                              dwork + kw, &ldw, &zero, bv, &ldbv);
                }
                {
                    i32 mm = p, nn = nv;
                    SLC_DLACPY("F", &mm, &nn, cv, &ldcv, dwork + kw, &p);
                    SLC_DGEMM("N", "N", &mm, &nn, &nn, &one, dwork + kw, &p,
                              dwork + kq, &ldw, &zero, cv, &ldcv);
                }
            } else {
                stdom = 'U';
                evtype = 'G';
                alpha = alpha - sqrt(tolinf);
                SLC_DGGES("V", "V", "N", delctg, &nv, av, &ldav, ev, &ldev,
                          &sdim, dwork + kar, dwork + kai, dwork + kb,
                          dwork + kq, &ldw, dwork + kz, &ldw,
                          dwork + kw, &(i32){ldwork - kw}, bwork, &ierr);
                if (ierr != 0) {
                    *info = 1;
                    return;
                }
                if (stabck) {
                    ierr = ab09jx(dico, &stdom, &evtype, nv, alpha,
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
                    i32 mm = nv, nn = p;
                    SLC_DLACPY("F", &mm, &nn, bv, &ldbv, dwork + kw, &ldw);
                    SLC_DGEMM("T", "N", &mm, &nn, &mm, &one, dwork + kq, &ldw,
                              dwork + kw, &ldw, &zero, bv, &ldbv);
                }
                {
                    i32 mm = pv, nn = nv;
                    SLC_DLACPY("F", &mm, &nn, cv, &ldcv, dwork + kw, &pv);
                    SLC_DGEMM("N", "N", &mm, &nn, &nn, &one, dwork + kw, &pv,
                              dwork + kz, &ldw, &zero, cv, &ldcv);
                }
            }
            {
                i32 maxpv = (p > pv) ? p : pv;
                f64 temp = (f64)(2 * nv * nv + nv * maxpv);
                if (work < temp) work = temp;
            }
        }

        kc = 0;
        kf = kc + nv * n;
        ke = kf + nv * n;
        kw = ke + n * n;

        {
            i32 mm = nv, nn = n;
            SLC_DLASET("F", &mm, &nn, &zero, &zero, dwork + kf, &ldw);
        }

        if (conjs) {
            {
                i32 mm = nv, nn = n, kk = p;
                SLC_DGEMM("T", "N", &mm, &nn, &kk, &one, cv, &ldcv, c, &ldc,
                          &zero, dwork + kc, &ldw);
            }

            if (discr) {
                if (n > 0) {
                    SLC_DLASET("F", &n, &n, &zero, &one, dwork + ke, &ldwn);
                    SLC_DTGSYL("N", &(i32){0}, &nv, &n, ev, &ldev, a, &lda,
                               dwork + kc, &ldw, av, &ldav, dwork + ke, &ldwn,
                               dwork + kf, &ldw, &scale, &dif,
                               dwork + kw, &(i32){ldwork - kw}, iwork, &ierr);
                    if (ierr != 0) {
                        *info = 2;
                        return;
                    }
                }

                kw = kf;
                {
                    i32 mm = pv, nn = n, kk = p;
                    SLC_DGEMM("T", "N", &mm, &nn, &kk, &one, dv, &lddv, c, &ldc,
                              &zero, dwork + kw, &pv);
                    SLC_DLACPY("F", &mm, &nn, dwork + kw, &pv, c, &ldc);
                }
                {
                    i32 mm = pv, nn = m, kk = p;
                    SLC_DGEMM("T", "N", &mm, &nn, &kk, &one, dv, &lddv, d, &ldd,
                              &zero, dwork + kw, &pv);
                    SLC_DLACPY("F", &mm, &nn, dwork + kw, &pv, d, &ldd);
                }
                {
                    f64 scale_inv = one / scale;
                    i32 mm = pv, nn = n, kk = nv;
                    SLC_DGEMM("T", "N", &mm, &nn, &kk, &scale_inv, bv, &ldbv,
                              dwork + kc, &ldw, &zero, dwork + kw, &pv);
                }
                {
                    i32 mm = pv, nn = n, kk = n;
                    SLC_DGEMM("N", "N", &mm, &nn, &kk, &one, dwork + kw, &pv,
                              a, &lda, &one, c, &ldc);
                }
                {
                    i32 mm = pv, nn = m, kk = n;
                    SLC_DGEMM("N", "N", &mm, &nn, &kk, &one, dwork + kw, &pv,
                              b, &ldb, &one, d, &ldd);
                }
            } else {
                if (n > 0) {
                    SLC_DLASET("F", &n, &n, &zero, &mone, dwork + ke, &ldwn);
                    SLC_DTGSYL("N", &(i32){0}, &nv, &n, av, &ldav, a, &lda,
                               dwork + kc, &ldw, ev, &ldev, dwork + ke, &ldwn,
                               dwork + kf, &ldw, &scale, &dif,
                               dwork + kw, &(i32){ldwork - kw}, iwork, &ierr);
                    if (ierr != 0) {
                        *info = 2;
                        return;
                    }
                }

                kw = kf;
                {
                    i32 mm = pv, nn = n, kk = p;
                    SLC_DGEMM("T", "N", &mm, &nn, &kk, &one, dv, &lddv, c, &ldc,
                              &zero, dwork + kw, &pv);
                    SLC_DLACPY("F", &mm, &nn, dwork + kw, &pv, c, &ldc);
                }
                {
                    f64 neg_scale_inv = -one / scale;
                    i32 mm = pv, nn = n, kk = nv;
                    SLC_DGEMM("T", "N", &mm, &nn, &kk, &neg_scale_inv, bv, &ldbv,
                              dwork + kc, &ldw, &one, c, &ldc);
                }
                {
                    i32 mm = pv, nn = m, kk = p;
                    SLC_DGEMM("T", "N", &mm, &nn, &kk, &one, dv, &lddv, d, &ldd,
                              &zero, dwork + kw, &pv);
                    SLC_DLACPY("F", &mm, &nn, dwork + kw, &pv, d, &ldd);
                }
            }
        } else {
            {
                i32 mm = nv, nn = n, kk = p;
                SLC_DGEMM("N", "N", &mm, &nn, &kk, &mone, bv, &ldbv, c, &ldc,
                          &zero, dwork, &ldw);
            }

            if (n > 0) {
                SLC_DLASET("F", &n, &n, &zero, &one, dwork + ke, &ldwn);
                SLC_DTGSYL("N", &(i32){0}, &nv, &n, av, &ldav, a, &lda,
                           dwork + kc, &ldw, ev, &ldev, dwork + ke, &ldwn,
                           dwork + kf, &ldw, &scale, &dif,
                           dwork + kw, &(i32){ldwork - kw}, iwork, &ierr);
                if (ierr != 0) {
                    *info = 2;
                    return;
                }
            }

            kw = kf;
            {
                i32 mm = pv, nn = n, kk = p;
                SLC_DGEMM("N", "N", &mm, &nn, &kk, &one, dv, &lddv, c, &ldc,
                          &zero, dwork + kw, &pv);
                SLC_DLACPY("F", &mm, &nn, dwork + kw, &pv, c, &ldc);
            }
            {
                f64 scale_inv = one / scale;
                i32 mm = pv, nn = n, kk = nv;
                SLC_DGEMM("N", "N", &mm, &nn, &kk, &scale_inv, cv, &ldcv,
                          dwork, &ldw, &one, c, &ldc);
            }
            {
                i32 mm = pv, nn = m, kk = p;
                SLC_DGEMM("N", "N", &mm, &nn, &kk, &one, dv, &lddv, d, &ldd,
                          &zero, dwork + kw, &pv);
                SLC_DLACPY("F", &mm, &nn, dwork + kw, &pv, d, &ldd);
            }
        }
    }

    if (work < (f64)lw) work = (f64)lw;
    dwork[0] = work;
}
