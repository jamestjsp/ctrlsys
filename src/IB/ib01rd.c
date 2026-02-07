/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * IB01RD - Estimate initial state for discrete-time LTI system
 *
 * Given system matrices (A,B,C,D) and input/output trajectories, estimates
 * the initial state x(0) for discrete-time system:
 *   x(k+1) = A*x(k) + B*u(k)
 *   y(k)   = C*x(k) + D*u(k)
 *
 * Matrix A must be in real Schur form.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

#define IBLOCK 16384

void slicot_ib01rd(
    const char* job,
    i32 n,
    i32 m,
    i32 l,
    i32 nsmp,
    const f64* a, i32 lda,
    const f64* b, i32 ldb,
    const f64* c, i32 ldc,
    const f64* d, i32 ldd,
    const f64* u, i32 ldu,
    const f64* y, i32 ldy,
    f64* x0,
    f64 tol,
    i32* iwork,
    f64* dwork, i32 ldwork,
    i32* iwarn,
    i32* info)
{
    bool withd = (job[0] == 'N' || job[0] == 'n');
    bool jobz = (job[0] == 'Z' || job[0] == 'z');

    *iwarn = 0;
    *info = 0;

    i32 nn = n * n;
    i32 minsmp = n;

    if (!jobz && !withd) {
        *info = -1;
        return;
    }
    if (n < 0) {
        *info = -2;
        return;
    }
    if (m < 0) {
        *info = -3;
        return;
    }
    if (l <= 0) {
        *info = -4;
        return;
    }
    if (nsmp < minsmp) {
        *info = -5;
        return;
    }
    i32 lda_min = (n > 1) ? n : 1;
    if (lda < lda_min) {
        *info = -7;
        return;
    }
    if (ldb < 1 || (ldb < n && m > 0)) {
        *info = -9;
        return;
    }
    if (ldc < l) {
        *info = -11;
        return;
    }
    if (ldd < 1 || (withd && ldd < l && m > 0)) {
        *info = -13;
        return;
    }
    if (ldu < 1 || (m > 0 && ldu < nsmp)) {
        *info = -15;
        return;
    }
    i32 ldy_min = (nsmp > 1) ? nsmp : 1;
    if (ldy < ldy_min) {
        *info = -17;
        return;
    }
    if (tol > 1.0) {
        *info = -19;
        return;
    }

    i32 nsmpl = nsmp * l;
    i32 iq = minsmp * l;
    i32 ncp1 = n + 1;
    i32 isize = nsmpl * ncp1;
    i32 ic = 2 * nn;
    i32 minwls = minsmp * ncp1;
    i32 itau_calc = ic + l * n;
    i32 ldw1 = isize + 2 * n + ((ic > 4 * n) ? ic : 4 * n);
    i32 ldw2 = minwls + 2 * n + ((iq * ncp1 + itau_calc > 4 * n) ? (iq * ncp1 + itau_calc) : 4 * n);
    i32 minwrk = (ldw1 < ldw2) ? ldw1 : ldw2;
    if (minwrk < 2) minwrk = 2;

    if (ldwork < minwrk) {
        *info = -22;
        dwork[0] = (f64)minwrk;
        return;
    }

    if (n == 0) {
        dwork[0] = 2.0;
        dwork[1] = 1.0;
        return;
    }

    i32 maxwrk = minwrk;
    i32 iypnt = 0;
    i32 iupnt = 0;
    i32 inir = 0;
    i32 inigam, lddw, nobs, ncycle, inih;

    if (ldwork >= ldw1) {
        ncycle = 1;
        nobs = nsmp;
        lddw = nsmpl;
        inigam = 0;
    } else {
        i32 jwork = ldwork - minwls - 2 * n - itau_calc;
        lddw = jwork / ncp1;
        nobs = lddw / l;
        lddw = l * nobs;
        ncycle = nsmp / nobs;
        if (nsmp % nobs != 0) ncycle++;
        inih = inir + nn;
        inigam = inih + n;
    }

    bool ncyc = (ncycle > 1);
    i32 irhs = inigam + lddw * n;
    i32 ixinit = irhs + lddw;
    ic = ixinit + n;

    i32 ia, ldr, ie;
    if (ncyc) {
        ia = ic + l * n;
        ldr = n;
        ie = inigam;
    } else {
        inih = irhs;
        ia = ic;
        ldr = lddw;
        ie = ixinit;
    }

    i32 iutran = ia;
    i32 ias = ia + nn;
    i32 itau = ia;

    i32 int1 = 1;
    f64 dbl0 = 0.0, dbl1 = 1.0, dblm1 = -1.0;
    i32 ierr;

    bool block = (m > 1 && nsmp * m >= IBLOCK);
    i32 nrbl = 0, nc = 0, init = 0;
    if (block) {
        nrbl = (ldwork - iutran) / m;
        nc = nobs / nrbl;
        if (nobs % nrbl != 0) nc++;
        init = (nc - 1) * nrbl;
        block = (nrbl > 1);
    }

    f64* x0_work = (f64*)malloc(n * sizeof(f64));
    if (!x0_work) {
        *info = -22;
        return;
    }

    for (i32 icycle = 0; icycle < ncycle; icycle++) {
        bool first = (icycle == 0);
        if (!first && icycle == ncycle - 1) {
            nobs = nsmp - (ncycle - 1) * nobs;
            lddw = l * nobs;
            if (block) {
                nc = nobs / nrbl;
                if (nobs % nrbl != 0) nc++;
                init = (nc - 1) * nrbl;
            }
        }

        i32 jwork = ias + nn;
        i32 iexpon = (i32)(log((f64)nobs) / log(2.0));
        i32 irem = l * (nobs - (1 << iexpon));
        bool power2 = (irem == 0);
        if (!power2) iexpon++;

        if (first) {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i < l; i++) {
                    dwork[inigam + i + j * lddw] = c[i + j * ldc];
                }
            }
        } else {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i < l; i++) {
                    dwork[inigam + i + j * lddw] = dwork[ic + i + j * l];
                }
            }
        }

        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i <= j; i++) {
                dwork[ia + i + j * n] = a[i + j * lda];
            }
        }
        if (n > 1) {
            for (i32 i = 0; i < n - 1; i++) {
                dwork[ia + (i + 1) + i * n] = a[(i + 1) + i * lda];
            }
        }

        i32 i2 = l;
        i32 nrow = 0;

        for (i32 jj = 0; jj < iexpon; jj++) {
            i32 ig = inigam;
            if (jj < iexpon - 1 || power2) {
                nrow = i2;
            } else {
                nrow = irem;
            }

            SLC_DLACPY("F", &nrow, &n, &dwork[ig], &lddw, &dwork[ig + i2], &lddw);
            SLC_DTRMM("R", "U", "N", "N", &nrow, &n, &dbl1, &dwork[ia], &n,
                      &dwork[ig + i2], &lddw);

            for (i32 ix = 0; ix < n - 1; ix++) {
                SLC_DAXPY(&nrow, &dwork[ia + ix * n + ix + 1], &dwork[ig + lddw],
                          &int1, &dwork[ig + i2], &int1);
                ig += lddw;
            }

            if (jj < iexpon - 1) {
                SLC_DLACPY("U", &n, &n, &dwork[ia], &n, &dwork[ias], &n);
                if (n > 1) {
                    i32 nm1 = n - 1;
                    i32 np1 = n + 1;
                    SLC_DCOPY(&nm1, &dwork[ia + 1], &np1, &dwork[ias + 1], &np1);
                }
                mb01td(n, &dwork[ias], n, &dwork[ia], n, &dwork[jwork], &ierr);
                i2 *= 2;
            }
        }

        if (ncyc) {
            i32 ig = inigam + i2 + nrow - l;
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i < l; i++) {
                    dwork[ic + i + j * l] = dwork[ig + i + j * lddw];
                }
            }
            SLC_DTRMM("R", "U", "N", "N", &l, &n, &dbl1, a, &lda, &dwork[ic], &l);

            ig = inigam + i2 + nrow - l;
            for (i32 ix = 0; ix < n - 1; ix++) {
                f64 aval = a[(ix + 1) + ix * lda];
                SLC_DAXPY(&l, &aval, &dwork[ig + lddw], &int1, &dwork[ic + ix * l], &int1);
                ig += lddw;
            }
        }

        if (first) {
            for (i32 i = 0; i < n; i++) {
                dwork[ixinit + i] = 0.0;
            }
        }
        for (i32 i = 0; i < n; i++) {
            x0[i] = dwork[ixinit + i];
        }

        i32 iy = irhs;
        for (i32 j = 0; j < l; j++) {
            for (i32 i = 0; i < nobs; i++) {
                dwork[iy + i * l + j] = y[iypnt + i + j * ldy];
            }
        }

        iy = irhs;
        i32 iu = iupnt;

        if (m > 0) {
            if (withd) {
                if (block) {
                    bool switch_flag = true;
                    i32 nrow_blk = nrbl;
                    i32 iut = iutran;

                    for (i32 k = 0; k < nobs; k++) {
                        if (k % nrow_blk == 0 && switch_flag) {
                            iut = iutran;
                            if (k >= init) {
                                nrow_blk = nobs - init;
                                switch_flag = false;
                            }
                            ma02ad("F", nrow_blk, m, &u[iu], ldu, &dwork[iut], m);
                            iu += nrow_blk;
                        }

                        SLC_DGEMV("N", &l, &n, &dblm1, c, &ldc, x0, &int1,
                                  &dbl1, &dwork[iy], &int1);
                        SLC_DGEMV("N", &l, &m, &dblm1, d, &ldd, &dwork[iut], &int1,
                                  &dbl1, &dwork[iy], &int1);
                        SLC_DTRMV("U", "N", "N", &n, a, &lda, x0, &int1);

                        for (i32 ix = 1; ix < n; ix++) {
                            x0[ix] += a[ix + (ix - 1) * lda] * dwork[ixinit + ix - 1];
                        }

                        SLC_DGEMV("N", &n, &m, &dbl1, b, &ldb, &dwork[iut], &int1,
                                  &dbl1, x0, &int1);
                        for (i32 i = 0; i < n; i++) {
                            dwork[ixinit + i] = x0[i];
                        }
                        iy += l;
                        iut += m;
                    }
                } else {
                    for (i32 k = 0; k < nobs; k++) {
                        SLC_DGEMV("N", &l, &n, &dblm1, c, &ldc, x0, &int1,
                                  &dbl1, &dwork[iy], &int1);
                        SLC_DGEMV("N", &l, &m, &dblm1, d, &ldd, &u[iu], &ldu,
                                  &dbl1, &dwork[iy], &int1);
                        SLC_DTRMV("U", "N", "N", &n, a, &lda, x0, &int1);

                        for (i32 ix = 1; ix < n; ix++) {
                            x0[ix] += a[ix + (ix - 1) * lda] * dwork[ixinit + ix - 1];
                        }

                        SLC_DGEMV("N", &n, &m, &dbl1, b, &ldb, &u[iu], &ldu,
                                  &dbl1, x0, &int1);
                        for (i32 i = 0; i < n; i++) {
                            dwork[ixinit + i] = x0[i];
                        }
                        iy += l;
                        iu++;
                    }
                }
            } else {
                if (block) {
                    bool switch_flag = true;
                    i32 nrow_blk = nrbl;
                    i32 iut = iutran;

                    for (i32 k = 0; k < nobs; k++) {
                        if (k % nrow_blk == 0 && switch_flag) {
                            iut = iutran;
                            if (k >= init) {
                                nrow_blk = nobs - init;
                                switch_flag = false;
                            }
                            ma02ad("F", nrow_blk, m, &u[iu], ldu, &dwork[iut], m);
                            iu += nrow_blk;
                        }

                        SLC_DGEMV("N", &l, &n, &dblm1, c, &ldc, x0, &int1,
                                  &dbl1, &dwork[iy], &int1);
                        SLC_DTRMV("U", "N", "N", &n, a, &lda, x0, &int1);

                        for (i32 ix = 1; ix < n; ix++) {
                            x0[ix] += a[ix + (ix - 1) * lda] * dwork[ixinit + ix - 1];
                        }

                        SLC_DGEMV("N", &n, &m, &dbl1, b, &ldb, &dwork[iut], &int1,
                                  &dbl1, x0, &int1);
                        for (i32 i = 0; i < n; i++) {
                            dwork[ixinit + i] = x0[i];
                        }
                        iy += l;
                        iut += m;
                    }
                } else {
                    for (i32 k = 0; k < nobs; k++) {
                        SLC_DGEMV("N", &l, &n, &dblm1, c, &ldc, x0, &int1,
                                  &dbl1, &dwork[iy], &int1);
                        SLC_DTRMV("U", "N", "N", &n, a, &lda, x0, &int1);

                        for (i32 ix = 1; ix < n; ix++) {
                            x0[ix] += a[ix + (ix - 1) * lda] * dwork[ixinit + ix - 1];
                        }

                        SLC_DGEMV("N", &n, &m, &dbl1, b, &ldb, &u[iu], &ldu,
                                  &dbl1, x0, &int1);
                        for (i32 i = 0; i < n; i++) {
                            dwork[ixinit + i] = x0[i];
                        }
                        iy += l;
                        iu++;
                    }
                }
            }
        } else {
            for (i32 k = 0; k < nobs; k++) {
                SLC_DGEMV("N", &l, &n, &dblm1, c, &ldc, x0, &int1,
                          &dbl1, &dwork[iy], &int1);
                SLC_DTRMV("U", "N", "N", &n, a, &lda, x0, &int1);

                for (i32 ix = 1; ix < n; ix++) {
                    x0[ix] += a[ix + (ix - 1) * lda] * dwork[ixinit + ix - 1];
                }

                for (i32 i = 0; i < n; i++) {
                    dwork[ixinit + i] = x0[i];
                }
                iy += l;
            }
        }

        jwork = itau + n;
        if (first) {
            i32 ldw_local = ldwork - jwork;
            SLC_DGEQRF(&lddw, &n, &dwork[inigam], &lddw, &dwork[itau],
                       &dwork[jwork], &ldw_local, &ierr);

            ldw_local = ldwork - jwork;
            SLC_DORMQR("L", "T", &lddw, &int1, &n, &dwork[inigam], &lddw,
                       &dwork[itau], &dwork[irhs], &lddw, &dwork[jwork],
                       &ldw_local, &ierr);

            if (ncyc) {
                SLC_DLACPY("U", &n, &ncp1, &dwork[inigam], &lddw,
                           &dwork[inir], &ldr);
            }
        } else {
            mb04od("F", n, 1, lddw, &dwork[inir], ldr,
                   &dwork[inigam], lddw, &dwork[inih], ldr,
                   &dwork[irhs], lddw, &dwork[itau], &dwork[jwork]);
        }

        iupnt += nobs;
        iypnt += nobs;
    }

    f64 rcond;
    SLC_DTRCON("1", "U", "N", &n, &dwork[inir], &ldr, &rcond, &dwork[ie], iwork, &ierr);

    f64 toll = tol;
    if (toll <= 0.0) toll = SLC_DLAMCH("P");

    f64 thresh = pow(toll, 2.0 / 3.0);
    if (rcond <= thresh) {
        *iwarn = 4;

        if (n > 1) {
            i32 nm1 = n - 1;
            SLC_DLASET("L", &nm1, &nm1, &dbl0, &dbl0, &dwork[inir + 1], &ldr);
        }

        i32 isv = ie;
        i32 jwork = isv + n;
        i32 rank;
        i32 ldw_local = ldwork - jwork;
        SLC_DGELSS(&n, &n, &int1, &dwork[inir], &ldr, &dwork[inih], &ldr,
                   &dwork[isv], &toll, &rank, &dwork[jwork], &ldw_local, &ierr);

        if (ierr > 0) {
            *info = 2;
            free(x0_work);
            return;
        }
        i32 opt = (i32)dwork[jwork] + jwork;
        if (opt > maxwrk) maxwrk = opt;
    } else {
        SLC_DTRSV("U", "N", "N", &n, &dwork[inir], &ldr, &dwork[inih], &int1);
    }

    for (i32 i = 0; i < n; i++) {
        x0[i] = dwork[inih + i];
    }

    dwork[0] = (f64)maxwrk;
    dwork[1] = rcond;

    free(x0_work);
}
