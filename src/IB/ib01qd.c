/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * IB01QD - Estimate initial state and system matrices B, D
 *
 * Given (A, C) and input/output trajectories, estimates the system matrices
 * B and D, and optionally the initial state x(0), for a discrete-time LTI
 * system: x(k+1) = A*x(k) + B*u(k), y(k) = C*x(k) + D*u(k).
 * Matrix A is assumed to be in real Schur form.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

void slicot_ib01qd(
    const char* jobx0,
    const char* job,
    i32 n,
    i32 m,
    i32 l,
    i32 nsmp,
    const f64* a, i32 lda,
    const f64* c, i32 ldc,
    f64* u, i32 ldu,
    const f64* y, i32 ldy,
    f64* x0,
    f64* b, i32 ldb,
    f64* d, i32 ldd,
    f64 tol,
    i32* iwork,
    f64* dwork, i32 ldwork,
    i32* iwarn,
    i32* info)
{
    bool withx0 = (jobx0[0] == 'X' || jobx0[0] == 'x');
    bool withd = (job[0] == 'D' || job[0] == 'd');
    bool withb = (job[0] == 'B' || job[0] == 'b') || withd;

    *iwarn = 0;
    *info = 0;

    i32 lm = l * m;
    i32 ln = l * n;
    i32 nn = n * n;
    i32 nm = n * m;
    i32 n2m = n * nm;
    i32 ncol = nm;
    if (withx0) ncol += n;

    i32 minsmp = ncol;
    i32 iq;
    if (withd) {
        minsmp += m;
        iq = minsmp;
    } else if (!withx0) {
        iq = minsmp;
        minsmp += 1;
    } else {
        iq = minsmp;
    }

    if (!(withx0 || jobx0[0] == 'N' || jobx0[0] == 'n')) {
        *info = -1;
        return;
    }
    if (!withb) {
        *info = -2;
        return;
    }
    if (n < 0) {
        *info = -3;
        return;
    }
    if (m < 0) {
        *info = -4;
        return;
    }
    if (l <= 0) {
        *info = -5;
        return;
    }
    if (nsmp < minsmp) {
        *info = -6;
        return;
    }
    i32 lda_min = (n > 1) ? n : 1;
    if (lda < lda_min) {
        *info = -8;
        return;
    }
    if (ldc < l) {
        *info = -10;
        return;
    }
    if (ldu < 1 || (m > 0 && ldu < nsmp)) {
        *info = -12;
        return;
    }
    i32 ldy_min = (nsmp > 1) ? nsmp : 1;
    if (ldy < ldy_min) {
        *info = -14;
        return;
    }
    if (ldb < 1 || (n > 0 && m > 0 && ldb < n)) {
        *info = -17;
        return;
    }
    if (ldd < 1 || (withd && m > 0 && ldd < l)) {
        *info = -19;
        return;
    }
    if (tol > 1.0) {
        *info = -20;
        return;
    }

    i32 nsmpl = nsmp * l;
    iq = iq * l;
    i32 ncp1 = ncol + 1;
    i32 isize = nsmpl * ncp1;

    i32 ic = 0;
    if (n > 0 && withx0) {
        ic = 2 * nn + n;
    }

    i32 minwls = ncol * ncp1;
    if (withd) minwls += lm * ncp1;

    i32 ia;
    if (m > 0 && withd) {
        ia = m;
        i32 t1 = 2 * ncol;
        if (m > t1) ia = m;
        else ia = m + t1;
        if (m > 2 * ncol) ia = m + m;
        else ia = m + 2 * ncol;
    } else {
        ia = 2 * ncol;
    }

    i32 itau = n2m + (ic > ia ? ic : ia);
    if (withx0) itau += ln;

    i32 ldw2 = isize + (n + (ic > ia ? ic : ia));
    i32 t = 6 * ncol;
    if (n + (ic > ia ? ic : ia) < t) ldw2 = isize + t;

    i32 ldw3 = minwls + (iq * ncp1 + itau);
    if (iq * ncp1 + itau < 6 * ncol) ldw3 = minwls + 6 * ncol;

    if (m > 0 && withd) {
        i32 t2 = isize + 2 * m * m + 6 * m;
        if (t2 > ldw2) ldw2 = t2;
        t2 = minwls + 2 * m * m + 6 * m;
        if (t2 > ldw3) ldw3 = t2;
    }

    i32 minwrk = (ldw2 < ldw3) ? ldw2 : ldw3;
    if (minwrk < 2) minwrk = 2;
    if (m > 0 && withd && minwrk < 3) minwrk = 3;

    if (ldwork < minwrk) {
        *info = -23;
        dwork[0] = (f64)minwrk;
        return;
    }

    i32 maxwrk = minwrk;

    // Quick return if both N and M are 0
    if (n == 0 && m == 0) {
        dwork[1] = 1.0;
        if (m > 0 && withd) {
            dwork[0] = 3.0;
            dwork[2] = 1.0;
        } else {
            dwork[0] = 2.0;
        }
        return;
    }

    i32 iypnt = 0;
    i32 iupnt = 0;
    i32 lddw = (ldwork - minwls - itau) / ncp1;
    i32 nobs = nsmp;
    if (lddw / l < nsmp) nobs = lddw / l;

    i32 ncycle, inir, inis, iny, inih, lnob;
    bool ncyc;

    if (ldwork >= ldw2 || nsmp <= nobs) {
        ncycle = 1;
        nobs = nsmp;
        lddw = (nsmpl > 1) ? nsmpl : 1;
        inir = withd ? m : 0;
        iny = 0;
        inis = 0;
        inih = 0;
    } else {
        lnob = l * nobs;
        lddw = (lnob > 1) ? lnob : 1;
        ncycle = nsmp / nobs;
        if (nsmp % nobs != 0) ncycle++;
        inir = 0;
        inih = inir + ncol * ncol;
        inis = inih + ncol;
        iny = withd ? (inis + lm * ncp1) : inis;
    }

    ncyc = (ncycle > 1);
    i32 inygam = iny + lddw * nm;
    i32 irhs = iny + lddw * ncol;
    i32 ixinit = irhs + lddw;

    i32 ie, ldr, ias;
    if (ncyc) {
        ic = ixinit + n2m;
        ia = withx0 ? (ic + ln) : ic;
        ldr = (ncol > 1) ? ncol : 1;
        ie = iny;
    } else {
        inih = withd ? (irhs + m) : irhs;
        ia = ixinit + n;
        ldr = lddw;
        ie = ixinit;
    }

    if (n > 0 && withx0) {
        ias = ia + nn;
    }

    i32 itauu = ia;
    itau = withd ? (itauu + m) : itauu;

    f64 dum = 0.0;
    i32 ierr;
    i32 int1 = 1;
    f64 dbl0 = 0.0, dbl1 = 1.0;

    f64* x0_work = (f64*)malloc(n * sizeof(f64));
    if (n > 0 && !x0_work) {
        *info = -23;
        return;
    }

    for (i32 icycle = 0; icycle < ncycle; icycle++) {
        bool first = (icycle == 0);
        if (!first && icycle == ncycle - 1) {
            nobs = nsmp - (ncycle - 1) * nobs;
            lnob = l * nobs;
        }

        i32 iy = iny;
        i32 ixsave = ixinit;

        // When M > 0, process inputs to build regression matrix
        if (m > 0) {
            for (i32 j = 0; j < m; j++) {
                for (i32 i = 0; i < n; i++) {
                    if (first) {
                        for (i32 k = 0; k < n; k++) dwork[ixsave + k] = 0.0;
                    }
                    for (i32 k = 0; k < n; k++) x0_work[k] = dwork[ixsave + k];

                    i32 ini = iy;

                    for (i32 kk = 0; kk < nobs; kk++) {
                        SLC_DGEMV("N", &l, &n, &dbl1, c, &ldc, x0_work, &int1,
                                  &dbl0, &dwork[iy], &nobs);
                        iy++;

                        SLC_DTRMV("U", "N", "N", &n, a, &lda, x0_work, &int1);

                        for (i32 ix = 1; ix < n; ix++) {
                            x0_work[ix] += a[ix + (ix - 1) * lda] * dwork[ixsave + ix - 1];
                        }

                        x0_work[i] += u[iupnt + kk + j * ldu];
                        for (i32 k = 0; k < n; k++) dwork[ixsave + k] = x0_work[k];
                    }

                    if (ncyc) ixsave += n;
                    iy = ini + lddw;
                }
            }
        } else if (n > 0 && withx0) {
            // When M = 0 and JOBX0 = 'X', we need to compute observation matrix
            // from the free system: x(k+1) = A*x(k), y(k) = C*x(k)
            // Initialize the starting state to what we'll solve for
            if (first) {
                for (i32 k = 0; k < n; k++) dwork[ixsave + k] = 0.0;
            }
            for (i32 k = 0; k < n; k++) x0_work[k] = dwork[ixsave + k];

            for (i32 kk = 0; kk < nobs; kk++) {
                SLC_DGEMV("N", &l, &n, &dbl1, c, &ldc, x0_work, &int1,
                          &dbl0, &dwork[iy], &nobs);
                iy++;

                SLC_DTRMV("U", "N", "N", &n, a, &lda, x0_work, &int1);

                for (i32 ix = 1; ix < n; ix++) {
                    x0_work[ix] += a[ix + (ix - 1) * lda] * dwork[ixsave + ix - 1];
                }

                for (i32 k = 0; k < n; k++) dwork[ixsave + k] = x0_work[k];
            }
        }

        if (n > 0 && withx0) {
            i32 jwork = ias + nn;
            i32 ig = inygam;
            i32 iexpon = (i32)(log((f64)nobs) / log(2.0));
            i32 irem = nobs - (1 << iexpon);
            bool power2 = (irem == 0);
            if (!power2) iexpon++;

            if (first) {
                for (i32 i = 0; i < n; i++) {
                    for (i32 k = 0; k < l; k++) {
                        dwork[ig + k * nobs] = c[k + i * ldc];
                    }
                    ig += lddw;
                }
            } else {
                i32 src = ic;
                ig = inygam;
                for (i32 i = 0; i < n; i++) {
                    for (i32 k = 0; k < l; k++) {
                        dwork[ig + k * nobs] = dwork[src + k];
                    }
                    src += l;
                    ig += lddw;
                }
            }

            SLC_DLACPY("U", &n, &n, a, &lda, &dwork[ia], &n);
            if (n > 1) {
                i32 nm1 = n - 1;
                i32 ldap1 = lda + 1;
                i32 np1 = n + 1;
                SLC_DCOPY(&nm1, &a[1], &ldap1, &dwork[ia + 1], &np1);
            }

            i32 i2 = 1;
            i32 nrow = 0;

            for (i32 jj = 0; jj < iexpon; jj++) {
                i32 igam = inygam;
                if (jj < iexpon - 1 || power2) {
                    nrow = i2;
                } else {
                    nrow = irem;
                }

                for (i32 ii = 0; ii < l; ii++) {
                    SLC_DLACPY("F", &nrow, &n, &dwork[igam], &lddw,
                               &dwork[igam + i2], &lddw);
                    SLC_DTRMM("R", "U", "N", "N", &nrow, &n, &dbl1,
                              &dwork[ia], &n, &dwork[igam + i2], &lddw);

                    i32 igg = igam;
                    for (i32 ix = 0; ix < n - 1; ix++) {
                        SLC_DAXPY(&nrow, &dwork[ia + ix * n + ix + 1],
                                  &dwork[igg + lddw], &int1, &dwork[igg + i2], &int1);
                        igg += lddw;
                    }

                    igam += nobs;
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

            if (ncyc && icycle < ncycle - 1) {
                ig = inygam + i2 + nrow - 1;
                i32 igs = ig;

                i32 dst = ic;
                for (i32 i = 0; i < n; i++) {
                    for (i32 k = 0; k < l; k++) {
                        dwork[dst + k] = dwork[ig + k * nobs];
                    }
                    dst += l;
                    ig += lddw;
                }

                SLC_DTRMM("R", "U", "N", "N", &l, &n, &dbl1, a, &lda,
                          &dwork[ic], &l);

                ig = igs;
                for (i32 ix = 0; ix < n - 1; ix++) {
                    f64 aval = a[ix + 1 + ix * lda];
                    SLC_DAXPY(&l, &aval, &dwork[ig + lddw], &nobs,
                              &dwork[ic + ix * l], &int1);
                    ig += lddw;
                }
            }
        }

        iy = irhs;
        for (i32 k = 0; k < l; k++) {
            SLC_DCOPY(&nobs, &y[iypnt + k * ldy], &int1, &dwork[iy], &int1);
            iy += nobs;
        }

        if (m > 0 && withd) {
            i32 jwork = itau;
            if (first) {
                i32 ini = iny + m;

                SLC_DGEQRF(&nobs, &m, u, &ldu, &dwork[itauu],
                           &dwork[jwork], &ldwork, &ierr);

                for (i32 k = 0; k < l; k++) {
                    i32 ldw_query = ldwork - jwork;
                    SLC_DORMQR("L", "T", &nobs, &ncp1, &m, u, &ldu,
                               &dwork[itauu], &dwork[iny + k * nobs],
                               &lddw, &dwork[jwork], &ldw_query, &ierr);
                }

                if (ncol > 0) {
                    jwork = itau + ncol;
                    i32 nobs_m = nobs - m;
                    SLC_DGEQRF(&nobs_m, &ncol, &dwork[ini], &lddw,
                               &dwork[itau], &dwork[jwork], &ldwork, &ierr);

                    i32 one = 1;
                    i32 ldw_query = ldwork - jwork;
                    SLC_DORMQR("L", "T", &nobs_m, &one, &ncol,
                               &dwork[ini], &lddw, &dwork[itau],
                               &dwork[irhs + m], &lddw, &dwork[jwork], &ldw_query, &ierr);

                    for (i32 k = 1; k < l; k++) {
                        i32 one_rhs = 1;
                        mb04od("F", ncol, one_rhs, nobs_m, &dwork[ini], lddw,
                               &dwork[ini + k * nobs], lddw,
                               &dwork[irhs + m], lddw,
                               &dwork[irhs + m + k * nobs], lddw,
                               &dwork[itau], &dwork[jwork]);
                    }
                }

                if (ncyc) {
                    SLC_DLACPY("U", &ncol, &ncp1, &dwork[ini], &lddw,
                               &dwork[inir], &ldr);

                    for (i32 k = 0; k < l; k++) {
                        SLC_DLACPY("F", &m, &ncp1, &dwork[iny + k * nobs], &lddw,
                                   &dwork[inis + k * m], &lm);
                    }
                }
            } else {
                i32 one_rhs = ncp1;
                mb04od("F", m, one_rhs, nobs, u, ldu, &u[iupnt], ldu,
                       &dwork[inis], lm, &dwork[iny], lddw,
                       &dwork[itauu], &dwork[jwork]);

                for (i32 k = 1; k < l; k++) {
                    for (i32 ix = 0; ix < m; ix++) {
                        SLC_MB04OY(nobs, ncp1, &u[iupnt + ix * ldu],
                               dwork[itauu + ix],
                               &dwork[inis + k * m + ix], lm,
                               &dwork[iny + k * nobs], lddw, &dwork[jwork]);
                    }
                }

                if (ncol > 0) {
                    jwork = itau + ncol;

                    for (i32 k = 0; k < l; k++) {
                        i32 one_rhs = 1;
                        mb04od("F", ncol, one_rhs, nobs, &dwork[inir], ldr,
                               &dwork[iny + k * nobs], lddw,
                               &dwork[inih], ldr,
                               &dwork[irhs + k * nobs], lddw,
                               &dwork[itau], &dwork[jwork]);
                    }
                }
            }
        } else if (ncol > 0) {
            i32 jwork = itau + ncol;
            if (first) {
                SLC_DGEQRF(&lddw, &ncol, &dwork[iny], &lddw, &dwork[itau],
                           &dwork[jwork], &ldwork, &ierr);

                i32 one = 1;
                i32 ldw_query = ldwork - jwork;
                SLC_DORMQR("L", "T", &lddw, &one, &ncol, &dwork[iny], &lddw,
                           &dwork[itau], &dwork[irhs], &lddw,
                           &dwork[jwork], &ldw_query, &ierr);

                if (ncyc) {
                    SLC_DLACPY("U", &ncol, &ncp1, &dwork[iny], &lddw,
                               &dwork[inir], &ldr);
                }
            } else {
                i32 one_rhs = 1;
                mb04od("F", ncol, one_rhs, lnob, &dwork[inir], ldr,
                       &dwork[iny], lddw, &dwork[inih], ldr,
                       &dwork[irhs], lddw, &dwork[itau], &dwork[jwork]);
            }
        }

        iupnt += nobs;
        iypnt += nobs;
    }

    f64 rcond;
    SLC_DTRCON("1", "U", "N", &ncol, &dwork[inir], &ldr, &rcond,
               &dwork[ie], iwork, &ierr);

    f64 toll = tol;
    if (toll <= 0.0) toll = SLC_DLAMCH("P");

    f64 thresh = pow(toll, 2.0 / 3.0);
    if (rcond <= thresh) {
        *iwarn = 4;

        if (ncol > 1) {
            i32 ncol_m1 = ncol - 1;
            SLC_DLASET("L", &ncol_m1, &ncol_m1, &dbl0, &dbl0,
                       &dwork[inir + 1], &ldr);
        }

        i32 isv = ie;
        i32 jwork = isv + ncol;
        i32 rank;
        i32 ldw_query = ldwork - jwork;
        SLC_DGELSS(&ncol, &ncol, &int1, &dwork[inir], &ldr, &dwork[inih], &ldr,
                   &dwork[isv], &toll, &rank, &dwork[jwork], &ldw_query, &ierr);

        if (ierr > 0) {
            *info = 2;
            free(x0_work);
            return;
        }
        i32 opt = (i32)dwork[jwork] - jwork + 1;
        if (opt > maxwrk) maxwrk = opt;
    } else {
        SLC_DTRSM("L", "U", "N", "N", &ncol, &int1, &dbl1,
                  &dwork[inir], &ldr, &dwork[inih], &ldr);
    }

    SLC_DLACPY("F", &n, &m, &dwork[inih], &n, b, &ldb);

    if (n > 0 && withx0) {
        SLC_DCOPY(&n, &dwork[inih + nm], &int1, x0, &int1);
    } else {
        for (i32 i = 0; i < n; i++) x0[i] = 0.0;
    }

    f64 rcondu = 1.0;
    if (m > 0 && withd) {
        if (ncyc) {
            irhs = inis + lm * ncol;
            f64 mone = -1.0;
            SLC_DGEMV("N", &lm, &ncol, &mone, &dwork[inis], &lm,
                      &dwork[inih], &int1, &dbl1, &dwork[irhs], &int1);
        } else {
            f64 mone = -1.0;
            for (i32 k = 0; k < l; k++) {
                SLC_DGEMV("N", &m, &ncol, &mone, &dwork[inis + k * nobs], &lddw,
                          &dwork[inih], &int1, &dbl1, &dwork[irhs + k * nobs], &int1);
            }

            for (i32 k = 1; k < l; k++) {
                SLC_DCOPY(&m, &dwork[irhs + k * nobs], &int1,
                          &dwork[irhs + k * m], &int1);
            }
        }

        SLC_DTRCON("1", "U", "N", &m, u, &ldu, &rcondu, &dwork[ie], iwork, &ierr);

        if (rcondu <= thresh) {
            *iwarn = 4;

            i32 iq = ie + m * m;
            i32 isv = iq + m * m;
            i32 jwork = isv + m;

            SLC_DLACPY("U", &m, &m, u, &ldu, &dwork[ie], &m);

            i32 rank;
            i32 ldw_query = ldwork - jwork;
            mb02ud("N", "L", "N", "N", m, l, 1.0, toll, &rank,
                   &dwork[ie], m, &dwork[iq], m, &dwork[isv],
                   &dwork[irhs], m, &dum, 1, &dwork[jwork], ldw_query, &ierr);

            if (ierr > 0) {
                *info = 2;
                free(x0_work);
                return;
            }
            i32 opt = (i32)dwork[jwork] - jwork + 1;
            if (opt > maxwrk) maxwrk = opt;
        } else {
            SLC_DTRSM("L", "U", "N", "N", &m, &l, &dbl1, u, &ldu,
                      &dwork[irhs], &m);
        }

        ma02ad("F", m, l, &dwork[irhs], m, d, ldd);
    }

    dwork[0] = (f64)maxwrk;
    dwork[1] = rcond;
    if (m > 0 && withd) {
        dwork[2] = rcondu;
    }

    free(x0_work);
}
