// SPDX-License-Identifier: BSD-3-Clause
//
// AB13DD - L-infinity norm of continuous/discrete-time system
//
// Translated from SLICOT AB13DD.f (Fortran 77)

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <complex.h>

#define MAXIT 30

static inline i32 max_i32(i32 a, i32 b) { return a > b ? a : b; }
static inline i32 min_i32(i32 a, i32 b) { return a < b ? a : b; }
static inline f64 max_f64(f64 a, f64 b) { return a > b ? a : b; }
static inline f64 min_f64(f64 a, f64 b) { return a < b ? a : b; }

void ab13dd(const char *dico, const char *jobe, const char *equil,
            const char *jobd, i32 n, i32 m, i32 p, f64 *fpeak,
            f64 *a, i32 lda, f64 *e, i32 lde, f64 *b, i32 ldb,
            f64 *c, i32 ldc, f64 *d, i32 ldd, f64 *gpeak, f64 tol,
            i32 *iwork, f64 *dwork, i32 ldwork, c128 *cwork, i32 lcwork,
            i32 *info)
{
    bool discr = (*dico == 'D' || *dico == 'd');
    bool fulle = (*jobe == 'G' || *jobe == 'g');
    bool lequil = (*equil == 'S' || *equil == 's');
    bool withd = (*jobd == 'D' || *jobd == 'd');

    i32 n2 = 2 * n;
    i32 nn = n * n;
    i32 pm = p + m;
    i32 n2pm = n2 + pm;
    i32 minpm = min_i32(p, m);

    *info = 0;

    if (!discr && !(*dico == 'C' || *dico == 'c')) {
        *info = -1;
    } else if (!fulle && !(*jobe == 'I' || *jobe == 'i')) {
        *info = -2;
    } else if (!lequil && !(*equil == 'N' || *equil == 'n')) {
        *info = -3;
    } else if (!withd && !(*jobd == 'Z' || *jobd == 'z')) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (m < 0) {
        *info = -6;
    } else if (p < 0) {
        *info = -7;
    } else if (min_f64(fpeak[0], fpeak[1]) < 0.0) {
        *info = -8;
    } else if (lda < max_i32(1, n)) {
        *info = -10;
    } else if (lde < 1 || (fulle && lde < n)) {
        *info = -12;
    } else if (ldb < max_i32(1, n)) {
        *info = -14;
    } else if (ldc < max_i32(1, p)) {
        *info = -16;
    } else if (ldd < 1 || (withd && ldd < p)) {
        *info = -18;
    } else if (tol < 0.0 || tol >= 1.0) {
        *info = -20;
    }

    if (*info != 0) {
        return;
    }

    f64 zero = 0.0, one = 1.0, two = 2.0, four = 4.0, p25 = 0.25;
    f64 ten = 10.0, hundrd = 100.0, thousd = 1000.0;

    f64 eps = SLC_DLAMCH("Epsilon");
    f64 safmin = SLC_DLAMCH("Safe minimum");
    f64 safmax = 1.0 / safmin;
    SLC_DLABAD(&safmin, &safmax);
    f64 smlnum = sqrt(safmin) / SLC_DLAMCH("Precision");
    f64 bignum = 1.0 / smlnum;
    f64 toler = sqrt(eps);

    i32 one_i = 1;
    f64 bnorm = SLC_DLANGE("1", &n, &m, b, &ldb, dwork);
    f64 cnorm = SLC_DLANGE("1", &p, &n, c, &ldc, dwork);
    bool nodyn = (n == 0) || (min_f64(bnorm, cnorm) == 0.0);
    bool usepen = fulle || discr;

    if (m == 0 || p == 0) {
        gpeak[0] = zero;
        fpeak[0] = zero;
        gpeak[1] = one;
        fpeak[1] = one;
        dwork[0] = one;
        cwork[0] = one;
        return;
    }

    f64 gammal = zero;
    i32 maxwrk = 1;
    i32 maxcwk = 1;
    i32 ierr = 0;

    i32 id = 0;
    i32 is = 0;
    i32 iwrk = 0;
    i32 ibv = 0, icu = 0, iu = 0, iv = 0;

    if (withd) {
        is = id + p * m;
        if (usepen || nodyn) {
            iu = is + minpm;
            iv = iu;
            iwrk = iv;
        } else {
            ibv = is + minpm;
            icu = ibv + n * m;
            iu = icu + p * n;
            iv = iu + p * p;
            iwrk = iv + m * m;
        }

        SLC_DLACPY("Full", &p, &m, d, &ldd, &dwork[id], &p);
        const char *vect = (usepen || nodyn) ? "N" : "A";
        SLC_DGESVD(vect, vect, &p, &m, &dwork[id], &p, &dwork[is],
                   &dwork[iu], &p, &dwork[iv], &m, &dwork[iwrk],
                   &(i32){ldwork - iwrk}, &ierr);
        if (ierr > 0) {
            *info = 3;
            return;
        }
        gammal = dwork[is];
        maxwrk = (i32)dwork[iwrk] + iwrk;
        SLC_DLACPY("Full", &p, &m, d, &ldd, &dwork[id], &p);
    } else {
        iwrk = 0;
        gammal = zero;
        maxwrk = 1;
    }

    if (nodyn) {
        gpeak[0] = gammal;
        fpeak[0] = zero;
        gpeak[1] = one;
        fpeak[1] = one;
        dwork[0] = (f64)maxwrk;
        cwork[0] = one;
        return;
    }

    if (!usepen && withd) {
        f64 one_d = 1.0, zero_d = 0.0;
        SLC_DGEMM("N", "T", &n, &m, &m, &one_d, b, &ldb, &dwork[iv], &m,
                  &zero_d, &dwork[ibv], &n);
        SLC_DGEMM("T", "N", &n, &p, &p, &one_d, c, &ldc, &dwork[iu], &p,
                  &zero_d, &dwork[icu], &n);
        iwrk = iu;
    }

    i32 ia = iwrk;
    i32 ie = ia + nn;
    i32 ib_local, ic_local, ir, ii_local, ibt;
    i32 ias = 0;

    if (fulle) {
        ib_local = ie + nn;
    } else {
        ib_local = ie;
    }
    ic_local = ib_local + n * m;
    ir = ic_local + p * n;
    ii_local = ir + n;
    ibt = ii_local + n;

    SLC_DLACPY("Full", &n, &n, a, &lda, &dwork[ia], &n);
    SLC_DLACPY("Full", &n, &m, b, &ldb, &dwork[ib_local], &n);
    SLC_DLACPY("Full", &p, &n, c, &ldc, &dwork[ic_local], &p);

    f64 anrm = SLC_DLANGE("M", &n, &n, &dwork[ia], &n, dwork);
    bool ilascl = false;
    f64 anrmto = anrm;
    if (anrm > zero && anrm < smlnum) {
        anrmto = smlnum;
        ilascl = true;
    } else if (anrm > bignum) {
        anrmto = bignum;
        ilascl = true;
    }
    if (ilascl) {
        SLC_DLASCL("G", &(i32){0}, &(i32){0}, &anrm, &anrmto, &n, &n,
                   &dwork[ia], &n, &ierr);
    }

    f64 enrm = 0.0, enrmto = 0.0;
    bool ilescl = false;
    i32 ilo = 1, ihi = n;

    if (fulle) {
        iwrk = ibt + n;
        SLC_DLACPY("Full", &n, &n, e, &lde, &dwork[ie], &n);

        enrm = SLC_DLANGE("M", &n, &n, &dwork[ie], &n, dwork);
        if (enrm > zero && enrm < smlnum) {
            enrmto = smlnum;
            ilescl = true;
        } else if (enrm > bignum) {
            enrmto = bignum;
            ilescl = true;
        } else if (enrm == zero) {
            *info = 1;
            return;
        }
        if (ilescl) {
            SLC_DLASCL("G", &(i32){0}, &(i32){0}, &enrm, &enrmto, &n, &n,
                       &dwork[ie], &n, &ierr);
        }

        if (lequil) {
            tg01ad("A", n, n, m, p, zero, &dwork[ia], n,
                   &dwork[ie], n, &dwork[ib_local], n, &dwork[ic_local], p,
                   &dwork[ii_local], &dwork[ir], &dwork[iwrk], &ierr);
        }

        SLC_DGGBAL("P", &n, &dwork[ia], &n, &dwork[ie], &n, &ilo, &ihi,
                   &dwork[ii_local], &dwork[ir], &dwork[iwrk], &ierr);

        for (i32 i = n - 1; i >= ihi; i--) {
            i32 k = (i32)dwork[ii_local + i] - 1;
            if (k >= 0 && k < n && k != i) {
                SLC_DSWAP(&m, &dwork[ib_local + i], &n, &dwork[ib_local + k], &n);
            }
            k = (i32)dwork[ir + i] - 1;
            if (k >= 0 && k < n && k != i) {
                SLC_DSWAP(&p, &dwork[ic_local + i * p], &one_i,
                          &dwork[ic_local + k * p], &one_i);
            }
        }
        for (i32 i = 0; i < ilo - 1; i++) {
            i32 k = (i32)dwork[ii_local + i] - 1;
            if (k >= 0 && k < n && k != i) {
                SLC_DSWAP(&m, &dwork[ib_local + i], &n, &dwork[ib_local + k], &n);
            }
            k = (i32)dwork[ir + i] - 1;
            if (k >= 0 && k < n && k != i) {
                SLC_DSWAP(&p, &dwork[ic_local + i * p], &one_i,
                          &dwork[ic_local + k * p], &one_i);
            }
        }

        tg01bd("G", "N", "N", n, m, p, ilo, ihi,
               &dwork[ia], n, &dwork[ie], n, &dwork[ib_local], n,
               &dwork[ic_local], p, dwork, one_i, dwork, one_i,
               &dwork[iwrk], ldwork - iwrk, &ierr);
        maxwrk = max_i32((i32)dwork[iwrk] + iwrk, maxwrk);

        f64 rcond_val;
        SLC_DTRCON("1", "U", "N", &n, &dwork[ie], &n, &rcond_val,
                   &dwork[iwrk], iwork, &ierr);
        if (rcond_val <= ten * (f64)n * eps) {
            *info = 1;
            return;
        }

        ias = iwrk;
        i32 ies = ias + nn;
        iwrk = ies + nn;
        SLC_DLACPY("F", &n, &n, &dwork[ia], &n, &dwork[ias], &n);
        SLC_DLACPY("F", &n, &n, &dwork[ie], &n, &dwork[ies], &n);
        SLC_DHGEQZ("E", "N", "N", &n, &ilo, &ihi, &dwork[ias], &n,
                   &dwork[ies], &n, &dwork[ir], &dwork[ii_local], &dwork[ibt],
                   dwork, &n, dwork, &n, &dwork[iwrk], &(i32){ldwork - iwrk}, &ierr);
        if (ierr != 0) {
            *info = 2;
            return;
        }
        maxwrk = max_i32((i32)dwork[iwrk] + iwrk, maxwrk);

        if (ilascl) {
            for (i32 i = 0; i < n; i++) {
                if (dwork[ii_local + i] != zero) {
                    if ((dwork[ir + i] / safmax) > (anrmto / anrm) ||
                        (safmin / dwork[ir + i]) > (anrm / anrmto)) {
                        f64 tm = fabs(dwork[ia + i * n + i + 1] / dwork[ir + i]);
                        dwork[ibt + i] *= tm;
                        dwork[ir + i] *= tm;
                        dwork[ii_local + i] *= tm;
                    } else if ((dwork[ii_local + i] / safmax) > (anrmto / anrm) ||
                               (safmin / dwork[ii_local + i]) > (anrm / anrmto)) {
                        f64 tm = fabs(dwork[ia + (i + 1) * (n + 1)] / dwork[ii_local + i]);
                        dwork[ibt + i] *= tm;
                        dwork[ir + i] *= tm;
                        dwork[ii_local + i] *= tm;
                    }
                }
            }
        }

        if (ilescl) {
            for (i32 i = 0; i < n; i++) {
                if (dwork[ii_local + i] != zero) {
                    if ((dwork[ibt + i] / safmax) > (enrmto / enrm) ||
                        (safmin / dwork[ibt + i]) > (enrm / enrmto)) {
                        f64 tm = fabs(dwork[ie + i * n + i + 1] / dwork[ibt + i]);
                        dwork[ibt + i] *= tm;
                        dwork[ir + i] *= tm;
                        dwork[ii_local + i] *= tm;
                    }
                }
            }
        }

        if (ilascl) {
            SLC_DLASCL("H", &(i32){0}, &(i32){0}, &anrmto, &anrm, &n, &n,
                       &dwork[ia], &n, &ierr);
            SLC_DLASCL("G", &(i32){0}, &(i32){0}, &anrmto, &anrm, &n, &one_i,
                       &dwork[ir], &n, &ierr);
            SLC_DLASCL("G", &(i32){0}, &(i32){0}, &anrmto, &anrm, &n, &one_i,
                       &dwork[ii_local], &n, &ierr);
        }
        if (ilescl) {
            SLC_DLASCL("U", &(i32){0}, &(i32){0}, &enrmto, &enrm, &n, &n,
                       &dwork[ie], &n, &ierr);
            SLC_DLASCL("G", &(i32){0}, &(i32){0}, &enrmto, &enrm, &n, &one_i,
                       &dwork[ibt], &n, &ierr);
        }

    } else {
        if (lequil) {
            f64 maxred = hundrd;
            tb01id("A", n, m, p, &maxred, &dwork[ia], n,
                   &dwork[ib_local], n, &dwork[ic_local], p,
                   &dwork[ii_local], &ierr);
        }

        SLC_DGEBAL("P", &n, &dwork[ia], &n, &ilo, &ihi, &dwork[ir], &ierr);

        for (i32 i = n - 1; i >= ihi; i--) {
            i32 k = (i32)dwork[ir + i] - 1;
            if (k >= 0 && k < n && k != i) {
                SLC_DSWAP(&m, &dwork[ib_local + i], &n, &dwork[ib_local + k], &n);
                SLC_DSWAP(&p, &dwork[ic_local + i * p], &one_i,
                          &dwork[ic_local + k * p], &one_i);
            }
        }
        for (i32 i = 0; i < ilo - 1; i++) {
            i32 k = (i32)dwork[ir + i] - 1;
            if (k >= 0 && k < n && k != i) {
                SLC_DSWAP(&m, &dwork[ib_local + i], &n, &dwork[ib_local + k], &n);
                SLC_DSWAP(&p, &dwork[ic_local + i * p], &one_i,
                          &dwork[ic_local + k * p], &one_i);
            }
        }

        i32 itau = ir;
        iwrk = itau + n;
        SLC_DGEHRD(&n, &ilo, &ihi, &dwork[ia], &n, &dwork[itau],
                   &dwork[iwrk], &(i32){ldwork - iwrk}, &ierr);
        maxwrk = max_i32((i32)dwork[iwrk] + iwrk, maxwrk);

        SLC_DORMHR("L", "T", &n, &m, &ilo, &ihi, &dwork[ia], &n,
                   &dwork[itau], &dwork[ib_local], &n, &dwork[iwrk],
                   &(i32){ldwork - iwrk}, &ierr);
        maxwrk = max_i32((i32)dwork[iwrk] + iwrk, maxwrk);

        SLC_DORMHR("R", "N", &p, &n, &ilo, &ihi, &dwork[ia], &n,
                   &dwork[itau], &dwork[ic_local], &p, &dwork[iwrk],
                   &(i32){ldwork - iwrk}, &ierr);
        maxwrk = max_i32((i32)dwork[iwrk] + iwrk, maxwrk);

        ias = ibt;
        iwrk = ias + nn;
        SLC_DLACPY("F", &n, &n, &dwork[ia], &n, &dwork[ias], &n);
        SLC_DHSEQR("E", "N", &n, &ilo, &ihi, &dwork[ias], &n,
                   &dwork[ir], &dwork[ii_local], dwork, &n,
                   &dwork[iwrk], &(i32){ldwork - iwrk}, &ierr);
        if (ierr > 0) {
            *info = 2;
            return;
        }
        maxwrk = max_i32((i32)dwork[iwrk] + iwrk, maxwrk);

        if (ilascl) {
            SLC_DLASCL("H", &(i32){0}, &(i32){0}, &anrmto, &anrm, &n, &n,
                       &dwork[ia], &n, &ierr);
            SLC_DLASCL("G", &(i32){0}, &(i32){0}, &anrmto, &anrm, &n, &one_i,
                       &dwork[ir], &n, &ierr);
            SLC_DLASCL("G", &(i32){0}, &(i32){0}, &anrmto, &anrm, &n, &one_i,
                       &dwork[ii_local], &n, &ierr);
        }
    }

    i32 im = fulle ? (iwrk - 2 * nn) : ias;
    i32 iar = im + n;
    i32 imin = ii_local;
    f64 wrmin = safmax;
    f64 bound = eps * thousd;
    f64 wmax = zero;

    if (discr) {
        gammal = zero;
        for (i32 i = 0; i < n; i++) {
            f64 tm = hypot(dwork[ir + i], dwork[ii_local + i]);
            if (fulle) {
                f64 bti = dwork[ibt + i];
                if (bti >= one || (bti < one && tm < safmax * bti)) {
                    tm = tm / bti;
                } else {
                    tm = safmax;
                }
            }
            if (tm != zero) {
                dwork[ii_local + i] = atan2(dwork[ii_local + i], dwork[ir + i]);
                dwork[ir + i] = log(tm);
            }
            dwork[im + i] = hypot(dwork[ir + i], dwork[ii_local + i]);
            f64 dist = fabs(one - tm);
            if (dist < wrmin) {
                imin = ii_local + i;
                wrmin = dist;
            }
            dwork[iar + i] = fabs(dwork[ir + i]);
        }
    } else {
        for (i32 i = 0; i < n; i++) {
            f64 tm = fabs(dwork[ir + i]);
            dwork[im + i] = hypot(dwork[ir + i], dwork[ii_local + i]);
            if (fulle) {
                f64 bti = dwork[ibt + i];
                if (bti >= one || (bti < one && dwork[im + i] < safmax * bti)) {
                    tm = tm / bti;
                    dwork[im + i] = dwork[im + i] / bti;
                } else {
                    if (tm < safmax * bti) {
                        tm = tm / bti;
                    } else {
                        tm = safmax;
                    }
                    dwork[im + i] = safmax;
                }
            }
            if (tm < wrmin) {
                imin = ii_local + i;
                wrmin = tm;
            }
            dwork[iar + i] = tm;
            if (dwork[im + i] > wmax) {
                wmax = dwork[im + i];
            }
        }
        bound = bound + eps * wmax;
    }

    if (wrmin < bound) {
        gpeak[0] = one;
        gpeak[1] = zero;
        f64 tm = fabs(dwork[imin]);
        if (discr) {
            tm = fabs(atan2(sin(tm), cos(tm)));
        }
        fpeak[0] = tm;
        if (tm < safmax) {
            fpeak[1] = one;
        } else {
            fpeak[1] = zero;
        }
        dwork[0] = (f64)maxwrk;
        cwork[0] = one;
        return;
    }

    i32 ias_local, ibs_local;
    if (discr) {
        ias_local = ia;
        ibs_local = ib_local;
        iwrk = iar + n;
    } else {
        ias_local = iar + n;
        ibs_local = ias_local + nn;
        iwrk = ibs_local + n * m;
        SLC_DLACPY("U", &n, &n, &dwork[ia], &n, &dwork[ias_local], &n);
        i32 nm1 = n - 1;
        i32 np1 = n + 1;
        SLC_DCOPY(&nm1, &dwork[ia + 1], &np1, &dwork[ias_local + 1], &np1);
        SLC_DLACPY("F", &n, &m, &dwork[ib_local], &n, &dwork[ibs_local], &n);
    }

    f64 gamma = ab13dx(dico, jobe, jobd, n, m, p, zero,
                       &dwork[ias_local], n, &dwork[ie], n,
                       &dwork[ibs_local], n, &dwork[ic_local], p,
                       &dwork[id], p, iwork, &dwork[iwrk], ldwork - iwrk,
                       cwork, lcwork, &ierr);
    maxwrk = max_i32((i32)dwork[iwrk] + iwrk, maxwrk);

    if (ierr >= 1 && ierr <= n) {
        gpeak[0] = one;
        fpeak[0] = zero;
        gpeak[1] = zero;
        fpeak[1] = one;
        goto done;
    } else if (ierr == n + 1) {
        *info = 3;
        return;
    }

    f64 fpeaks = fpeak[0];
    f64 fpeaki = fpeak[1];
    if (gammal < gamma) {
        gammal = gamma;
        fpeak[0] = zero;
        fpeak[1] = one;
    } else if (!discr) {
        fpeak[0] = one;
        fpeak[1] = zero;
    }
    maxcwk = (i32)creal(cwork[0]);

    if (discr) {
        f64 pi = four * atan(one);
        gamma = ab13dx(dico, jobe, jobd, n, m, p, pi,
                       &dwork[ia], n, &dwork[ie], n,
                       &dwork[ib_local], n, &dwork[ic_local], p,
                       &dwork[id], p, iwork, &dwork[iwrk], ldwork - iwrk,
                       cwork, lcwork, &ierr);
        maxcwk = max_i32((i32)creal(cwork[0]), maxcwk);
        maxwrk = max_i32((i32)dwork[iwrk] + iwrk, maxwrk);
        if (ierr >= 1 && ierr <= n) {
            gpeak[0] = one;
            fpeak[0] = pi;
            gpeak[1] = zero;
            fpeak[1] = one;
            goto done;
        } else if (ierr == n + 1) {
            *info = 3;
            return;
        }
        if (gammal < gamma) {
            gammal = gamma;
            fpeak[0] = pi;
            fpeak[1] = one;
        }
    } else {
        iwrk = ias_local;
        if (withd) {
            SLC_DLACPY("F", &p, &m, d, &ldd, &dwork[id], &p);
        }
    }

    if (min_f64(fpeaks, fpeaki) != zero) {
        gamma = ab13dx(dico, jobe, jobd, n, m, p, fpeaks,
                       &dwork[ia], n, &dwork[ie], n,
                       &dwork[ib_local], n, &dwork[ic_local], p,
                       &dwork[id], p, iwork, &dwork[iwrk], ldwork - iwrk,
                       cwork, lcwork, &ierr);
        maxcwk = max_i32((i32)creal(cwork[0]), maxcwk);
        maxwrk = max_i32((i32)dwork[iwrk] + iwrk, maxwrk);
        f64 tm = discr ? fabs(atan2(sin(fpeaks), cos(fpeaks))) : fpeaks;
        if (ierr >= 1 && ierr <= n) {
            gpeak[0] = one;
            fpeak[0] = tm;
            gpeak[1] = zero;
            fpeak[1] = one;
            goto done;
        } else if (ierr == n + 1) {
            *info = 3;
            return;
        }
        if (gammal < gamma) {
            gammal = gamma;
            fpeak[0] = tm;
            fpeak[1] = one;
        }
    }

    for (i32 i = 0; i < n; i++) {
        if (dwork[ii_local + i] >= zero && dwork[im + i] > zero) {
            f64 rat;
            if (dwork[im + i] >= one ||
                (dwork[im + i] < one && dwork[iar + i] < safmax * dwork[im + i])) {
                rat = dwork[iar + i] / dwork[im + i];
            } else {
                rat = one;
            }
            f64 omega = dwork[im + i] * sqrt(max_f64(p25, one - two * rat * rat));

            gamma = ab13dx(dico, jobe, jobd, n, m, p, omega,
                           &dwork[ia], n, &dwork[ie], n,
                           &dwork[ib_local], n, &dwork[ic_local], p,
                           &dwork[id], p, iwork, &dwork[iwrk], ldwork - iwrk,
                           cwork, lcwork, &ierr);
            maxcwk = max_i32((i32)creal(cwork[0]), maxcwk);
            maxwrk = max_i32((i32)dwork[iwrk] + iwrk, maxwrk);
            f64 tm = discr ? fabs(atan2(sin(omega), cos(omega))) : omega;
            if (ierr >= 1 && ierr <= n) {
                gpeak[0] = one;
                fpeak[0] = tm;
                gpeak[1] = zero;
                fpeak[1] = one;
                goto done;
            } else if (ierr == n + 1) {
                *info = 3;
                return;
            }
            if (gammal < gamma) {
                gammal = gamma;
                fpeak[0] = tm;
                fpeak[1] = one;
            }
        }
    }

    if (gammal == zero) {
        gpeak[0] = zero;
        fpeak[0] = zero;
        gpeak[1] = one;
        fpeak[1] = one;
        goto done;
    }

    f64 rtol = discr ? 0.0 : hundrd * toler;
    i32 iter = 0;

    while (iter <= MAXIT) {
        iter++;
        gamma = (one + tol) * gammal;
        usepen = fulle || discr;

        bool use_withd = false;
        if (!usepen && withd) {
            if (dwork[is] > eps * gamma) {
                f64 rcond_val;
                if (m != p) {
                    rcond_val = one - (dwork[is] / gamma) * (dwork[is] / gamma);
                } else if (minpm > 1) {
                    rcond_val = (gamma * gamma - dwork[is] * dwork[is]) /
                                (gamma * gamma - dwork[is + p - 1] * dwork[is + p - 1]);
                } else {
                    rcond_val = gamma * gamma - dwork[is] * dwork[is];
                }
                usepen = (rcond_val < rtol);
                if (!usepen) use_withd = true;
            }
        }

        if (usepen) {
            ii_local = ir + n2;
            ibt = ii_local + n2;
            i32 ih12 = ibt + n2;
            im = ih12;

            i32 ih = ih12;

            if (discr) {
                for (i32 j = 0; j < m; j++) {
                    for (i32 i_l = 0; i_l < n; i_l++) {
                        dwork[ih++] = b[i_l + j * ldb] / bnorm;
                    }
                    for (i32 i_l = 0; i_l < n + m; i_l++) {
                        dwork[ih + i_l] = zero;
                    }
                    dwork[ih + n + j] = one;
                    ih += n + m;
                    for (i32 i_l = 0; i_l < p; i_l++) {
                        dwork[ih++] = d[i_l + j * ldd] / gamma;
                    }
                }
                for (i32 j = 0; j < p; j++) {
                    for (i32 i_l = 0; i_l < n; i_l++) {
                        dwork[ih++] = zero;
                    }
                    for (i32 i_l = 0; i_l < n; i_l++) {
                        dwork[ih++] = c[j + i_l * ldc] / bnorm;
                    }
                    for (i32 i_l = 0; i_l < m; i_l++) {
                        dwork[ih++] = d[j + i_l * ldd] / gamma;
                    }
                    for (i32 i_l = 0; i_l < p; i_l++) {
                        dwork[ih + i_l] = zero;
                    }
                    dwork[ih + j] = one;
                    ih += p;
                }
            } else {
                for (i32 j = 0; j < p; j++) {
                    for (i32 i_l = 0; i_l < n; i_l++) {
                        dwork[ih++] = zero;
                    }
                    for (i32 i_l = 0; i_l < n; i_l++) {
                        dwork[ih++] = c[j + i_l * ldc] / bnorm;
                    }
                    for (i32 i_l = 0; i_l < p; i_l++) {
                        dwork[ih + i_l] = zero;
                    }
                    dwork[ih + j] = one;
                    ih += p;
                    for (i32 i_l = 0; i_l < m; i_l++) {
                        dwork[ih++] = d[j + i_l * ldd] / gamma;
                    }
                }
                for (i32 j = 0; j < m; j++) {
                    for (i32 i_l = 0; i_l < n; i_l++) {
                        dwork[ih++] = b[i_l + j * ldb] / bnorm;
                    }
                    for (i32 i_l = 0; i_l < n; i_l++) {
                        dwork[ih++] = zero;
                    }
                    for (i32 i_l = 0; i_l < p; i_l++) {
                        dwork[ih++] = d[i_l + j * ldd] / gamma;
                    }
                    for (i32 i_l = 0; i_l < m; i_l++) {
                        dwork[ih + i_l] = zero;
                    }
                    dwork[ih + j] = one;
                    ih += m;
                }
            }

            i32 itau = ih12 + n2pm * n2pm;
            iwrk = itau + pm;
            SLC_DGEQRF(&n2pm, &pm, &dwork[ih12], &n2pm, &dwork[itau],
                       &dwork[iwrk], &(i32){ldwork - iwrk}, &ierr);
            maxwrk = max_i32((i32)dwork[iwrk] + iwrk, maxwrk);

            SLC_DORGQR(&n2pm, &n2pm, &pm, &dwork[ih12], &n2pm,
                       &dwork[itau], &dwork[iwrk], &(i32){ldwork - iwrk}, &ierr);
            maxwrk = max_i32((i32)dwork[iwrk] + iwrk, maxwrk);

            i32 ipa = itau;
            i32 ipe = ipa + 4 * nn;
            iwrk = ipe + 4 * nn;

            f64 one_d = 1.0, zero_d = 0.0, mone = -1.0;
            SLC_DGEMM("T", "N", &n2, &n, &n, &one_d,
                      &dwork[ih12 + pm * n2pm], &n2pm, a, &lda,
                      &zero_d, &dwork[ipa], &n2);

            if (discr) {
                f64 scl = bnorm / gamma;
                SLC_DGEMM("T", "N", &n2, &n, &p, &scl,
                          &dwork[ih12 + pm * n2pm + n2 + m], &n2pm,
                          c, &ldc, &one_d, &dwork[ipa], &n2);
                if (fulle) {
                    SLC_DGEMM("T", "T", &n2, &n, &n, &one_d,
                              &dwork[ih12 + pm * n2pm + n], &n2pm,
                              e, &lde, &zero_d, &dwork[ipa + 2 * nn], &n2);
                } else {
                    ma02ad("F", n, n2, &dwork[ih12 + pm * n2pm + n],
                           n2pm, &dwork[ipa + 2 * nn], n2);
                }
            } else {
                f64 scl = bnorm / gamma;
                SLC_DGEMM("T", "N", &n2, &n, &p, &scl,
                          &dwork[ih12 + pm * n2pm + n2], &n2pm,
                          c, &ldc, &one_d, &dwork[ipa], &n2);
                SLC_DGEMM("T", "T", &n2, &n, &n, &mone,
                          &dwork[ih12 + pm * n2pm + n], &n2pm,
                          a, &lda, &zero_d, &dwork[ipa + 2 * nn], &n2);
                scl = -bnorm / gamma;
                SLC_DGEMM("T", "T", &n2, &n, &m, &scl,
                          &dwork[ih12 + pm * n2pm + n2 + p], &n2pm,
                          b, &ldb, &one_d, &dwork[ipa + 2 * nn], &n2);
            }

            i32 ny = discr ? n : n2;
            if (fulle) {
                SLC_DGEMM("T", "N", &n2, &n, &n, &one_d,
                          &dwork[ih12 + pm * n2pm], &n2pm, e, &lde,
                          &zero_d, &dwork[ipe], &n2);
            } else {
                ma02ad("F", ny, n2, &dwork[ih12 + pm * n2pm],
                       n2pm, &dwork[ipe], n2);
            }

            if (discr) {
                SLC_DGEMM("T", "T", &n2, &n, &n, &one_d,
                          &dwork[ih12 + pm * n2pm + n], &n2pm,
                          a, &lda, &zero_d, &dwork[ipe + 2 * nn], &n2);
                f64 scl = bnorm / gamma;
                SLC_DGEMM("T", "T", &n2, &n, &m, &scl,
                          &dwork[ih12 + pm * n2pm + n2], &n2pm,
                          b, &ldb, &one_d, &dwork[ipe + 2 * nn], &n2);
            } else {
                if (fulle) {
                    SLC_DGEMM("T", "T", &n2, &n, &n, &one_d,
                              &dwork[ih12 + pm * n2pm + n], &n2pm,
                              e, &lde, &zero_d, &dwork[ipe + 2 * nn], &n2);
                }
            }

            SLC_DGGEV("N", "N", &n2, &dwork[ipa], &n2, &dwork[ipe], &n2,
                      &dwork[ir], &dwork[ii_local], &dwork[ibt],
                      dwork, &n2, dwork, &n2,
                      &dwork[iwrk], &(i32){ldwork - iwrk}, &ierr);
            if (ierr > 0) {
                *info = 2;
                return;
            }
            maxwrk = max_i32((i32)dwork[iwrk] + iwrk, maxwrk);

            wmax = zero;
            for (i32 i = 0; i < n2; i++) {
                f64 tm = hypot(dwork[ir + i], dwork[ii_local + i]);
                f64 bt_i = dwork[ibt + i];
                if (bt_i >= one || (bt_i < one && tm < safmax * bt_i)) {
                    tm = tm / bt_i;
                } else {
                    tm = safmax;
                }
                wmax = max_f64(wmax, tm);
                if (discr) {
                    dwork[im + i] = tm;
                }
            }

            i32 nei = 0;
            f64 tol_thresh = toler * sqrt(hundrd + wmax);

            for (i32 i = 0; i < n2; i++) {
                f64 tm;
                if (discr) {
                    tm = fabs(one - dwork[im + i]);
                } else {
                    tm = fabs(dwork[ir + i]);
                    f64 bt_i = dwork[ibt + i];
                    if (bt_i >= one || (bt_i < one && tm < safmax * bt_i)) {
                        tm = tm / bt_i;
                    } else {
                        tm = safmax;
                    }
                }
                if (tm <= tol_thresh) {
                    dwork[ir + nei] = dwork[ir + i] / dwork[ibt + i];
                    dwork[ii_local + nei] = dwork[ii_local + i] / dwork[ibt + i];
                    nei++;
                }
            }

            if (nei == 0) {
                gpeak[0] = gammal;
                gpeak[1] = one;
                goto done;
            }

            i32 nws = 0;
            if (discr) {
                for (i32 i = 0; i < nei; i++) {
                    f64 tm = atan2(dwork[ii_local + i], dwork[ir + i]);
                    dwork[ir + i] = max_f64(eps, tm);
                    nws++;
                }
            } else {
                i32 j_cnt = 0;
                for (i32 i = 0; i < nei; i++) {
                    if (dwork[ii_local + i] > eps) {
                        dwork[ir + nws] = dwork[ii_local + i];
                        nws++;
                    } else if (dwork[ii_local + i] == eps) {
                        j_cnt++;
                        if (j_cnt == 1) {
                            dwork[ir + nws] = eps;
                            nws++;
                        }
                    }
                }
            }

            for (i32 i = 0; i < nws - 1; i++) {
                for (i32 j = i + 1; j < nws; j++) {
                    if (dwork[ir + j] < dwork[ir + i]) {
                        f64 tmp = dwork[ir + i];
                        dwork[ir + i] = dwork[ir + j];
                        dwork[ir + j] = tmp;
                    }
                }
            }

            i32 lw = 1;
            for (i32 i = 1; i < nws; i++) {
                if (dwork[ir + lw - 1] != dwork[ir + i]) {
                    dwork[ir + lw] = dwork[ir + i];
                    lw++;
                }
            }

            if (lw == 1) {
                if (iter == 1 && nws >= 1) {
                    dwork[ir + 1] = dwork[ir];
                    lw++;
                } else {
                    gpeak[0] = gammal;
                    gpeak[1] = one;
                    goto done;
                }
            }

            i32 iwrk2 = ir + lw;
            f64 gammas = gammal;

            for (i32 i = 0; i < lw - 1; i++) {
                f64 omega;
                if (discr) {
                    omega = (dwork[ir + i] + dwork[ir + i + 1]) / two;
                } else {
                    omega = sqrt(dwork[ir + i] * dwork[ir + i + 1]);
                }

                f64 test_gamma = ab13dx(dico, jobe, jobd, n, m, p, omega,
                                        &dwork[ia], n, &dwork[ie], n,
                                        &dwork[ib_local], n, &dwork[ic_local], p,
                                        &dwork[id], p, iwork, &dwork[iwrk2],
                                        ldwork - iwrk2, cwork, lcwork, &ierr);
                maxcwk = max_i32((i32)creal(cwork[0]), maxcwk);
                maxwrk = max_i32((i32)dwork[iwrk2] + iwrk2, maxwrk);
                f64 tm = discr ? fabs(atan2(sin(omega), cos(omega))) : omega;
                if (ierr >= 1 && ierr <= n) {
                    gpeak[0] = one;
                    fpeak[0] = tm;
                    gpeak[1] = zero;
                    fpeak[1] = one;
                    goto done;
                } else if (ierr == n + 1) {
                    *info = 3;
                    return;
                }
                if (gammal < test_gamma) {
                    gammal = test_gamma;
                    fpeak[0] = tm;
                    fpeak[1] = one;
                }
            }

            if (gammal < gammas * (one + tol / ten)) {
                gpeak[0] = gammal;
                gpeak[1] = one;
                goto done;
            }
        } else if (!withd || !use_withd) {
            i32 ih = ibt;
            i32 ih12 = ih + nn;
            i32 isl_nwd = ih12 + nn + n;

            SLC_DLACPY("F", &n, &n, a, &lda, &dwork[ih], &n);

            f64 neg_inv_gamma = -one / gamma;
            f64 pos_inv_gamma = one / gamma;
            f64 zero_d = 0.0;
            SLC_DSYRK("L", "T", &n, &p, &neg_inv_gamma, c, &ldc,
                      &zero_d, &dwork[ih12], &n);
            SLC_DSYRK("U", "N", &n, &m, &pos_inv_gamma, b, &ldb,
                      &zero_d, &dwork[ih12 + n], &n);

            iwrk = isl_nwd + nn;
            i32 ldt_work = (n > 1) ? n : 1;
            f64 *wr = &dwork[ir];
            f64 *wi = &dwork[ii_local];

            mb03xd("B", "E", "N", "N", n, &dwork[ih], n, &dwork[ih12], n,
                   &dwork[isl_nwd], ldt_work, NULL, 1, NULL, 1, NULL, 1, NULL, 1,
                   wr, wi, &ilo, &dwork[iwrk],
                   &dwork[iwrk + n], ldwork - iwrk - n, &ierr);

            if (ierr > 0) {
                usepen = true;
                continue;
            }
            maxwrk = max_i32((i32)dwork[iwrk] + iwrk + n - 1, maxwrk);

            wmax = zero;
            for (i32 i = 0; i < n; i++) {
                f64 tm = hypot(wr[i], wi[i]);
                wmax = max_f64(wmax, tm);
            }

            i32 nei = 0;
            f64 tol_thresh = toler * sqrt(hundrd + wmax);

            for (i32 i = 0; i < n; i++) {
                f64 tm = fabs(wr[i]);
                if (tm <= tol_thresh) {
                    wr[nei] = wr[i];
                    wi[nei] = wi[i];
                    nei++;
                }
            }

            if (nei == 0) {
                gpeak[0] = gammal;
                gpeak[1] = one;
                goto done;
            }

            i32 nws = 0;
            i32 j_cnt = 0;
            for (i32 i = 0; i < nei; i++) {
                if (wi[i] > eps) {
                    wr[nws] = wi[i];
                    nws++;
                } else if (wi[i] == eps) {
                    j_cnt++;
                    if (j_cnt == 1) {
                        wr[nws] = eps;
                        nws++;
                    }
                }
            }

            for (i32 i = 0; i < nws - 1; i++) {
                for (i32 j = i + 1; j < nws; j++) {
                    if (wr[j] < wr[i]) {
                        f64 tmp = wr[i];
                        wr[i] = wr[j];
                        wr[j] = tmp;
                    }
                }
            }

            i32 lw = 1;
            for (i32 i = 1; i < nws; i++) {
                if (wr[lw - 1] != wr[i]) {
                    wr[lw] = wr[i];
                    lw++;
                }
            }

            if (lw == 1) {
                if (iter == 1 && nws >= 1) {
                    wr[1] = wr[0];
                    lw++;
                } else {
                    gpeak[0] = gammal;
                    gpeak[1] = one;
                    goto done;
                }
            }

            i32 iwrk3 = iwrk + lw;
            f64 gammas = gammal;

            for (i32 i = 0; i < lw - 1; i++) {
                f64 omega = sqrt(wr[i] * wr[i + 1]);

                f64 test_gamma = ab13dx(dico, jobe, jobd, n, m, p, omega,
                                        &dwork[ia], n, &dwork[ie], n,
                                        &dwork[ib_local], n, &dwork[ic_local], p,
                                        &dwork[id], p, iwork, &dwork[iwrk3],
                                        ldwork - iwrk3, cwork, lcwork, &ierr);
                maxcwk = max_i32((i32)creal(cwork[0]), maxcwk);
                maxwrk = max_i32((i32)dwork[iwrk3] + iwrk3, maxwrk);
                if (ierr >= 1 && ierr <= n) {
                    gpeak[0] = one;
                    fpeak[0] = omega;
                    gpeak[1] = zero;
                    fpeak[1] = one;
                    goto done;
                } else if (ierr == n + 1) {
                    *info = 3;
                    return;
                }
                if (gammal < test_gamma) {
                    gammal = test_gamma;
                    fpeak[0] = omega;
                    fpeak[1] = one;
                }
            }

            if (gammal < gammas * (one + tol / ten)) {
                gpeak[0] = gammal;
                gpeak[1] = one;
                goto done;
            }
        } else {
            i32 ih = ibt;
            i32 ih12 = ih + nn;
            i32 isl = ih12 + nn + n;

            for (i32 i = 0; i < minpm; i++) {
                dwork[isl + i] = one / sqrt(gamma * gamma - dwork[is + i] * dwork[is + i]);
            }

            if (m < p) {
                dwork[isl + m] = one / gamma;
                for (i32 i = 0; i < p - m - 1; i++) {
                    dwork[isl + m + 1 + i] = dwork[isl + m];
                }
            }

            i32 isc = isl + max_i32(m, p);
            SLC_DLACPY("F", &n, &p, &dwork[icu], &n, &dwork[isc], &n);
            mb01sd('C', n, p, &dwork[isc], n, dwork, &dwork[isl]);

            i32 isb = isc + p * n;
            SLC_DLACPY("F", &n, &m, &dwork[ibv], &n, &dwork[isb], &n);
            mb01sd('C', n, minpm, &dwork[isb], n, dwork, &dwork[is]);

            mb01sd('C', n, minpm, &dwork[isb], n, dwork, &dwork[isl]);

            f64 one_d = 1.0, zero_d = 0.0;
            SLC_DLACPY("F", &n, &n, a, &lda, &dwork[ih], &n);
            SLC_DGEMM("N", "T", &n, &n, &minpm, &one_d,
                      &dwork[isb], &n, &dwork[isc], &n, &one_d,
                      &dwork[ih], &n);

            if (p < m) {
                dwork[isl + p] = one / gamma;
                for (i32 i = 0; i < m - p - 1; i++) {
                    dwork[isl + p + 1 + i] = dwork[isl + p];
                }
            }
            SLC_DLACPY("F", &n, &m, &dwork[ibv], &n, &dwork[isb], &n);
            mb01sd('C', n, m, &dwork[isb], n, dwork, &dwork[isl]);

            f64 mgamma = -gamma;
            SLC_DSYRK("L", "N", &n, &p, &mgamma, &dwork[isc], &n,
                      &zero_d, &dwork[ih12], &n);
            SLC_DSYRK("U", "N", &n, &m, &gamma, &dwork[isb], &n,
                      &zero_d, &dwork[ih12 + n], &n);

            iwrk = isl + nn;
            i32 ldt_work = (n > 1) ? n : 1;
            f64 *wr = &dwork[ir];
            f64 *wi = &dwork[ii_local];

            mb03xd("B", "E", "N", "N", n, &dwork[ih], n, &dwork[ih12], n,
                   &dwork[isl], ldt_work, NULL, 1, NULL, 1, NULL, 1, NULL, 1,
                   wr, wi, &ilo, &dwork[iwrk],
                   &dwork[iwrk + n], ldwork - iwrk - n, &ierr);

            if (ierr > 0) {
                usepen = true;
                continue;
            }
            maxwrk = max_i32((i32)dwork[iwrk] + iwrk + n - 1, maxwrk);

            wmax = zero;
            for (i32 i = 0; i < n; i++) {
                f64 tm = hypot(wr[i], wi[i]);
                wmax = max_f64(wmax, tm);
            }

            i32 nei = 0;
            f64 tol_thresh = toler * sqrt(hundrd + wmax);

            for (i32 i = 0; i < n; i++) {
                f64 tm = fabs(wr[i]);
                if (tm <= tol_thresh) {
                    wr[nei] = wr[i];
                    wi[nei] = wi[i];
                    nei++;
                }
            }

            if (nei == 0) {
                gpeak[0] = gammal;
                gpeak[1] = one;
                goto done;
            }

            i32 nws = 0;
            i32 j_cnt = 0;
            for (i32 i = 0; i < nei; i++) {
                if (wi[i] > eps) {
                    wr[nws] = wi[i];
                    nws++;
                } else if (wi[i] == eps) {
                    j_cnt++;
                    if (j_cnt == 1) {
                        wr[nws] = eps;
                        nws++;
                    }
                }
            }

            for (i32 i = 0; i < nws - 1; i++) {
                for (i32 j = i + 1; j < nws; j++) {
                    if (wr[j] < wr[i]) {
                        f64 tmp = wr[i];
                        wr[i] = wr[j];
                        wr[j] = tmp;
                    }
                }
            }

            i32 lw = 1;
            for (i32 i = 1; i < nws; i++) {
                if (wr[lw - 1] != wr[i]) {
                    wr[lw] = wr[i];
                    lw++;
                }
            }

            if (lw == 1) {
                if (iter == 1 && nws >= 1) {
                    wr[1] = wr[0];
                    lw++;
                } else {
                    gpeak[0] = gammal;
                    gpeak[1] = one;
                    goto done;
                }
            }

            iwrk = ir + lw;
            f64 gammas = gammal;

            for (i32 i = 0; i < lw - 1; i++) {
                f64 omega = sqrt(wr[i] * wr[i + 1]);

                f64 test_gamma = ab13dx(dico, jobe, jobd, n, m, p, omega,
                                        &dwork[ia], n, &dwork[ie], n,
                                        &dwork[ib_local], n, &dwork[ic_local], p,
                                        &dwork[id], p, iwork, &dwork[iwrk],
                                        ldwork - iwrk, cwork, lcwork, &ierr);
                maxcwk = max_i32((i32)creal(cwork[0]), maxcwk);
                maxwrk = max_i32((i32)dwork[iwrk] + iwrk, maxwrk);
                if (ierr >= 1 && ierr <= n) {
                    gpeak[0] = one;
                    fpeak[0] = omega;
                    gpeak[1] = zero;
                    fpeak[1] = one;
                    goto done;
                } else if (ierr == n + 1) {
                    *info = 3;
                    return;
                }
                if (gammal < test_gamma) {
                    gammal = test_gamma;
                    fpeak[0] = omega;
                    fpeak[1] = one;
                }
            }

            if (gammal < gammas * (one + tol / ten)) {
                gpeak[0] = gammal;
                gpeak[1] = one;
                goto done;
            }
        }
    }

    if (iter > MAXIT) {
        *info = 4;
        gpeak[0] = gammal;
        gpeak[1] = one;
        dwork[0] = (f64)maxwrk;
        cwork[0] = (c128)maxcwk;
        return;
    }

    gpeak[0] = gammal;
    gpeak[1] = one;

done:
    dwork[0] = (f64)maxwrk;
    cwork[0] = (c128)maxcwk;
}
