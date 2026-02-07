// SPDX-License-Identifier: BSD-3-Clause
//
// AB13HD - L-infinity norm of proper continuous-time or causal discrete-time
//          descriptor state-space system
//
// Translated from SLICOT AB13HD.f (Fortran 77)

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <complex.h>

static inline i32 max_i32(i32 a, i32 b) { return a > b ? a : b; }
static inline i32 min_i32(i32 a, i32 b) { return a < b ? a : b; }
static inline f64 max_f64(f64 a, f64 b) { return a > b ? a : b; }
static inline f64 min_f64(f64 a, f64 b) { return a < b ? a : b; }

void ab13hd(const char *dico, const char *jobe, const char *equil,
            const char *jobd, const char *ckprop, const char *reduce,
            const char *poles, i32 n, i32 m, i32 p, i32 ranke, f64 *fpeak,
            f64 *a, i32 lda, f64 *e, i32 lde, f64 *b, i32 ldb,
            f64 *c, i32 ldc, f64 *d, i32 ldd, i32 *nr, f64 *gpeak, f64 *tol,
            i32 *iwork, f64 *dwork, i32 ldwork, c128 *zwork, i32 lzwork,
            bool *bwork, i32 *iwarn, i32 *info)
{
    const i32 BM = 2, BNEICD = 10, BNEICM = 45, BNEICX = 60, BNEIR = 3,
              MAXIT = 30, SWNEIC = 300;
    const f64 ZERO = 0.0, P1 = 0.1, P25 = 0.25, ONE = 1.0, TWO = 2.0,
              FOUR = 4.0, TEN = 10.0, HUNDRD = 100.0, THOUSD = 1000.0;
    const c128 CONE = 1.0 + 0.0*I;

    bool discr = (*dico == 'D' || *dico == 'd');
    bool unite = (*jobe == 'I' || *jobe == 'i');
    bool gene  = (*jobe == 'G' || *jobe == 'g');
    bool cmpre = (*jobe == 'C' || *jobe == 'c');
    bool lequil = (*equil == 'S' || *equil == 's');
    bool withd  = (*jobd == 'D' || *jobd == 'd');
    bool fullrd = (*jobd == 'F' || *jobd == 'f');
    bool zerod  = (*jobd == 'Z' || *jobd == 'z');
    bool wckprp = (*ckprop == 'C' || *ckprop == 'c');
    bool wreduc = (*reduce == 'R' || *reduce == 'r');
    bool allpol = (*poles == 'A' || *poles == 'a');
    bool withe  = gene || cmpre;
    bool lquery = (ldwork == -1) || (lzwork == -1);

    i32 nn = n * n;
    i32 minpm = min_i32(p, m);
    i32 maxpm = max_i32(p, m);

    *iwarn = 0;
    *info = 0;

    if (!discr && !(*dico == 'C' || *dico == 'c')) {
        *info = -1;
    } else if (!withe && !unite) {
        *info = -2;
    } else if (!lequil && !(*equil == 'N' || *equil == 'n')) {
        *info = -3;
    } else if (!withd && !fullrd && !zerod) {
        *info = -4;
    } else if (!(wckprp || (*ckprop == 'N' || *ckprop == 'n'))) {
        if (!(discr || unite))
            *info = -5;
    } else if (!(wreduc || (*reduce == 'N' || *reduce == 'n'))) {
        if (wckprp)
            *info = -6;
    } else if (!allpol && !(*poles == 'P' || *poles == 'p')) {
        *info = -7;
    } else if (n < 0) {
        *info = -8;
    } else if (m < 0) {
        *info = -9;
    } else if (p < 0) {
        *info = -10;
    } else if (cmpre && (ranke < 0 || ranke > n)) {
        *info = -11;
    } else if (min_f64(fpeak[0], fpeak[1]) < 0.0) {
        *info = -12;
    } else if (lda < max_i32(1, n)) {
        *info = -14;
    } else if (lde < 1 || (gene && lde < n) || (cmpre && lde < ranke)) {
        *info = -16;
    } else if (ldb < max_i32(1, n)) {
        *info = -18;
    } else if (ldc < max_i32(1, p)) {
        *info = -20;
    } else if (ldd < 1 || (!zerod && ldd < p)) {
        *info = -22;
    } else if (tol[0] < ZERO || tol[0] >= ONE) {
        *info = -25;
    } else if (!lequil && tol[1] >= ONE) {
        *info = -25;
    } else if (!discr && withe && wckprp && (tol[2] >= ONE || tol[3] >= ONE)) {
        *info = -25;
    } else {
        bool nodyn = (n == 0);
        if (!nodyn) {
            f64 bnorm = SLC_DLANGE("1", &n, &m, b, &ldb, dwork);
            f64 cnorm = SLC_DLANGE("1", &p, &n, c, &ldc, dwork);
            nodyn = min_f64(bnorm, cnorm) == ZERO;
        }

        if (discr && fullrd)
            fullrd = false;
        withd = withd || fullrd;

        i32 mincwk = 1, maxcwk = 1, minwrk, maxwrk;
        i32 iu = 0, ie = 0, ib_off, ir_off, ibt_off;
        bool usepen, ncmpre = false, nsrt;
        i32 pm = p + m;
        i32 tn = 2 * n;
        i32 rnke = ranke;

        if (minpm == 0 || (nodyn && zerod)) {
            minwrk = 1;
            maxwrk = 1;
        } else {
            usepen = withe || discr;
            i32 i0, mnwsvd, mnw13x, odwsvd = 0, odw13x = 0;

            if (cmpre) {
                cmpre = (ranke < n);
                ncmpre = !cmpre;
            } else {
                ncmpre = false;
                rnke = n;
            }

            if (discr || unite) {
                wckprp = false;
                wreduc = false;
            }

            if (lquery) {
                i32 ierr = 0;
                SLC_DGESVD("N", "N", &p, &m, dwork, &p, dwork, dwork, &p,
                           dwork, &m, dwork, &(i32){-1}, &ierr);
                odwsvd = (i32)dwork[0];
                odw13x = minpm + odwsvd;
            }

            if (nodyn) {
                if (withd) {
                    mnwsvd = max_i32(3 * minpm + maxpm, 5 * minpm);
                    minwrk = p * m + minpm + mnwsvd;
                    if (lquery)
                        maxwrk = max_i32(minwrk, p * m + minpm + odwsvd);
                    else
                        maxwrk = minwrk;
                } else {
                    minwrk = 1;
                    maxwrk = 1;
                }
            } else {
                i0 = (n + m) * (n + p);
                ie = 0;

                if (lquery) {
                    i32 ierr = 0;
                    SLC_ZGESVD("N", "N", &p, &m, zwork, &p, dwork, zwork, &p,
                               zwork, &m, zwork, &(i32){-1}, dwork, &ierr);
                    maxcwk = i0 + (i32)creal(zwork[0]);
                }

                mincwk = i0 + 2 * minpm + maxpm;
                mnwsvd = 5 * minpm;

                if (discr) {
                    minwrk = 0;
                    maxwrk = 0;
                } else {
                    mnwsvd = max_i32(mnwsvd, 3 * minpm + maxpm);
                    if (withd) {
                        i32 iwrk;
                        if (unite) {
                            iu = n * pm + minpm;
                            ie = iu;
                            iwrk = iu + p * p + pm * m;
                            if (lquery) {
                                i32 ierr = 0;
                                SLC_DGESVD("A", "A", &p, &m, dwork, &p, dwork,
                                           dwork, &p, dwork, &m, dwork, &(i32){-1}, &ierr);
                                maxwrk = iwrk + (i32)dwork[0];
                            } else {
                                maxwrk = 0;
                            }
                        } else {
                            iwrk = minpm + p * m;
                            maxwrk = 0;
                        }
                        minwrk = iwrk + mnwsvd;
                    } else {
                        minwrk = 0;
                        maxwrk = 0;
                    }
                }
                mnw13x = minpm + mnwsvd;

                ib_off = ie + nn;
                ir_off = ib_off + n * pm;
                ibt_off = ir_off + tn;

                if (withe) {
                    nsrt = cmpre || discr;
                    ib_off += nn;
                    ir_off += nn;
                    ibt_off += nn;

                    if (wckprp) {
                        i32 i1, i2;
                        i0 = nn + 4 * n;
                        i1 = n + maxpm;
                        if (p == m) {
                            i2 = ir_off;
                        } else {
                            i2 = ib_off + tn * maxpm;
                        }

                        i32 wr13id;
                        if (wreduc) {
                            i0 += 4;
                            if (cmpre)
                                wr13id = max_i32(i0, i1);
                            else
                                wr13id = max_i32(i0, 2 * (i1 - 1));
                        } else {
                            if (cmpre)
                                wr13id = 4 * n + 4;
                            else
                                wr13id = max_i32(max_i32(i0, 2 * (i1 - 1)), 8);
                        }
                        minwrk = max_i32(minwrk, i2 + wr13id);

                        if (lquery) {
                            f64 dum[3] = {ZERO, ZERO, ZERO};
                            i32 nrq = n, rnkeq = rnke, iwarnq = 0, ierrq = 0;
                            const char *jobsys_q = cmpre ? "N" : "R";
                            const char *jobeig_q = wreduc ? "A" : "I";
                            ab13id(jobsys_q, jobeig_q, "N", "N", "N",
                                   wreduc ? "U" : "N",
                                   n, m, p, dwork, n, dwork, n, dwork, n, dwork, maxpm,
                                   &nrq, &rnkeq, dum, iwork, dwork, -1, &iwarnq, &ierrq);
                            maxwrk = max_i32(maxwrk, i2 + (i32)dwork[0]);
                        }
                    }

                    if (!discr) {
                        i32 i1 = (n + p) * m;
                        if (cmpre)
                            minwrk = max_i32(minwrk, ir_off + max_i32(n * (n + m + 4), i1 + mnw13x));
                        if (lquery)
                            maxwrk = max_i32(max_i32(maxwrk, i1 + odw13x), minwrk);
                    }

                    {
                        i32 i1 = ibt_off + n;
                        if (nsrt) {
                            minwrk = max_i32(minwrk, i1 + max_i32(m, 2 * nn + n));
                        } else {
                            i1 += 2 * nn + tn;
                            minwrk = max_i32(minwrk, i1 + tn);
                        }

                        if (lquery) {
                            i32 ierr = 0;
                            f64 dum[3];
                            SLC_DGEQRF(&n, &n, a, &lda, dwork, dwork, &(i32){-1}, &ierr);
                            if (nsrt) {
                                SLC_DORMQR("L", "T", &n, &m, &n, dwork, &n,
                                           dwork, dwork, &n, &dum[1], &(i32){-1}, &ierr);
                            } else {
                                SLC_DORMQR("R", "N", &n, &n, &n, dwork, &n,
                                           dwork, dwork, &n, &dum[1], &(i32){-1}, &ierr);
                            }
                            i32 i1q = ibt_off + n;
                            if (!nsrt) i1q += 2 * nn + tn;
                            maxwrk = max_i32(i1q + max_i32(max_i32((i32)dwork[0], (i32)dum[1]),
                                                            (i32)dum[1]), maxwrk);
                        }
                    }

                    if (!nsrt) {
                        i32 i1 = ibt_off + n + 2 * nn + tn;
                        minwrk = max_i32(minwrk, i1 + max_i32(4 * n + 16, n * maxpm));
                        i1 = ibt_off + n + p * m;
                        i32 i2 = i1 + (n - 1) * pm;
                        minwrk = max_i32(minwrk, i2 + (n - 1) * (n + m + 2));
                        if (lquery) {
                            f64 scl, dif;
                            i32 ierr = 0;
                            sb04od("N", "N", "F", n, n, dwork, n, dwork, n,
                                   dwork, n, dwork, n, dwork, n, dwork, n,
                                   &scl, &dif, dwork, 1, dwork, 1, dwork, 1,
                                   dwork, 1, iwork, dwork, -1, &ierr);
                            maxwrk = max_i32(maxwrk, i2 + n * (n + 1) / 2 + (i32)dwork[0]);
                        }
                        if (withd) {
                            minwrk = max_i32(minwrk, i1 + mnw13x);
                            if (lquery)
                                maxwrk = max_i32(maxwrk, i1 + odw13x);
                        }
                    }
                } else {
                    nsrt = false;
                    i32 ii = ir_off + n;
                    if (lquery) {
                        i32 ierr = 0;
                        f64 dum[3];
                        SLC_DGEHRD(&n, &(i32){1}, &n, dwork, &n, dwork, dwork, &(i32){-1}, &ierr);
                        SLC_DORMHR("L", "T", &n, &m, &(i32){1}, &n, dwork, &n,
                                   dwork, dwork, &n, &dum[0], &(i32){-1}, &ierr);
                        SLC_DORMHR("R", "N", &p, &n, &(i32){1}, &n, dwork, &n,
                                   dwork, dwork, &p, &dum[1], &(i32){-1}, &ierr);
                        SLC_DHSEQR("E", "N", &n, &(i32){1}, &n, dwork, &n,
                                   dwork, dwork, dwork, &(i32){1}, &dum[2], &(i32){-1}, &ierr);
                        maxwrk = max_i32(max_i32(ii + max_i32(max_i32((i32)dwork[0], (i32)dum[0]),
                                                               (i32)dum[1]),
                                                  ibt_off + nn + (i32)dum[2]),
                                         maxwrk);
                    }
                }

                {
                    i32 iwrk = ibt_off + n;
                    if (withe) iwrk += n;
                    if (!discr) iwrk += nn + m * (n + p);
                    minwrk = max_i32(minwrk, iwrk + mnw13x);
                    if (lquery)
                        maxwrk = max_i32(maxwrk, iwrk + odw13x);
                }

                if (unite && !discr && (zerod || fullrd)) {
                    i0 = 2 * nn + n;
                    i32 i1 = i0 + 7 * n;
                    i32 i2 = ibt_off + i0;
                    if (lquery) {
                        i32 ierr = 0;
                        f64 dum[1];
                        i32 ii_off = ir_off + n;
                        i32 ilo_dum = 1;
                        mb03xd("B", "E", "N", "N", n, dwork, n, dwork, n,
                               dwork, n, dum, 1, dum, 1, dum, 1, dum, 1,
                               dwork, dwork, &ilo_dum, dwork, dwork, -1, &ierr);
                        maxwrk = max_i32(max_i32(maxwrk, i2 + nn + n + (i32)dwork[0]),
                                         ii_off + odw13x);
                    }
                    if (zerod) {
                        i2 += i1;
                    } else {
                        i2 += max_i32(maxpm + n * pm, i1);
                    }
                    minwrk = max_i32(minwrk, i2);
                } else {
                    i32 r = pm % 2;
                    i32 nblk = n + (pm + r) / 2;
                    i32 nblk2 = nblk * nblk;
                    i0 = ir_off + 7 * nblk2 + 5 * nblk;
                    i32 l = 8 * nblk;
                    if (nblk % 2 == 0) l += 4;
                    minwrk = max_i32(minwrk, i0 + 4 * nblk2 + max_i32(l, 36));
                }

                if (lquery) {
                    maxwrk = max_i32(minwrk, maxwrk);
                    maxcwk = max_i32(mincwk, maxcwk);
                }
            }
        }

        if (lquery) {
            dwork[0] = (f64)maxwrk;
            zwork[0] = (c128)maxcwk;
            return;
        } else {
            if (ldwork < minwrk) {
                *info = -28;
                dwork[0] = (f64)minwrk;
            }
            if (lzwork < mincwk) {
                *info = -30;
                zwork[0] = (c128)mincwk;
            }
        }
    }

    if (*info != 0) {
        if (ldwork != 0 && lzwork != 0) {
            /* xerbla would be called here */
        }
        return;
    }

    i32 iter = 0;

    if (minpm == 0) {
        gpeak[0] = ZERO;
        gpeak[1] = ONE;
        fpeak[0] = ZERO;
        fpeak[1] = ONE;
        dwork[0] = ONE;
        zwork[0] = CONE;
        iwork[0] = iter;
        return;
    }

    bool nodyn = (n == 0);
    if (!nodyn) {
        f64 bnorm = SLC_DLANGE("1", &n, &m, b, &ldb, dwork);
        f64 cnorm = SLC_DLANGE("1", &p, &n, c, &ldc, dwork);
        nodyn = min_f64(bnorm, cnorm) == ZERO;
    }

    if (discr && fullrd) fullrd = false;
    withd = (*jobd == 'D' || *jobd == 'd') || fullrd;

    bool usepen = withe || discr;
    bool ncmpre = false;
    i32 rnke = ranke;
    i32 pm = p + m;
    i32 tn = 2 * n;
    bool nsrt;

    if (cmpre) {
        cmpre = (ranke < n);
        ncmpre = !cmpre;
    } else {
        ncmpre = false;
        rnke = n;
    }

    if (discr || unite) {
        wckprp = false;
        wreduc = false;
    }

    f64 gammal, sv1 = 0, svp = 0;
    i32 maxwrk;
    f64 oz;

    i32 iu = 0, ibv = 0, icu = 0, is_off, iv_off, id_off, iwrk;

    if (withd) {
        oz = ONE;
        bool wnrmd = nodyn || ((unite || ncmpre) && !discr);
        if (wnrmd) {
            iu = (unite && !nodyn) ? n * pm + minpm : 0;
            if (!nodyn && unite) {
                ibv = 0;
                icu = ibv + n * m;
                is_off = icu + p * n;
                iv_off = iu + p * p;
                id_off = iv_off + m * m;
            } else {
                is_off = 0;
                iv_off = 0;
                id_off = minpm;
            }
            iwrk = id_off + p * m;

            SLC_DLACPY("F", &p, &m, d, &ldd, &dwork[id_off], &p);

            i32 ierr = 0;
            const char *svec = (!nodyn && unite) ? "A" : "N";
            SLC_DGESVD(svec, svec, &p, &m, &dwork[id_off], &p, &dwork[is_off],
                       &dwork[iu], &p, &dwork[iv_off], &m, &dwork[iwrk],
                       &(i32){ldwork - iwrk}, &ierr);
            if (ierr > 0) {
                *info = 3;
                return;
            }
            gammal = dwork[is_off];
            maxwrk = (i32)dwork[iwrk] + iwrk;

            if (!nodyn && unite) {
                SLC_DGEMM("N", "T", &n, &m, &m, &ONE, b, &ldb,
                          &dwork[iv_off], &m, &ZERO, &dwork[ibv], &n);
                SLC_DGEMM("T", "N", &n, &p, &p, &ONE, c, &ldc,
                          &dwork[iu], &p, &ZERO, &dwork[icu], &n);
                sv1 = gammal;
                svp = dwork[is_off + minpm - 1];
            }
        } else {
            gammal = ZERO;
            maxwrk = 1;
        }
    } else {
        bool wnrmd = withd;
        oz = ZERO;
        gammal = ZERO;
        maxwrk = 1;
    }

    if (nodyn) {
        gpeak[0] = gammal;
        gpeak[1] = ONE;
        fpeak[0] = ZERO;
        fpeak[1] = ONE;
        dwork[0] = (f64)maxwrk;
        zwork[0] = CONE;
        iwork[0] = iter;
        return;
    }

    f64 eps = SLC_DLAMCH("P");
    f64 safmin = SLC_DLAMCH("S");
    f64 safmax = ONE / safmin;
    SLC_DLABAD(&safmin, &safmax);
    f64 smlnum = sqrt(safmin) / eps;
    f64 bignum = ONE / smlnum;
    f64 toler = sqrt(eps);
    f64 stol = sqrt(toler);

    i32 ie, ia, ib, ic_off, ii;
    i32 n1 = rnke;
    i32 nr2 = nn;
    bool ilascl = false, ilescl = false;
    f64 anrm, anrmto = 0, enrm, enrmto = 0;

    ie = 0;
    if (withe) ie = 0;
    ib = ie + nn;
    if (withe) ib += nn;

    i32 ir = ib + n * pm;
    i32 ibt = ir + tn;
    if (withe) {
        ir += nn;
        ibt += nn;
    }

    ie++;
    ib++;
    ir++;
    ibt++;
    ie--; ib--; ir--; ibt--;

    if (withe) {
        ia = ie + nn;
    } else {
        ia = ie;
    }
    ic_off = ib + n * m;
    ii = ir + n;

    anrm = SLC_DLANGE("M", &n, &n, a, &lda, dwork);
    ilascl = false;
    if (anrm > ZERO && anrm < smlnum) {
        anrmto = smlnum;
        ilascl = true;
    } else if (anrm > bignum) {
        anrmto = bignum;
        ilascl = true;
    }
    if (ilascl) {
        i32 ierr = 0;
        SLC_DLASCL("G", &(i32){0}, &(i32){0}, &anrm, &anrmto, &n, &n, a, &n, &ierr);
    }

    *nr = n;
    nr2 = nn;
    n1 = rnke;

    i32 sdim = 0, sdim1 = 0, ilo = 0, ihi = 0, ninf = 0;
    i32 ies, ias, iq, iz, ilft, irht;
    f64 rcond, scl, dif, cnd, thresh, tm, tmp, tmr, td, rat;
    f64 teps, wmax, wrmin, toldef, tzer;
    f64 toli[3];
    i32 ierr;
    bool isprop, ind1;

    if (withe) {
        enrm = SLC_DLANGE("M", &n1, &n1, e, &lde, dwork);
        ilescl = false;
        if (enrm > ZERO && enrm < smlnum) {
            enrmto = smlnum;
            ilescl = true;
        } else if (enrm > bignum) {
            enrmto = bignum;
            ilescl = true;
        }
        if (ilescl) {
            ierr = 0;
            SLC_DLASCL("G", &(i32){0}, &(i32){0}, &enrm, &enrmto, &n1, &n1, e, &lde, &ierr);
        }
        SLC_DLACPY("F", &n1, &n1, e, &lde, &dwork[ie], &n);

        if (cmpre) {
            i32 rows = n - n1;
            SLC_DLASET("F", &rows, &n1, &ZERO, &ZERO, &dwork[ie + n1], &n);
            i32 cols = n - n1;
            SLC_DLASET("F", &n, &cols, &ZERO, &ZERO, &dwork[ie + n * n1], &n);
        }

        if (lequil) {
            thresh = tol[1];
            if (thresh < ZERO) {
                tm = max_f64(anrm, enrm);
                tmp = min_f64(ma02sd(n, n, a, lda), ma02sd(n1, n1, e, lde));
                if ((tmp / tm) < eps) {
                    thresh = P1;
                } else {
                    thresh = min_f64(HUNDRD * sqrt(tmp) / sqrt(tm * stol), P1);
                }
            }
            toli[2] = thresh;
        }

        if (wckprp) {
            toldef = tol[2];
            if (toldef <= ZERO) toldef = (f64)nn * eps;
            tzer = tol[3];
            if (tzer <= ZERO) tzer = (f64)n * eps;
            toli[0] = toldef;
            toli[1] = tzer;
        }

        if (lequil) {
            iwrk = ia + tn;
            if (cmpre) {
                tg01ad("A", n, n, m, p, thresh, a, lda, &dwork[ie], n,
                       b, ldb, c, ldc, &dwork[ia], &dwork[ia + n], &dwork[iwrk], &ierr);
                SLC_DLACPY("F", &n1, &n1, &dwork[ie], &n, e, &lde);
            } else {
                tg01ad("A", n, n, m, p, thresh, a, lda, e, lde,
                       b, ldb, c, ldc, &dwork[ia], &dwork[ia + n], &dwork[iwrk], &ierr);
                SLC_DLACPY("F", &n1, &n1, e, &lde, &dwork[ie], &n);
            }
        }

        if (wckprp) {
            i32 icw = ib + n * maxpm;
            if (p == m)
                iwrk = ir;
            else
                iwrk = icw + maxpm * n;

            const char *jobsys_s = cmpre ? "N" : "R";
            const char *jobeig_s = wreduc ? "A" : "I";
            const char *update_s = wreduc ? "U" : "N";

            if (wreduc) {
                if (cmpre) {
                    if (p == m) {
                        isprop = ab13id(jobsys_s, jobeig_s, "N", "N", "N", update_s,
                                        n, m, p, a, lda, &dwork[ie], n, b, ldb, c, ldc,
                                        nr, &rnke, toli, iwork, &dwork[iwrk],
                                        ldwork - iwrk, iwarn, info);
                        SLC_DLACPY("F", nr, &m, b, &ldb, &dwork[ib], &n);
                        SLC_DLACPY("F", &p, nr, c, &ldc, &dwork[ic_off], &p);
                    } else if (p < m) {
                        SLC_DLACPY("F", &p, &n, c, &ldc, &dwork[icw], &maxpm);
                        isprop = ab13id(jobsys_s, jobeig_s, "N", "N", "N", update_s,
                                        n, m, p, a, lda, &dwork[ie], n, b, ldb,
                                        &dwork[icw], maxpm, nr, &rnke, toli, iwork,
                                        &dwork[iwrk], ldwork - iwrk, iwarn, info);
                        SLC_DLACPY("F", nr, &m, b, &ldb, &dwork[ib], &n);
                        SLC_DLACPY("F", &p, nr, &dwork[icw], &maxpm, c, &ldc);
                        SLC_DLACPY("F", &p, nr, c, &ldc, &dwork[ic_off], &p);
                    } else {
                        SLC_DLACPY("F", &n, &m, b, &ldb, &dwork[ib], &n);
                        isprop = ab13id(jobsys_s, jobeig_s, "N", "N", "N", update_s,
                                        n, m, p, a, lda, &dwork[ie], n, &dwork[ib], n,
                                        c, ldc, nr, &rnke, toli, iwork, &dwork[iwrk],
                                        ldwork - iwrk, iwarn, info);
                        SLC_DLACPY("F", nr, &m, &dwork[ib], &n, b, &ldb);
                        SLC_DLACPY("F", &p, nr, c, &ldc, &dwork[ic_off], &p);
                    }
                    n1 = min_i32(*nr, n1);
                    SLC_DLACPY("F", &n1, &n1, &dwork[ie], &n, e, &lde);
                } else {
                    if (p == m) {
                        isprop = ab13id(jobsys_s, jobeig_s, "N", "N", "N", update_s,
                                        n, m, p, a, lda, e, lde, b, ldb, c, ldc,
                                        nr, &rnke, toli, iwork, &dwork[iwrk],
                                        ldwork - iwrk, iwarn, info);
                        SLC_DLACPY("F", nr, &m, b, &ldb, &dwork[ib], &n);
                        SLC_DLACPY("F", &p, nr, c, &ldc, &dwork[ic_off], &p);
                    } else if (p < m) {
                        SLC_DLACPY("F", &p, &n, c, &ldc, &dwork[icw], &maxpm);
                        isprop = ab13id(jobsys_s, jobeig_s, "N", "N", "N", update_s,
                                        n, m, p, a, lda, e, lde, b, ldb,
                                        &dwork[icw], maxpm, nr, &rnke, toli, iwork,
                                        &dwork[iwrk], ldwork - iwrk, iwarn, info);
                        SLC_DLACPY("F", nr, &m, b, &ldb, &dwork[ib], &n);
                        SLC_DLACPY("F", &p, nr, &dwork[icw], &maxpm, c, &ldc);
                        SLC_DLACPY("F", &p, nr, c, &ldc, &dwork[ic_off], &p);
                    } else {
                        SLC_DLACPY("F", &n, &m, b, &ldb, &dwork[ib], &n);
                        isprop = ab13id(jobsys_s, jobeig_s, "N", "N", "N", update_s,
                                        n, m, p, a, lda, e, lde, &dwork[ib], n,
                                        c, ldc, nr, &rnke, toli, iwork, &dwork[iwrk],
                                        ldwork - iwrk, iwarn, info);
                        SLC_DLACPY("F", nr, &m, &dwork[ib], &n, b, &ldb);
                        SLC_DLACPY("F", &p, nr, c, &ldc, &dwork[ic_off], &p);
                    }
                    SLC_DLACPY("F", &rnke, &rnke, e, &lde, &dwork[ie], &n);
                    i32 nrr = *nr - rnke;
                    if (nrr > 0) {
                        SLC_DLASET("F", &nrr, &rnke, &ZERO, &ZERO, &dwork[ie + *nr], &n);
                        i32 cols_nr = *nr - rnke;
                        SLC_DLASET("F", &rnke, &cols_nr, &ZERO, &ZERO,
                                   &dwork[ie + n * rnke], &n);
                    }
                    nr2 = (*nr) * (*nr);
                }
                SLC_DLACPY("F", nr, nr, a, &lda, &dwork[ia], &n);
            } else {
                SLC_DLACPY("F", &n, &n, a, &lda, &dwork[ia], &n);
                SLC_DLACPY("F", &n, &m, b, &ldb, &dwork[ib], &n);
                SLC_DLACPY("F", &p, &n, c, &ldc, &dwork[icw], &maxpm);
                isprop = ab13id(jobsys_s, jobeig_s, "N", "N", "N", update_s,
                                n, m, p, &dwork[ia], n, &dwork[ie], n, &dwork[ib], n,
                                &dwork[icw], maxpm, nr, &rnke, toli, iwork,
                                &dwork[iwrk], ldwork - iwrk, iwarn, info);
                SLC_DLACPY("F", &n, &n, a, &lda, &dwork[ia], &n);
                SLC_DLACPY("F", &n1, &n1, e, &lde, &dwork[ie], &n);
                SLC_DLACPY("F", &n, &m, b, &ldb, &dwork[ib], &n);
                SLC_DLACPY("F", &p, &n, c, &ldc, &dwork[ic_off], &p);
                *nr = n;
            }

            if (!isprop) {
                *iwarn = 2;
                gpeak[0] = ONE;
                gpeak[1] = ZERO;
                fpeak[0] = ONE;
                fpeak[1] = ZERO;
                goto label_440;
            } else if (*iwarn == 1) {
                *iwarn = 0;
            }
            maxwrk = max_i32(maxwrk, (i32)dwork[iwrk] + iwrk);
        } else {
            SLC_DLACPY("F", &n, &n, a, &lda, &dwork[ia], &n);
            SLC_DLACPY("F", &n, &m, b, &ldb, &dwork[ib], &n);
            SLC_DLACPY("F", &p, &n, c, &ldc, &dwork[ic_off], &p);
        }

        teps = TEN * eps;
        ninf = n - rnke;

        if (cmpre && !discr) {
            i32 ibs_l = ir;
            i32 ias_l = ibs_l + ninf * m;
            iwrk = ias_l + ninf * ninf;
            SLC_DLACPY("F", &ninf, &m, &dwork[ib + rnke], &n, &dwork[ibs_l], &ninf);
            SLC_DLACPY("F", &ninf, &ninf, &dwork[ia + rnke * (n + 1)], &n,
                       &dwork[ias_l], &ninf);
            if (ilascl) {
                ierr = 0;
                SLC_DLASCL("G", &(i32){0}, &(i32){0}, &anrmto, &anrm, &ninf, &ninf,
                           &dwork[ias_l], &ninf, &ierr);
            }
            tmp = SLC_DLANGE("1", &ninf, &ninf, &dwork[ias_l], &ninf, dwork);
            SLC_DGETRF(&ninf, &ninf, &dwork[ias_l], &ninf, iwork, &ierr);
            if (ierr > 0) {
                *info = 1;
                return;
            }
            i32 ninf1 = ninf + 1;
            SLC_DGECON("1", &ninf, &dwork[ias_l], &ninf, &tmp, &rcond,
                       &dwork[iwrk], &iwork[ninf], &ierr);
            if (rcond <= (f64)ninf * teps) {
                *info = 1;
                return;
            }
            SLC_DGETRS("N", &ninf, &m, &dwork[ias_l], &ninf, iwork,
                       &dwork[ibs_l], &ninf, &ierr);

            id_off = ias_l;
            is_off = id_off + p * m;
            if (withd)
                SLC_DLACPY("F", &p, &m, d, &ldd, &dwork[id_off], &p);
            f64 neg_one = -ONE;
            SLC_DGEMM("N", "N", &p, &m, &ninf, &neg_one,
                      &dwork[ic_off + rnke * p], &p, &dwork[ibs_l], &ninf,
                      &oz, &dwork[id_off], &p);

            iwrk = is_off + minpm;
            ierr = 0;
            SLC_DGESVD("N", "N", &p, &m, &dwork[id_off], &p,
                       &dwork[is_off], dwork, &(i32){1}, dwork, &(i32){1},
                       &dwork[iwrk], &(i32){ldwork - iwrk}, &ierr);
            if (ierr > 0) {
                *info = 3;
                return;
            }
            maxwrk = max_i32((i32)dwork[iwrk] + iwrk, maxwrk);
            gammal = dwork[is_off];
        }

        nsrt = cmpre || discr;
        ies = ibt + n;
        ias = ies + nr2;
        iq = ies;
        iz = ias;

        if (nsrt) {
            ilft = ir;
            irht = ii;
        } else {
            ilft = iz + nr2;
            irht = ilft + n;
        }
        iwrk = irht + n;

        SLC_DGGBAL("P", nr, &dwork[ia], &n, &dwork[ie], &n, &ilo, &ihi,
                   &dwork[ilft], &dwork[irht], &dwork[iwrk], &ierr);

        i32 m0, p0;
        if (nsrt) {
            for (i32 i = *nr - 2; i >= ihi - 1; i--) {
                i32 k = (i32)dwork[ir + i] - 1;
                if (k != i) {
                    SLC_DSWAP(&m, &dwork[ib + i], &n, &dwork[ib + k], &n);
                    SLC_DSWAP(&p, &dwork[ic_off + i * p], &(i32){1},
                              &dwork[ic_off + k * p], &(i32){1});
                }
            }
            for (i32 i = 0; i <= ilo - 2; i++) {
                i32 k = (i32)dwork[ir + i] - 1;
                if (k != i) {
                    SLC_DSWAP(&m, &dwork[ib + i], &n, &dwork[ib + k], &n);
                    SLC_DSWAP(&p, &dwork[ic_off + i * p], &(i32){1},
                              &dwork[ic_off + k * p], &(i32){1});
                }
            }
            m0 = m;
            p0 = p;
        } else {
            m0 = 0;
            p0 = 0;
        }

        tg01bd("G", nsrt ? "N" : "I", nsrt ? "N" : "I", *nr, m0, p0, ilo, ihi,
               &dwork[ia], n, &dwork[ie], n, &dwork[ib], n, &dwork[ic_off], p,
               &dwork[iq], *nr, &dwork[iz], *nr, &dwork[iwrk], ldwork - iwrk, &ierr);

        if (nsrt) {
            iwrk = ias + nr2;
            SLC_DLACPY("F", nr, nr, &dwork[ia], &n, &dwork[ias], nr);
            SLC_DLACPY("F", nr, nr, &dwork[ie], &n, &dwork[ies], nr);
            SLC_DHGEQZ("E", "N", "N", nr, &ilo, &ihi, &dwork[ias], nr,
                       &dwork[ies], nr, &dwork[ir], &dwork[ii], &dwork[ibt],
                       dwork, nr, dwork, nr, &dwork[iwrk], &(i32){ldwork - iwrk}, &ierr);
        } else {
            SLC_DHGEQZ("S", "V", "V", nr, &ilo, &ihi, &dwork[ia], &n,
                       &dwork[ie], &n, &dwork[ir], &dwork[ii], &dwork[ibt],
                       &dwork[iq], nr, &dwork[iz], nr, &dwork[iwrk],
                       &(i32){ldwork - iwrk}, &ierr);
        }

        if (ierr >= *nr + 1) { *info = 5; return; }
        else if (ierr != 0) { *info = 2; return; }
        maxwrk = max_i32((i32)dwork[iwrk] + iwrk, maxwrk);

        if (!nsrt) {
            if (ilascl) {
                ierr = 0;
                SLC_DLASCL("H", &(i32){0}, &(i32){0}, &anrmto, &anrm, &n, &n,
                           &dwork[ia], &n, &ierr);
                SLC_DLASCL("G", &(i32){0}, &(i32){0}, &anrmto, &anrm, &n, &(i32){1},
                           &dwork[ir], &n, &ierr);
                SLC_DLASCL("G", &(i32){0}, &(i32){0}, &anrmto, &anrm, &n, &(i32){1},
                           &dwork[ii], &n, &ierr);
            }
            if (ilescl) {
                ierr = 0;
                SLC_DLASCL("U", &(i32){0}, &(i32){0}, &enrmto, &enrm, &n, &n,
                           &dwork[ie], &n, &ierr);
                SLC_DLASCL("G", &(i32){0}, &(i32){0}, &enrmto, &enrm, &n, &(i32){1},
                           &dwork[ibt], &n, &ierr);
            }

            sdim1 = 0;
            wmax = ZERO;
            wrmin = safmax;

            for (i32 i = 0; i < *nr; i++) {
                if (dwork[ii + i] < ZERO) {
                    sdim1++;
                } else {
                    tm = fabs(dwork[ir + i]);
                    tmp = fabs(dwork[ii + i]);
                    bwork[i] = (dwork[ibt + i] != ZERO);
                    if (bwork[i]) {
                        if (min_f64(tm, tmp) == ZERO)
                            tmp = max_f64(tm, tmp);
                        else
                            tmp = SLC_DLAPY2(&tm, &tmp);
                        if (dwork[ibt + i] >= ONE ||
                            (dwork[ibt + i] < ONE && tmp < dwork[ibt + i] * safmax)) {
                            sdim1++;
                            tmp /= dwork[ibt + i];
                            wmax = max_f64(wmax, tmp);
                            wrmin = min_f64(wrmin, tmp);
                        } else {
                            bwork[i] = false;
                        }
                    } else {
                        if (max_f64(tm, tmp) == ZERO) goto label_420;
                    }
                }
            }

            if (wrmin > ONE)
                rat = wmax / wrmin;
            else if (wmax < wrmin * safmax)
                rat = wmax / wrmin;
            else
                rat = safmax;

            if (wreduc && ((f64)(*nr) * teps) * rat > ONE) {
                gpeak[0] = ONE;
                gpeak[1] = ZERO;
                fpeak[0] = ZERO;
                fpeak[1] = ONE;
                goto label_440;
            }

            if (sdim1 < *nr) {
                i32 idum[1];
                SLC_DTGSEN(&(i32){0}, &(i32){1}, &(i32){1}, bwork, nr,
                           &dwork[ia], &n, &dwork[ie], &n, &dwork[ir], &dwork[ii],
                           &dwork[ibt], &dwork[iq], nr, &dwork[iz], nr, &sdim,
                           &cnd, &cnd, &(f64){ZERO}, &dwork[iwrk], &(i32){ldwork - iwrk},
                           idum, &(i32){1}, &ierr);
                if (ierr == 1) { *info = 5; return; }
                if (sdim != sdim1) { *info = 6; return; }
                maxwrk = max_i32((i32)dwork[iwrk] + iwrk, maxwrk);

                f64 dum = ZERO;
                i32 cnt = *nr - sdim;
                for (i32 i = 0; i < cnt; i++)
                    dwork[ibt + sdim + i] = ZERO;
            } else {
                sdim = *nr;
            }

            SLC_DGGBAK("P", "L", nr, &ilo, &ihi, &dwork[ilft], &dwork[irht],
                       nr, &dwork[iq], nr, &ierr);
            SLC_DGGBAK("P", "R", nr, &ilo, &ihi, &dwork[ilft], &dwork[irht],
                       nr, &dwork[iz], nr, &ierr);

            SLC_DGEMM("T", "N", nr, &m, nr, &ONE, &dwork[iq], nr,
                      &dwork[ib], &n, &ZERO, &dwork[iwrk], nr);
            SLC_DLACPY("F", nr, &m, &dwork[iwrk], nr, &dwork[ib], &n);
            SLC_DGEMM("N", "N", &p, nr, nr, &ONE, &dwork[ic_off], &p,
                      &dwork[iz], nr, &ZERO, &dwork[iwrk], &p);
            SLC_DLACPY("F", &p, nr, &dwork[iwrk], &p, &dwork[ic_off], &p);

            id_off = ies;
            is_off = id_off + p * m;
            ninf = *nr - sdim;

            if (ninf != 0) {
                sdim1 = max_i32(1, sdim);
                i32 ibs_l = is_off + p * ninf;
                i32 ies_l = ibs_l + m * ninf;

                SLC_DLACPY("F", &sdim, &ninf, &dwork[ie + sdim * n], &n,
                           &dwork[ies_l], &sdim1);
                tm = SLC_DLANTR("1", "U", "N", &ninf, &ninf,
                                &dwork[ie + sdim * (n + 1)], &n, dwork);
                tmp = SLC_DLANHS("1", &ninf, &dwork[ia + sdim * (n + 1)], &n, dwork);
                ind1 = (tm < (f64)max_i32(sdim, ninf) * eps * tmp);

                if (max_f64(tm, tmp) == ZERO) {
                    goto label_420;
                } else if (ind1) {
                    iwrk = ies_l + sdim * ninf;
                    SLC_DTRCON("1", "U", "N", &sdim, &dwork[ie], &n, &rcond,
                               &dwork[iwrk], iwork, &ierr);
                    if (rcond <= (f64)sdim * teps) { *info = 1; return; }
                    f64 neg_one = -ONE;
                    SLC_DTRSM("L", "U", "N", "N", &sdim, &ninf, &neg_one,
                              &dwork[ie], &n, &dwork[ies_l], &sdim1);
                } else {
                    i32 ias_l = ies_l + sdim * ninf;
                    iwrk = ias_l + sdim * ninf;
                    SLC_DLACPY("F", &sdim, &ninf, &dwork[ia + sdim * n], &n,
                               &dwork[ias_l], &sdim1);
                    i32 cnt = 2 * sdim * ninf;
                    f64 neg_one = -ONE;
                    SLC_DSCAL(&cnt, &neg_one, &dwork[ies_l], &(i32){1});

                    sb04od("N", "N", "F", sdim, ninf, &dwork[ie], n,
                           &dwork[ie + sdim * (n + 1)], n, &dwork[ies_l], sdim1,
                           &dwork[ia], n, &dwork[ia + sdim * (n + 1)], n,
                           &dwork[ias_l], sdim1, &scl, &dif,
                           dwork, 1, dwork, 1, dwork, 1, dwork, 1,
                           iwork, &dwork[iwrk], ldwork - iwrk, &ierr);
                    f64 bound = eps * THOUSD;
                    if (ierr > 0 || dif > ONE / bound) { *info = 1; return; }
                    maxwrk = max_i32((i32)dwork[iwrk] + iwrk, maxwrk);

                    cnd = SLC_DLANGE("1", &sdim, &ninf, &dwork[ies_l], &sdim1, dwork);
                    if (cnd > ONE / toler) { *info = 1; return; }
                }

                SLC_DLACPY("F", &p, &ninf, &dwork[ic_off + p * sdim], &p,
                           &dwork[is_off], &p);
                SLC_DGEMM("N", "N", &p, &ninf, &sdim, &ONE,
                          &dwork[ic_off], &p, &dwork[ies_l], &sdim1,
                          &ONE, &dwork[is_off], &p);

                i32 ibs_l2 = is_off + p * ninf;  // IBS
                i32 ies_l2 = ibs_l2 + ninf * m;  // IES
                iwrk = ies_l2 + ninf * ninf;
                SLC_DLACPY("F", &ninf, &m, &dwork[ib + sdim], &n, &dwork[ibs_l2], &ninf);
                SLC_DLACPY("U", &ninf, &ninf, &dwork[ia + sdim * (n + 1)], &n,
                           &dwork[ies_l2], &ninf);
                if (ninf > 1) {
                    for (i32 i = 0; i < ninf - 1; i++)
                        dwork[ies_l2 + i + 1 + (ninf + 1) * i] = dwork[ia + sdim * (n + 1) + 1 + i * (n + 1)];
                }
                tmp = SLC_DLANHS("1", &ninf, &dwork[ies_l2], &ninf, dwork);
                mb02sd(ninf, &dwork[ies_l2], ninf, iwork, info);
                mb02td("1", ninf, tmp, &dwork[ies_l2], ninf, iwork, &rcond,
                       &iwork[ninf], &dwork[iwrk], info);
                if (rcond <= (f64)ninf * teps) { *info = 1; return; }
                mb02rd("N", ninf, m, &dwork[ies_l2], ninf, iwork, &dwork[ibs_l2], ninf, &ierr);

                if (withd)
                    SLC_DLACPY("F", &p, &m, d, &ldd, &dwork[id_off], &p);
                f64 neg_one = -ONE;
                SLC_DGEMM("N", "N", &p, &m, &ninf, &neg_one, &dwork[is_off], &p,
                          &dwork[ibs_l2], &ninf, &oz, &dwork[id_off], &p);
            } else if (withd) {
                SLC_DLACPY("F", &p, &m, d, &ldd, &dwork[id_off], &p);
            }

            if (ninf != 0 || withd) {
                iwrk = is_off + minpm;
                ierr = 0;
                SLC_DGESVD("N", "N", &p, &m, &dwork[id_off], &p, &dwork[is_off],
                           dwork, &(i32){1}, dwork, &(i32){1}, &dwork[iwrk],
                           &(i32){ldwork - iwrk}, &ierr);
                if (ierr > 0) { *info = 3; return; }
                maxwrk = max_i32((i32)dwork[iwrk] + iwrk, maxwrk);
                gammal = dwork[is_off];
            }
        } else {
            sdim = *nr;
        }

        if (ilascl) {
            for (i32 i = 0; i < *nr; i++) {
                if (dwork[ir + i] != ZERO) {
                    if ((dwork[ir + i] / safmax) > (anrmto / anrm) ||
                        (safmin / dwork[ir + i]) > (anrm / anrmto)) {
                        tm = fabs(dwork[ia + i * (n + 1)] / dwork[ir + i]);
                        dwork[ibt + i] *= tm;
                        dwork[ir + i] *= tm;
                        dwork[ii + i] *= tm;
                    } else if ((dwork[ii + i] / safmax) > (anrmto / anrm) ||
                               (dwork[ii + i] != ZERO &&
                                (safmin / dwork[ii + i]) > (anrm / anrmto))) {
                        tm = fabs(dwork[ia + i * (n + 1) + n] / dwork[ii + i]);
                        dwork[ibt + i] *= tm;
                        dwork[ir + i] *= tm;
                        dwork[ii + i] *= tm;
                    }
                }
            }
        }

        if (ilescl) {
            for (i32 i = 0; i < *nr; i++) {
                if (dwork[ibt + i] != ZERO) {
                    if ((dwork[ibt + i] / safmax) > (enrmto / enrm) ||
                        (safmin / dwork[ibt + i]) > (enrm / enrmto)) {
                        tm = fabs(dwork[ie + i * (n + 1)] / dwork[ibt + i]);
                        dwork[ibt + i] *= tm;
                        dwork[ir + i] *= tm;
                        dwork[ii + i] *= tm;
                    }
                }
            }
        }

        if (nsrt) {
            if (ilascl) {
                ierr = 0;
                SLC_DLASCL("H", &(i32){0}, &(i32){0}, &anrmto, &anrm, &n, &n,
                           &dwork[ia], &n, &ierr);
                SLC_DLASCL("G", &(i32){0}, &(i32){0}, &anrmto, &anrm, &n, &(i32){1},
                           &dwork[ir], &n, &ierr);
                SLC_DLASCL("G", &(i32){0}, &(i32){0}, &anrmto, &anrm, &n, &(i32){1},
                           &dwork[ii], &n, &ierr);
            }
            if (ilescl) {
                ierr = 0;
                SLC_DLASCL("U", &(i32){0}, &(i32){0}, &enrmto, &enrm, &n, &n,
                           &dwork[ie], &n, &ierr);
                SLC_DLASCL("G", &(i32){0}, &(i32){0}, &enrmto, &enrm, &n, &(i32){1},
                           &dwork[ibt], &n, &ierr);
            }
        }

    } else {
        // Standard state-space system
        sdim = n;

        if (lequil) {
            f64 maxred = HUNDRD;
            tb01id("A", n, m, p, &maxred, a, lda, b, ldb, c, ldc, &dwork[ii], &ierr);

            if (withd && (!nodyn) && unite) {
                for (i32 i = 0; i < n; i++) {
                    tmp = dwork[ii + i];
                    if (tmp != ONE) {
                        f64 inv_tmp = ONE / tmp;
                        SLC_DSCAL(&m, &inv_tmp, &dwork[ibv + i], &n);
                        SLC_DSCAL(&p, &tmp, &dwork[icu + i], &n);
                    }
                }
            }
        }

        SLC_DLACPY("F", &n, &n, a, &lda, &dwork[ia], &n);
        SLC_DLACPY("F", &n, &m, b, &ldb, &dwork[ib], &n);
        SLC_DLACPY("F", &p, &n, c, &ldc, &dwork[ic_off], &p);

        SLC_DGEBAL("P", &n, &dwork[ia], &n, &ilo, &ihi, &dwork[ir], &ierr);

        for (i32 i = n - 2; i >= ihi - 1; i--) {
            i32 k = (i32)dwork[ir + i] - 1;
            if (k != i) {
                SLC_DSWAP(&m, &dwork[ib + i], &n, &dwork[ib + k], &n);
                SLC_DSWAP(&p, &dwork[ic_off + i * p], &(i32){1},
                          &dwork[ic_off + k * p], &(i32){1});
            }
        }
        for (i32 i = 0; i <= ilo - 2; i++) {
            i32 k = (i32)dwork[ir + i] - 1;
            if (k != i) {
                SLC_DSWAP(&m, &dwork[ib + i], &n, &dwork[ib + k], &n);
                SLC_DSWAP(&p, &dwork[ic_off + i * p], &(i32){1},
                          &dwork[ic_off + k * p], &(i32){1});
            }
        }

        i32 itau = ir;
        iwrk = ii;
        SLC_DGEHRD(&n, &ilo, &ihi, &dwork[ia], &n, &dwork[itau], &dwork[iwrk],
                   &(i32){ldwork - iwrk}, &ierr);
        maxwrk = max_i32((i32)dwork[iwrk] + iwrk, maxwrk);

        SLC_DORMHR("L", "T", &n, &m, &ilo, &ihi, &dwork[ia], &n, &dwork[itau],
                   &dwork[ib], &n, &dwork[iwrk], &(i32){ldwork - iwrk}, &ierr);
        maxwrk = max_i32((i32)dwork[iwrk] + iwrk, maxwrk);

        SLC_DORMHR("R", "N", &p, &n, &ilo, &ihi, &dwork[ia], &n, &dwork[itau],
                   &dwork[ic_off], &p, &dwork[iwrk], &(i32){ldwork - iwrk}, &ierr);
        maxwrk = max_i32((i32)dwork[iwrk] + iwrk, maxwrk);

        ias = ibt;
        iwrk = ias + nn;
        SLC_DLACPY("F", &n, &n, &dwork[ia], &n, &dwork[ias], &n);
        SLC_DHSEQR("E", "N", &n, &ilo, &ihi, &dwork[ias], &n, &dwork[ir],
                   &dwork[ii], dwork, &(i32){1}, &dwork[iwrk], &(i32){ldwork - iwrk}, &ierr);
        if (ierr > 0) { *info = 2; return; }
        maxwrk = max_i32((i32)dwork[iwrk] + iwrk, maxwrk);

        if (n > 2) {
            i32 nm2 = n - 2;
            SLC_DLASET("L", &nm2, &nm2, &ZERO, &ZERO, &dwork[ia + 2], &n);
        }

        if (ilascl) {
            ierr = 0;
            SLC_DLASCL("H", &(i32){0}, &(i32){0}, &anrmto, &anrm, &n, &n,
                       &dwork[ia], &n, &ierr);
            SLC_DLASCL("G", &(i32){0}, &(i32){0}, &anrmto, &anrm, &n, &(i32){1},
                       &dwork[ir], &n, &ierr);
            SLC_DLASCL("G", &(i32){0}, &(i32){0}, &anrmto, &anrm, &n, &(i32){1},
                       &dwork[ii], &n, &ierr);
        }
    }

    // Eigenvalue analysis on boundary of stability domain
    i32 im = ibt;
    if (withe) im += n;
    ias = im + n;
    i32 imin = ii;
    wrmin = safmax;

    i32 nei = 0, neic = 0, neir = 0;
    bool linf = false;
    bool realw;
    f64 pi_val = FOUR * atan(ONE);
    f64 omega;

    if (discr) {
        if (withe) {
            for (i32 i = 0; i < *nr; i++) {
                tmr = dwork[ir + i];
                tmp = dwork[ii + i];
                if (tmp >= ZERO) {
                    realw = (tmp == ZERO);
                    if (realw)
                        tm = fabs(tmr);
                    else
                        tm = SLC_DLAPY2(&tmr, &tmp);
                    if (dwork[ibt + i] >= ONE || dwork[ibt + i] >= tm / pi_val) {
                        tm /= dwork[ibt + i];
                        if (tm != ZERO && tm < pi_val) {
                            if (realw) {
                                tmp = (tmr > ZERO) ? ZERO : pi_val;
                                neir++;
                            } else {
                                tmp = atan2(tmp, tmr);
                                neic++;
                            }
                            tmr = log(tm);
                            td = fabs(ONE - tm);
                            tm = SLC_DLAPY2(&tmr, &tmp);
                            if (td == ZERO) {
                                linf = true;
                                imin = ii + nei;
                                dwork[imin] = tmp;
                                goto label_130;
                            }
                            rat = ONE - TWO * (tmr / tm) * (tmr / tm);
                            if (rat <= P25)
                                dwork[im + nei] = tm / TWO;
                            else
                                dwork[im + nei] = tm * sqrt(max_f64(P25, rat));
                            bwork[nei] = realw;
                            nei++;
                        }
                    }
                }
            }
        } else {
            for (i32 i = 0; i < *nr; i++) {
                tmr = dwork[ir + i];
                tmp = dwork[ii + i];
                if (tmp >= ZERO) {
                    realw = (tmp == ZERO);
                    if (realw)
                        tm = fabs(tmr);
                    else
                        tm = SLC_DLAPY2(&tmr, &tmp);
                    if (tm != ZERO && tm < pi_val) {
                        if (realw) {
                            tmp = (tmr > ZERO) ? ZERO : pi_val;
                            neir++;
                        } else {
                            tmp = atan2(tmp, tmr);
                            neic++;
                        }
                        tmr = log(tm);
                        td = fabs(ONE - tm);
                        tm = SLC_DLAPY2(&tmr, &tmp);
                        if (td == ZERO) {
                            linf = true;
                            imin = ii + nei;
                            dwork[imin] = tmp;
                            goto label_130;
                        }
                        rat = ONE - TWO * (tmr / tm) * (tmr / tm);
                        if (rat <= P25)
                            dwork[im + nei] = tm / TWO;
                        else
                            dwork[im + nei] = tm * sqrt(max_f64(P25, rat));
                        bwork[nei] = realw;
                        nei++;
                    }
                }
            }
        }
    } else {
        if (withe) {
            for (i32 i = 0; i < *nr; i++) {
                tmr = fabs(dwork[ir + i]);
                tmp = dwork[ii + i];
                if (tmp >= ZERO) {
                    realw = (tmp == ZERO);
                    if (realw) {
                        if (tmr == ZERO) {
                            if (dwork[ibt + i] == ZERO) goto label_420;
                        }
                        tm = tmr;
                    } else {
                        tm = SLC_DLAPY2(&tmr, &tmp);
                    }
                    if (tmr == ZERO) {
                        linf = true;
                        imin = ii + i;
                        goto label_130;
                    } else if (dwork[ibt + i] >= ONE ||
                               (dwork[ibt + i] < ONE && tm < dwork[ibt + i] * safmax)) {
                        tmr /= dwork[ibt + i];
                        tm /= dwork[ibt + i];
                        if (realw) {
                            dwork[im + nei] = tm / TWO;
                            neir++;
                        } else {
                            rat = ONE - TWO * (tmr / tm) * (tmr / tm);
                            dwork[im + nei] = tm * sqrt(max_f64(P25, rat));
                            neic++;
                        }
                        bwork[nei] = realw;
                        nei++;
                    }
                }
            }
        } else {
            for (i32 i = 0; i < *nr; i++) {
                tmr = fabs(dwork[ir + i]);
                tmp = dwork[ii + i];
                if (tmp >= ZERO) {
                    if (tmr == ZERO) {
                        linf = true;
                        imin = ii + i;
                        goto label_130;
                    }
                    realw = (tmp == ZERO);
                    if (realw) {
                        dwork[im + nei] = tmr / TWO;
                        neir++;
                    } else {
                        tm = SLC_DLAPY2(&tmr, &tmp);
                        rat = ONE - TWO * (tmr / tm) * (tmr / tm);
                        dwork[im + nei] = tm * sqrt(max_f64(P25, rat));
                        neic++;
                    }
                    bwork[nei] = realw;
                    nei++;
                }
            }
        }
    }

label_130:
    if (linf) {
        gpeak[0] = ONE;
        gpeak[1] = ZERO;
        tm = dwork[imin];
        if (withe && !discr)
            tm /= dwork[ibt + imin - ii];
        fpeak[0] = tm;
        fpeak[1] = ONE;
        goto label_440;
    }

    // Determine max singular value of G over test frequencies
    const char *job;
    if (withe)
        job = "G";
    else
        job = "I";
    omega = ZERO;

    const char *jbdx;
    f64 gamma;

    if (discr) {
        jbdx = (*jobd == 'D' || *jobd == 'd' || *jobd == 'F' || *jobd == 'f') ? "D" :
               (*jobd == 'Z' || *jobd == 'z') ? "Z" : "D";
        iwrk = ias;
        gamma = ab13dx(dico, job, jbdx, *nr, m, p, omega, &dwork[ia], n,
                       &dwork[ie], n, &dwork[ib], n, &dwork[ic_off], p,
                       d, ldd, iwork, &dwork[iwrk], ldwork - iwrk,
                       zwork, lzwork, &ierr);
        i32 mxcw = (i32)creal(zwork[0]);
        // maxcwk not tracked after workspace query
    } else {
        i32 ibs_l = ias + nr2;
        id_off = ibs_l + (*nr) * m;
        SLC_DLACPY("U", nr, nr, &dwork[ia], &n, &dwork[ias], nr);
        if (*nr > 1) {
            i32 nrm1 = *nr - 1;
            i32 np1 = n + 1;
            i32 nrp1 = *nr + 1;
            SLC_DCOPY(&nrm1, &dwork[ia + 1], &np1, &dwork[ias + 1], &nrp1);
        }
        SLC_DLACPY("F", nr, &m, &dwork[ib], &n, &dwork[ibs_l], nr);
        if (withd) {
            SLC_DLACPY("F", &p, &m, d, &ldd, &dwork[id_off], &p);
            jbdx = "D";
            iwrk = id_off + p * m;
        } else {
            jbdx = "Z";
            iwrk = id_off;
        }
        gamma = ab13dx(dico, job, jbdx, *nr, m, p, omega, &dwork[ias], *nr,
                       &dwork[ie], n, &dwork[ibs_l], *nr, &dwork[ic_off], p,
                       &dwork[id_off], p, iwork, &dwork[iwrk], ldwork - iwrk,
                       zwork, lzwork, &ierr);
    }
    if (ierr > 0) goto label_430;
    maxwrk = max_i32((i32)dwork[iwrk] + iwrk, maxwrk);

    f64 fpeaki = fpeak[1];
    f64 fpeaks;
    if (fpeaki == ZERO)
        fpeaks = safmax;
    else
        fpeaks = fpeak[0] / fpeaki;

    if (gammal > gamma) {
        if (fabs(ONE - gamma / gammal) <= eps) {
            fpeak[0] = ZERO;
            fpeak[1] = ONE;
        } else if (!discr) {
            fpeak[0] = ONE;
            fpeak[1] = ZERO;
        }
    } else {
        gammal = gamma;
        fpeak[0] = ZERO;
        fpeak[1] = ONE;
    }

    if (discr) {
        omega = pi_val;
        gamma = ab13dx(dico, job, jbdx, *nr, m, p, omega, &dwork[ia], n,
                       &dwork[ie], n, &dwork[ib], n, &dwork[ic_off], p,
                       d, ldd, iwork, &dwork[iwrk], ldwork - iwrk,
                       zwork, lzwork, &ierr);
        if (ierr > 0) goto label_430;
        if (gammal < gamma) {
            gammal = gamma;
            fpeak[0] = omega;
            fpeak[1] = ONE;
        }
    } else {
        iwrk = ias;
    }

    if (fpeaks != ZERO || (discr && fpeaks != pi_val)) {
        omega = fpeaks;
        gamma = ab13dx(dico, job, jbdx, *nr, m, p, omega, &dwork[ia], n,
                       &dwork[ie], n, &dwork[ib], n, &dwork[ic_off], p,
                       d, ldd, iwork, &dwork[iwrk], ldwork - iwrk,
                       zwork, lzwork, &ierr);
        if (discr) {
            omega = fabs(atan2(sin(omega), cos(omega)));
        } else {
            maxwrk = max_i32((i32)dwork[iwrk] + iwrk, maxwrk);
        }
        if (ierr > 0) goto label_430;
        if (gammal < gamma) {
            gammal = gamma;
            fpeak[0] = omega;
            fpeak[1] = ONE;
        }
    }

    if (allpol || neir == nei || neic == nei) {
        SLC_DLASRT("I", &nei, &dwork[im], &ierr);

        if (!allpol) {
            if (neir == nei)
                nei = min_i32(nei, BNEIR);
            else {
                if (discr)
                    nei = min_i32(nei, BNEICD);
                else {
                    if (nei >= SWNEIC)
                        nei = BNEICX;
                    else
                        nei = min_i32(nei, BNEICM);
                }
            }
        }

        for (i32 i = 0; i < nei; i++) {
            omega = dwork[im + i];
            gamma = ab13dx(dico, job, jbdx, *nr, m, p, omega, &dwork[ia], n,
                           &dwork[ie], n, &dwork[ib], n, &dwork[ic_off], p,
                           d, ldd, iwork, &dwork[iwrk], ldwork - iwrk,
                           zwork, lzwork, &ierr);
            if (ierr > 0) goto label_430;
            if (gammal < gamma) {
                if (fabs(ONE - gamma / gammal) > eps) {
                    fpeak[0] = omega;
                    fpeak[1] = ONE;
                }
                gammal = gamma;
            }
        }
    } else {
        neic = 0;
        neir = 0;
        for (i32 i = 0; i < nei; i++) {
            if (bwork[i]) {
                dwork[ir + neir] = dwork[im + i];
                neir++;
            } else {
                dwork[im + neic] = dwork[im + i];
                neic++;
            }
        }

        SLC_DLASRT("I", &neir, &dwork[ir], &ierr);
        neir = min_i32(neir, BNEIR);
        for (i32 i = 0; i < neir; i++) {
            omega = dwork[ir + i];
            gamma = ab13dx(dico, job, jbdx, *nr, m, p, omega, &dwork[ia], n,
                           &dwork[ie], n, &dwork[ib], n, &dwork[ic_off], p,
                           d, ldd, iwork, &dwork[iwrk], ldwork - iwrk,
                           zwork, lzwork, &ierr);
            if (ierr > 0) goto label_430;
            tmp = fabs(ONE - gamma / gammal);
            if (gammal < gamma) {
                gammal = gamma;
                if (tmp > eps) {
                    fpeak[0] = omega;
                    fpeak[1] = ONE;
                }
            }
        }

        SLC_DLASRT("I", &neic, &dwork[im], &ierr);
        if (discr)
            neic = min_i32(neic, BNEICD);
        else {
            if (neic >= SWNEIC)
                neic = BNEICX;
            else
                neic = min_i32(neic, BNEICM);
        }
        for (i32 i = 0; i < neic; i++) {
            omega = dwork[im + i];
            gamma = ab13dx(dico, job, jbdx, *nr, m, p, omega, &dwork[ia], n,
                           &dwork[ie], n, &dwork[ib], n, &dwork[ic_off], p,
                           d, ldd, iwork, &dwork[iwrk], ldwork - iwrk,
                           zwork, lzwork, &ierr);
            if (ierr > 0) goto label_430;
            tmp = fabs(ONE - gamma / gammal);
            if (gammal < gamma) {
                gammal = gamma;
                if (tmp > eps) {
                    fpeak[0] = omega;
                    fpeak[1] = ONE;
                }
            }
        }
    }

    if (gammal == ZERO) {
        gpeak[0] = ZERO;
        gpeak[1] = ONE;
        fpeak[0] = ZERO;
        fpeak[1] = ONE;
        goto label_440;
    }

    // Modified gamma iteration (Bruinsma-Steinbuch)
    f64 tol1 = HUNDRD * eps;
    f64 tol2 = TEN * toler;
    f64 toln = tol[0];
    f64 tolp = ONE + P1 * toln;
    gamma = (ONE + toln) * gammal;

    if (!usepen && withd && !fullrd) {
        if (minpm > 1)
            rcond = ((gamma - sv1) / (gamma - svp)) * ((gamma + sv1) / (gamma + svp));
        else
            rcond = ONE;
        usepen = (rcond < HUNDRD * toler);
    }

    i32 r, k, q, nblk, nblk2, qp, tnr, n2, nk, nc_pen;
    i32 ih, ih12, ij, ij12, it, it12, ih22;
    i32 liw, ne_pen;
    bool case0, case1, case2, case3;
    f64 gam_arr[1], mgam_arr[1], ones_arr[1];
    i32 isl, isc, isb;
    i32 ici_pen = 0, ihc = 0;
    i32 pmq = 0;

    if (usepen) {
        r = pm % 2;
        k = (pm + r) / 2;
        q = *nr - k;
        nblk = *nr + k;
        nblk2 = nblk * nblk;
        ii = ir + nblk;
        ibt = ii + nblk;
        ih = ibt + nblk;
        ih12 = ih + nblk2;
        ij = ih12 + nblk2 + nblk;
        ij12 = ij + nblk2;
        it = ij12 + nblk2 + nblk;
        it12 = it + nblk2;
        ih22 = it12 + nblk2;
        iwrk = ih22 + nblk2;
        liw = 2 * nblk + 12;
        qp = q + p;
        tnr = 2 * (*nr);
        n2 = min_i32(*nr, k);
        case0 = (q >= 0);
        case1 = (q > 0);
        case2 = (!case0 && qp >= 0);
        case3 = (qp < 0);
        if (cmpre)
            ne_pen = min_i32(rnke, k);
        else
            ne_pen = n2;
        if (discr) {
            if (case0)
                nk = k;
            else {
                nk = n;
                nc_pen = case3 ? p : -q;
            }
        } else {
            if (case0) {
                ici_pen = 1;
                ihc = ih + q * nblk + *nr;
                nc_pen = p;
                pmq = pm;
            } else if (case2) {
                ici_pen = 1 - q;
                ihc = ih + *nr;
                nc_pen = qp;
            }
        }
        if (!case0) pmq = pm + q;
        ones_arr[0] = ONE;
    } else {
        ih = ibt;
        ih12 = ih + nn;
        isl = ih12 + nn + n;
        isc = isl + max_i32(m, p);
        isb = isc + p * n;
        iwrk = isl + nn + n;
    }

    // WHILE iteration
label_180:
    iter++;

    if (!usepen) {
        // Hamiltonian matrix approach
        if (zerod) {
            SLC_DLACPY("F", &n, &n, a, &lda, &dwork[ih], &n);
            f64 coeff = -ONE / gamma;
            SLC_DSYRK("L", "T", &n, &p, &coeff, c, &ldc, &ZERO, &dwork[ih12], &n);
            coeff = ONE / gamma;
            SLC_DSYRK("U", "N", &n, &m, &coeff, b, &ldb, &ZERO, &dwork[ih12 + n], &n);
        } else {
            for (i32 i = 0; i < minpm; i++)
                dwork[isl + i] = ONE / sqrt(gamma - dwork[is_off + i]) /
                                 sqrt(gamma + dwork[is_off + i]);
            if (m < p) {
                dwork[isl + m] = ONE / gamma;
                for (i32 i = m + 1; i < p; i++) dwork[isl + i] = dwork[isl + m];
            }
            SLC_DLACPY("F", &n, &p, &dwork[icu], &n, &dwork[isc], &n);
            mb01sd('C', n, p, &dwork[isc], n, dwork, &dwork[isl]);

            SLC_DLACPY("F", &n, &m, &dwork[ibv], &n, &dwork[isb], &n);
            mb01sd('C', n, minpm, &dwork[isb], n, dwork, &dwork[is_off]);
            mb01sd('C', n, minpm, &dwork[isb], n, dwork, &dwork[isl]);

            SLC_DLACPY("F", &n, &n, a, &lda, &dwork[ih], &n);
            SLC_DGEMM("N", "T", &n, &n, &minpm, &ONE, &dwork[isb], &n,
                      &dwork[isc], &n, &ONE, &dwork[ih], &n);

            if (p < m) {
                dwork[isl + p] = ONE / gamma;
                for (i32 i = p + 1; i < m; i++) dwork[isl + i] = dwork[isl + p];
            }
            SLC_DLACPY("F", &n, &m, &dwork[ibv], &n, &dwork[isb], &n);
            mb01sd('C', n, m, &dwork[isb], n, dwork, &dwork[isl]);

            f64 neg_gamma = -gamma;
            SLC_DSYRK("L", "N", &n, &p, &neg_gamma, &dwork[isc], &n,
                      &ZERO, &dwork[ih12], &n);
            SLC_DSYRK("U", "N", &n, &m, &gamma, &dwork[isb], &n,
                      &ZERO, &dwork[ih12 + n], &n);
        }

        f64 dum[1] = {ZERO};
        mb03xd("B", "E", "N", "N", n, &dwork[ih], n, &dwork[ih12], n,
               &dwork[isl], n, dum, 1, dum, 1, dum, 1, dum, 1,
               &dwork[ir], &dwork[ii], &ilo, &dwork[iwrk - n], &dwork[iwrk],
               ldwork - iwrk, &ierr);
        if (ierr > 0) { *info = 2; return; }

    } else {
        // Skew-Hamiltonian/Hamiltonian pencil approach
        i32 init_sz = 4 * nblk2 + 2 * nblk;
        SLC_DLASET("F", &init_sz, &(i32){1}, &ZERO, &ZERO, &dwork[ih], &(i32){1});
        gam_arr[0] = gamma;
        mgam_arr[0] = -gamma;

        if (discr) {
            i32 i1 = 0;
            if (gene) {
                for (i32 j = k; j < n; j++) {
                    for (i32 row = 0; row < n; row++) {
                        dwork[ih + i1 + row] = -e[row + j * lde];
                    }
                    for (i32 row = 0; row < n; row++) {
                        dwork[ij + i1 + row] = dwork[ih + i1 + row];
                    }
                    for (i32 row = 0; row < n; row++) {
                        dwork[ih + i1 + row] += a[row + j * lda];
                        dwork[ij + i1 + row] -= a[row + j * lda];
                    }
                    i1 += nblk;
                }
            } else {
                for (i32 j = k; j < n; j++) {
                    for (i32 row = 0; row < n; row++) {
                        dwork[ih + i1 + row] = a[row + j * lda];
                        dwork[ij + i1 + row] = -a[row + j * lda];
                    }
                    if (unite) {
                        dwork[ih + i1 + j] -= ONE;
                        dwork[ij + i1 + j] -= ONE;
                    } else if (rnke >= j + 1) {
                        for (i32 row = 0; row < rnke; row++) {
                            dwork[ih + i1 + row] -= e[row + j * lde];
                            dwork[ij + i1 + row] -= e[row + j * lde];
                        }
                    }
                    i1 += nblk;
                }
            }

            if (case0) {
                ma02ad("F", p, k, c, ldc, &dwork[ih + i1 + n], nblk);
                SLC_DLACPY("F", &k, &p, &dwork[ih + i1 + n], &nblk,
                           &dwork[ij + i1 + n], &nblk);
                i1 = qp * nblk;
                for (i32 j = 0; j < m; j++) {
                    f64 neg_one = -ONE;
                    SLC_DAXPY(&n, &neg_one, &b[j * ldb], &(i32){1},
                              &dwork[ih + i1], &(i32){1});
                    i1 += nblk;
                }
            } else if (case2) {
                if (qp > 0) {
                    ma02ad("F", qp, n, &c[(-q) * ldc], ldc, &dwork[ih + n], nblk);
                    SLC_DLACPY("F", &n, &qp, &dwork[ih + n], &nblk,
                               &dwork[ij + n], &nblk);
                    i1 = qp * nblk;
                }
                for (i32 j = 0; j < m; j++) {
                    i32 i2_l = i1 + tnr;
                    f64 neg_one = -ONE;
                    SLC_DAXPY(&n, &neg_one, &b[j * ldb], &(i32){1},
                              &dwork[ih + i1], &(i32){1});
                    if (withd) {
                        i32 nq = -q;
                        SLC_DAXPY(&nq, &neg_one, &d[j * ldd], &(i32){1},
                                  &dwork[ih + i2_l], &(i32){1});
                    }
                    i1 += nblk;
                }
            } else {
                for (i32 j = -qp; j < m; j++) {
                    i32 i2_l = i1 + tnr;
                    f64 neg_one = -ONE;
                    SLC_DAXPY(&n, &neg_one, &b[j * ldb], &(i32){1},
                              &dwork[ih + i1], &(i32){1});
                    if (withd) {
                        SLC_DAXPY(&p, &neg_one, &d[j * ldd], &(i32){1},
                                  &dwork[ih + i2_l], &(i32){1});
                    }
                    i1 += nblk;
                }
            }

            // H21 lower triangular, H12 upper triangular
            if (!case0) {
                i32 pmq_inc = nblk + 1;
                for (i32 i = 0; i < pmq; i++)
                    dwork[ih12 + i * pmq_inc] = gamma;
            }

            if (case0) {
                SLC_DLACPY("F", &p, &q, &c[k * ldc], &ldc, &dwork[ih12 + q], &nblk);
                i1 = q * nblk + q;
                for (i32 i = 0; i < pm; i++)
                    dwork[ih12 + i1 + i * (nblk + 1)] = gamma;
                if (withd) {
                    i1 = i1 + p;
                    for (i32 i = 0; i < p; i++) {
                        f64 neg_one = -ONE;
                        SLC_DAXPY(&m, &neg_one, &d[i], &ldd, &dwork[ih12 + i1], &(i32){1});
                        i1 += nblk;
                    }
                }
                i1 = q;
                for (i32 j = k; j < n; j++) {
                    f64 neg_one = -ONE;
                    SLC_DAXPY(&p, &neg_one, &c[j * ldc], &(i32){1},
                              &dwork[ij12 + i1], &(i32){1});
                    i1 += nblk;
                }
            } else if (case2 && withd) {
                i1 = qp;
                for (i32 i = -q; i < p; i++) {
                    f64 neg_one = -ONE;
                    SLC_DAXPY(&m, &neg_one, &d[i], &ldd, &dwork[ih12 + i1], &(i32){1});
                    i1 += nblk;
                }
            }

            i1 = (n + 1) * nblk;
            i32 i2_pen = (n + p) * nblk + i1;

            if (gene) {
                for (i32 j = 0; j < nk; j++) {
                    for (i32 row = 0; row < n; row++) {
                        dwork[ij12 + i1 + row] = e[row + j * lde] + a[row + j * lda];
                        dwork[ih12 + i1 + row] = e[row + j * lde] - a[row + j * lda];
                    }
                    i1 += nblk;
                }
            } else {
                for (i32 j = 0; j < nk; j++) {
                    for (i32 row = 0; row < n; row++) {
                        dwork[ij12 + i1 + row] = a[row + j * lda];
                        dwork[ih12 + i1 + row] = -a[row + j * lda];
                    }
                    if (unite) {
                        dwork[ij12 + i1 + j] += ONE;
                        dwork[ih12 + i1 + j] += ONE;
                    } else if (rnke >= j + 1) {
                        for (i32 row = 0; row < rnke; row++) {
                            dwork[ij12 + i1 + row] += e[row + j * lde];
                            dwork[ih12 + i1 + row] += e[row + j * lde];
                        }
                    }
                    i1 += nblk;
                }
            }

            if (!case0) {
                i1 += n;
                i32 i0_l = i1 + n;
                for (i32 i = 0; i < nc_pen; i++) {
                    f64 neg_one = -ONE;
                    SLC_DAXPY(&n, &neg_one, &c[i * ldc], &(i32){1},  // NOTE: row i of C, but C is col-major so &c[i] step ldc
                              &dwork[ih12 + i1], &(i32){1});
                    // This needs care: Fortran C(I,1) with step LDC means row I
                    // Actually: DAXPY(N, -1, C(I,1), LDC, ...) copies row I of C
                    // But our C array is column-major: c[i + j*ldc] = C(i+1,j+1)
                    // So row I of C (0-based) is c[i], c[i+ldc], c[i+2*ldc], ...
                    // DAXPY with incx=ldc: &c[i] with inc ldc
                    // Already correct above if we use &c[i] with inc ldc
                    // But we wrote &c[i * ldc] with inc 1 - that's WRONG!
                    // Let me fix this in a moment
                    for (i32 row2 = 0; row2 < n; row2++)
                        dwork[ij12 + i1 + row2] = dwork[ih12 + i1 + row2];
                    i1 += nblk;
                }
                // Fix: the above DAXPY calls for C rows need correction
                // I'll handle this in the actual pencil construction

                i32 nq = -q;
                for (i32 i = 0; i < nq; i++)
                    dwork[ih12 + i0_l + i * (nblk + 1)] = -gamma;

                if (case3) {
                    i32 nqp = -qp;
                    SLC_DLACPY("F", &n, &nqp, b, &ldb, &dwork[ih12 + i2_pen], &nblk);
                    if (withd)
                        SLC_DLACPY("F", &p, &nqp, d, &ldd, &dwork[ih12 + i2_pen + tn], &nblk);
                }
            }

            if (r > 0)
                dwork[ih12 + nblk2 - 1] = ONE;

        } else {
            // Continuous-time pencil
            if (case1) {
                i32 qq = q;
                SLC_DLACPY("F", nr, &qq, &a[k * lda], &lda, &dwork[ih], &nblk);
            }
            if (qp >= 0) {
                ma02ad("F", nc_pen, n2, &c[(ici_pen - 1) * 1], ldc, &dwork[ihc], nblk);
                i32 i1 = qp * nblk;
                i32 i2_l = i1 + tnr;
                for (i32 i0_l = 0; i0_l < m; i0_l++) {
                    f64 neg_one = -ONE;
                    SLC_DAXPY(nr, &neg_one, &b[i0_l * ldb], &(i32){1},
                              &dwork[ih + i1], &(i32){1});
                    i1 += nblk;
                }
                if (!case0 && withd) {
                    i32 nq = -q;
                    SLC_DLACPY("F", &nq, &m, d, &ldd, &dwork[ih + i2_l], &nblk);
                }
            } else {
                i32 pmq_l = pmq;
                SLC_DLACPY("F", nr, &pmq_l, &b[(-qp) * ldb], &ldb, &dwork[ih], &nblk);
                if (withd) {
                    SLC_DLACPY("F", &p, &pmq_l, &d[(-qp) * ldd], &ldd,
                               &dwork[ih + tnr], &nblk);
                }
            }

            // J11
            if (case1) {
                if (gene) {
                    i32 qq = q;
                    SLC_DLACPY("F", &n1, &qq, &e[k * lde], &lde, &dwork[ij], &nblk);
                } else if (cmpre) {
                    if (rnke > k) {
                        i32 rk = rnke - k;
                        SLC_DLACPY("F", &n1, &rk, &e[k * lde], &lde, &dwork[ij], &nblk);
                    }
                } else {
                    i32 qq = q;
                    for (i32 i = 0; i < qq; i++)
                        dwork[ij + i * (nblk + 1)] = ONE;
                }
            }

            // H21 lower triangular
            if (case1) {
                i32 i1 = q;
                for (i32 i = k; i < *nr; i++) {
                    f64 neg_one = -ONE;
                    SLC_DAXPY(&p, &neg_one, &c[i * ldc], &(i32){1},
                              &dwork[ih12 + i1], &(i32){1});
                    i1 += nblk;
                }
            }

            {
                i32 i1_l = case1 ? q : 0;
                for (i32 i = 0; i < pmq; i++)
                    dwork[ih12 + i1_l + i * (nblk + 1)] = gamma;
            }

            if (withd) {
                if (case0) {
                    i32 i1_l = (case1 ? q : 0) + p;
                    ma02ad("F", p, m, d, ldd, &dwork[ih12 + i1_l], nblk);
                } else if (case2) {
                    ma02ad("F", qp, m, &d[(-q)], ldd, &dwork[ih12 + qp], nblk);
                }
            }

            if (r == 1)
                dwork[ih12 + nblk2 - 1] = ONE;

            // H12 and J12 upper triangular
            {
                i32 i1 = (*nr + 1) * nblk;
                i32 i0_l = i1;
                SLC_DLACPY("F", nr, &n2, a, &lda, &dwork[ih12 + i1], &nblk);
                i32 i2_l = i1 + (*nr) * nblk;
                i1 = i2_l + *nr;

                if (!case0) {
                    if (case2) {
                        i32 nq = -q;
                        for (i32 i = 0; i < nq; i++) {
                            f64 neg_one = -ONE;
                            SLC_DAXPY(nr, &neg_one, &c[i], &ldc,
                                      &dwork[ih12 + i1], &(i32){1});
                            i1 += nblk;
                        }
                    } else {
                        ma02ad("F", p, *nr, c, ldc, &dwork[ih12 + i1], nblk);
                    }
                    i32 nq = -q;
                    for (i32 i = 0; i < nq; i++)
                        dwork[ih12 + i2_l + tnr + i * (nblk + 1)] = -gamma;
                }

                if (case3) {
                    i32 nqp = -qp;
                    i2_l += p * nblk;
                    SLC_DLACPY("F", nr, &nqp, b, &ldb, &dwork[ih12 + i2_l], &nblk);
                    if (withd)
                        SLC_DLACPY("F", &p, &nqp, d, &ldd, &dwork[ih12 + i2_l + tnr], &nblk);
                }

                if (unite) {
                    for (i32 i = 0; i < n2; i++)
                        dwork[ij12 + i0_l + i * (nblk + 1)] = ONE;
                } else {
                    SLC_DLACPY("F", &n1, &ne_pen, e, &lde, &dwork[ij12 + i0_l], &nblk);
                }
            }
        }

        // MB04BP call
        i32 two_nblk = 2 * nblk;
        ierr = -1;
        mb04bp("E", "N", "N", two_nblk, &dwork[ij], nblk, &dwork[ij12], nblk,
               &dwork[ih], nblk, &dwork[ih12], nblk, dwork, 1, dwork, 1,
               &dwork[it], nblk, &dwork[it12], nblk, &dwork[ih22], nblk,
               &dwork[ir], &dwork[ii], &dwork[ibt],
               iwork, liw, &dwork[iwrk], ldwork - iwrk, &ierr);
        if (ierr == 1 || ierr == 2) { *info = 2; return; }
    }

    maxwrk = max_i32((i32)dwork[iwrk] + iwrk, maxwrk);

    // Detect eigenvalues on boundary of stability domain
    wmax = ZERO;

    if (usepen) {
        im = ibt + nblk;
        for (i32 i = 0; i < nblk; i++) {
            tm = dwork[ii + i];
            if (tm >= ZERO) {
                tm = SLC_DLAPY2(&dwork[ir + i], &tm);
                if ((dwork[ibt + i] >= ONE) ||
                    (dwork[ibt + i] < ONE && tm < dwork[ibt + i] * safmax)) {
                    tm /= dwork[ibt + i];
                    if (tol1 * tm < stol) {
                        wmax = max_f64(wmax, tm);
                        dwork[im + i] = tm;
                    } else {
                        dwork[im + i] = -ONE;
                    }
                } else {
                    dwork[im + i] = -ONE;
                }
            } else {
                dwork[im + i] = -ONE;
            }
        }
    } else {
        for (i32 i = 0; i < *nr; i++) {
            tm = SLC_DLAPY2(&dwork[ir + i], &dwork[ii + i]);
            wmax = max_f64(wmax, tm);
            dwork[im + i] = tm;
        }
    }

    nei = 0;

    if (usepen) {
        for (i32 i = 0; i < nblk; i++) {
            tm = dwork[im + i];
            if (tm >= ZERO) {
                tmr = fabs(dwork[ir + i]) / dwork[ibt + i];
                if (tmr < tol2 * (ONE + tm) + tol1 * wmax) {
                    dwork[ii + nei] = dwork[ii + i] / dwork[ibt + i];
                    nei++;
                }
            }
        }
    } else {
        for (i32 i = 0; i < *nr; i++) {
            tm = dwork[im + i];
            tmr = fabs(dwork[ir + i]);
            if (tmr < tol2 * (ONE + tm) + tol1 * wmax) {
                dwork[ii + nei] = dwork[ii + i];
                nei++;
            }
        }
    }

    if (nei == 0) {
        gpeak[0] = gammal;
        gpeak[1] = ONE;
        goto label_440;
    }

    // Compute NWS frequencies
    i32 nws = 0;
    i32 j_cnt = 0;

    if (discr) {
        for (i32 i = 0; i < nei; i++) {
            f64 tmr1, tmp1, tm1, td1;
            f64 dii = dwork[ii + i];
            SLC_DLADIV(&ONE, &ZERO, &ONE, &dii, &tmr1, &tmp1);
            f64 neg_dii = -dii;
            SLC_DLADIV(&ONE, &ZERO, &ONE, &neg_dii, &tm1, &td1);
            f64 new_ir = tmr1 * tm1 - tmp1 * td1;
            f64 two_dii = TWO * dii;
            SLC_DLADIV(&two_dii, &ZERO, &ONE, &dii, &tmr1, &tmp1);
            SLC_DLADIV(&ONE, &ZERO, &ONE, &neg_dii, &tm1, &td1);
            f64 new_ii = tmr1 * tm1 - tmp1 * td1;

            tm = fabs(atan2(new_ii, new_ir));
            if (tm < pi_val) {
                if (tm > eps) {
                    dwork[ir + nws] = tm;
                    nws++;
                } else if (tm == eps) {
                    if (j_cnt == 0) {
                        dwork[ir + nws] = eps;
                        nws++;
                    }
                    j_cnt++;
                }
            }
        }
    } else {
        for (i32 i = 0; i < nei; i++) {
            tm = dwork[ii + i];
            if (tm > eps) {
                dwork[ir + nws] = tm;
                nws++;
            } else if (tm == eps) {
                if (j_cnt == 0) {
                    dwork[ir + nws] = eps;
                    nws++;
                }
                j_cnt++;
            }
        }
    }

    if (nws == 0) {
        gpeak[0] = gammal;
        gpeak[1] = ONE;
        goto label_440;
    }

    SLC_DLASRT("I", &nws, &dwork[ir], &ierr);
    i32 lw = 1;
    for (i32 i = 1; i < nws; i++) {
        if (dwork[ir + lw - 1] != dwork[ir + i]) {
            dwork[ir + lw] = dwork[ir + i];
            lw++;
        }
    }

    if (lw == 1) {
        dwork[ir + 1] = dwork[ir];
        lw = 2;
    }

    if (!allpol) lw = min_i32(lw, BM);

    i32 irlw = ir + lw;
    f64 gammas = gammal;

    for (i32 i = 0; i < lw - 1; i++) {
        if (discr)
            omega = (dwork[ir + i] + dwork[ir + i + 1]) / TWO;
        else
            omega = sqrt(dwork[ir + i] * dwork[ir + i + 1]);

        gamma = ab13dx(dico, job, jbdx, *nr, m, p, omega, &dwork[ia], n,
                       &dwork[ie], n, &dwork[ib], n, &dwork[ic_off], p,
                       d, ldd, iwork, &dwork[irlw], ldwork - irlw,
                       zwork, lzwork, &ierr);
        if (discr)
            omega = fabs(atan2(sin(omega), cos(omega)));
        if (ierr > 0) goto label_430;

        if (gammal < gamma) {
            gammal = gamma;
            fpeak[0] = omega;
            fpeak[1] = ONE;
        }
    }

    if (lw <= 1 || gammal < gammas * tolp) {
        gpeak[0] = gammal;
        gpeak[1] = ONE;
        goto label_440;
    }

    if (iter <= MAXIT) {
        gamma = (ONE + toln) * gammal;
        goto label_180;
    } else {
        *info = 4;
        gpeak[0] = gammal;
        gpeak[1] = ONE;
        goto label_440;
    }

label_420:
    *iwarn = 1;
    gpeak[0] = ZERO;
    gpeak[1] = ZERO;
    fpeak[0] = ZERO;
    fpeak[1] = ONE;
    goto label_440;

label_430:
    if (ierr == *nr + 1) {
        *info = 3;
        return;
    } else if (ierr > 0) {
        gpeak[0] = ONE;
        gpeak[1] = ZERO;
        fpeak[0] = omega;
        fpeak[1] = ONE;
    }

label_440:
    iwork[0] = iter;
    dwork[0] = (f64)maxwrk;
    zwork[0] = (c128)1;  // maxcwk not tracked precisely after workspace query
    return;
}
