/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 *
 * MB03BD - Compute eigenvalues of periodic Hessenberg matrix product
 *
 * Computes the eigenvalues of a generalized matrix product using
 * a double-shift version of the periodic QZ method.
 */

/* #define MB03BD_DEBUG 1 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <ctype.h>
#include <stdio.h>
#include <string.h>

void mb03bd(const char *job, const char *defl, const char *compq,
            const i32 *qind, i32 k, i32 n, i32 h, i32 ilo, i32 ihi,
            const i32 *s, f64 *a, i32 lda1, i32 lda2,
            f64 *q, i32 ldq1, i32 ldq2,
            f64 *alphar, f64 *alphai, f64 *beta, i32 *scal,
            i32 *iwork, i32 liwork, f64 *dwork, i32 ldwork, i32 *iwarn, i32 *info) {

    const i32 MCOUNT = 1;
    const i32 NITER = 10;
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TEN = 10.0;

    char job_upper = (char)toupper((unsigned char)job[0]);
    char defl_upper = (char)toupper((unsigned char)defl[0]);
    char compq_upper = (char)toupper((unsigned char)compq[0]);

    bool lsvd = (job_upper == 'T');
    bool lschr = (job_upper == 'S') || lsvd;
    bool liniq = (compq_upper == 'I');
    bool lcmpq = (compq_upper == 'U') || liniq;
    bool lparq = (compq_upper == 'P');
    bool adefl = (defl_upper == 'A');

    *iwarn = 0;
    i32 optdw = k + (2 * n > 8 * k ? 2 * n : 8 * k);
    i32 optiw = 2 * k + n;

    *info = 0;

    if (!(lschr || job_upper == 'E')) {
        *info = -1;
    } else if (!(adefl || defl_upper == 'C')) {
        *info = -2;
    } else if (!(lcmpq || lparq || compq_upper == 'N')) {
        *info = -3;
    } else if (k < 1) {
        *info = -5;
    } else if (n < 0) {
        *info = -6;
    } else if (h < 1 || h > k) {
        *info = -7;
    } else if (ilo < 1) {
        *info = -8;
    } else if (ihi > n || ihi < ilo - 1) {
        *info = -9;
    } else if (lda1 < (1 > n ? 1 : n)) {
        *info = -12;
    } else if (lda2 < (1 > n ? 1 : n)) {
        *info = -13;
    } else if (ldq1 < 1 || ((lcmpq || lparq) && ldq1 < n)) {
        *info = -15;
    } else if (ldq2 < 1 || ((lcmpq || lparq) && ldq2 < n)) {
        *info = -16;
    } else if (liwork < optiw) {
        iwork[0] = optiw;
        *info = -22;
    } else if (ldwork < optdw) {
        dwork[0] = (f64)optdw;
        *info = -24;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        dwork[0] = ONE;
        iwork[0] = 1;
        return;
    }

    i32 mapa = 0;
    i32 maph = 1;
    i32 mapq = k;
    i32 sinv;

    mb03ba(k, h, s, &sinv, &iwork[mapa], &iwork[mapq]);

    i32 ldas = lda1 * lda2;
    i32 ldqs = ldq1 * ldq2;
    i32 in_sz = ihi + 1 - ilo;

    f64 safmin, safmax, ulp, toll, smlnum, base, lgbas;
    {
        char param = 'S';
        safmin = SLC_DLAMCH(&param);
        safmax = ONE / safmin;
        param = 'P';
        ulp = SLC_DLAMCH(&param);
        toll = TEN * ulp;
        SLC_DLABAD(&safmin, &safmax);
        smlnum = safmin * ((f64)in_sz / ulp);
        param = 'B';
        base = SLC_DLAMCH(&param);
        lgbas = log(base);
    }

    f64 macpar[5];
    {
        char param = 'U';
        macpar[1] = SLC_DLAMCH(&param);
    }
    if (lsvd) {
        char param = 'O';
        macpar[0] = SLC_DLAMCH(&param);
        macpar[2] = safmin;
        param = 'E';
        macpar[3] = SLC_DLAMCH(&param);
        macpar[4] = base;
    }

    for (i32 i = 2 * k; i < 2 * k + n; i++) {
        iwork[i] = 0;
    }

    i32 pnorm = 0;
    i32 pfree = k;
    for (i32 i = 0; i < k; i++) {
        i32 aind = iwork[mapa + i] - 1;
        f64 *a_slice = a + aind * ldas;
        char norm = 'F';
        dwork[i] = SLC_DLANHS(&norm, &in_sz, &a_slice[(ilo - 1) + (ilo - 1) * lda1], &lda1, dwork);

        i32 j = 0;
        if (liniq) {
            j = i + 1;
        } else if (lparq) {
            j = -qind[i];
        }
        if (j > 0) {
            f64 *q_slice = q + (j - 1) * ldqs;
            SLC_DLASET("Full", &n, &n, &ZERO, &ONE, q_slice, &ldq1);
        }
    }

    for (i32 j = ihi; j < n; j++) {
        ma01bd(base, lgbas, k, s, &a[j + j * lda1], ldas, &alphar[j], &beta[j], &scal[j]);
        alphai[j] = ZERO;
    }

    if (ihi < ilo) {
        goto done;
    }

    i32 ilast = ihi;
    i32 ifrstm, ilastm;
    if (lschr) {
        ifrstm = 1;
        ilastm = n;
    } else {
        ifrstm = ilo;
        ilastm = ihi;
    }

    i32 iiter = 0;
    i32 titer = 0;
    i32 count = 0;
    i32 counte = 0;
    i32 maxit = 120 * in_sz;

    for (i32 jiter = 0; jiter < maxit; jiter++) {
#ifdef MB03BD_DEBUG
        fprintf(stderr, "MB03BD: jiter=%d, ilo=%d, ilast=%d, ifrstm=%d, ilastm=%d\n", jiter, ilo, ilast, ifrstm, ilastm);
        for (i32 dbg_l = 0; dbg_l < k; dbg_l++) {
            fprintf(stderr, "  factor %d (physical): subdiags: ", dbg_l);
            for (i32 dbg_j = 1; dbg_j < n; dbg_j++) {
                fprintf(stderr, "(%d,%d)=%.4e ", dbg_j, dbg_j-1, a[dbg_j + (dbg_j-1)*lda1 + dbg_l*ldas]);
            }
            fprintf(stderr, "\n");
        }
#endif
        if (ilast == ilo) {
            goto deflate_single;
        }

        i32 aind = iwork[mapa] - 1;
        f64 *a_h = a + aind * ldas;
        f64 tol;
        i32 jlo = ilo;

        if (adefl) {
            tol = fmax(safmin, dwork[pnorm] * ulp);
        }

        for (i32 j = ilast; j >= ilo + 1; j--) {
            if (!adefl) {
                tol = fabs(a_h[(j - 2) + (j - 2) * lda1]) + fabs(a_h[(j - 1) + (j - 1) * lda1]);
                if (tol == ZERO) {
                    char norm = '1';
                    i32 sz = j - ilo + 1;
                    tol = SLC_DLANHS(&norm, &sz, &a_h[(ilo - 1) + (ilo - 1) * lda1], &lda1, dwork);
                }
                tol = fmax(ulp * tol, smlnum);
            }
            if (fabs(a_h[(j - 1) + (j - 2) * lda1]) <= tol) {
                a_h[(j - 1) + (j - 2) * lda1] = ZERO;
                jlo = j;
                if (j == ilast) {
                    goto deflate_single;
                }
                break;
            }
        }

        // Test 2: Deflation in triangular matrices with index 1 (s[aind] == sinv)
        i32 ldef = 0;
        i32 jdef_val = 0;
        for (i32 l = 1; l < k; l++) {
            i32 aind_l = iwork[mapa + l] - 1;
            if (s[aind_l] == sinv) {
                if (adefl) {
                    tol = fmax(safmin, dwork[pnorm + l] * ulp);
                }
                for (i32 jt = ilast; jt >= jlo; jt--) {
                    if (!adefl) {
                        if (jt == ilast) {
                            tol = fabs(a[(jt - 2) + (jt - 1) * lda1 + aind_l * ldas]);
                        } else if (jt == jlo) {
                            tol = fabs(a[(jt - 1) + jt * lda1 + aind_l * ldas]);
                        } else {
                            tol = fabs(a[(jt - 2) + (jt - 1) * lda1 + aind_l * ldas]) +
                                  fabs(a[(jt - 1) + jt * lda1 + aind_l * ldas]);
                        }
                        if (tol == ZERO) {
                            char norm_c = '1';
                            i32 sz = jt - jlo + 1;
                            tol = SLC_DLANHS(&norm_c, &sz, &a[(jlo - 1) + (jlo - 1) * lda1 + aind_l * ldas], &lda1, dwork);
                        }
                        tol = fmax(ulp * tol, smlnum);
                    }
                    if (fabs(a[(jt - 1) + (jt - 1) * lda1 + aind_l * ldas]) <= tol) {
                        a[(jt - 1) + (jt - 1) * lda1 + aind_l * ldas] = ZERO;
                        ldef = l + 1;
                        jdef_val = jt;
                        goto case_ii;
                    }
                }
            }
        }

        // Test 3: Deflation in triangular matrices with index -1 (s[aind] != sinv)
        for (i32 l = 1; l < k; l++) {
            i32 aind_l = iwork[mapa + l] - 1;
            if (s[aind_l] != sinv) {
                if (adefl) {
                    tol = fmax(safmin, dwork[pnorm + l] * ulp);
                }
                for (i32 jt = ilast; jt >= jlo; jt--) {
                    if (!adefl) {
                        if (jt == ilast) {
                            tol = fabs(a[(jt - 2) + (jt - 1) * lda1 + aind_l * ldas]);
                        } else if (jt == jlo) {
                            tol = fabs(a[(jt - 1) + jt * lda1 + aind_l * ldas]);
                        } else {
                            tol = fabs(a[(jt - 2) + (jt - 1) * lda1 + aind_l * ldas]) +
                                  fabs(a[(jt - 1) + jt * lda1 + aind_l * ldas]);
                        }
                        if (tol == ZERO) {
                            char norm_c = '1';
                            i32 sz = jt - jlo + 1;
                            tol = SLC_DLANHS(&norm_c, &sz, &a[(jlo - 1) + (jlo - 1) * lda1 + aind_l * ldas], &lda1, dwork);
                        }
                        tol = fmax(ulp * tol, smlnum);
                    }
                    if (fabs(a[(jt - 1) + (jt - 1) * lda1 + aind_l * ldas]) <= tol) {
                        a[(jt - 1) + (jt - 1) * lda1 + aind_l * ldas] = ZERO;
                        ldef = l + 1;
                        jdef_val = jt;
                        goto case_iii;
                    }
                }
            }
        }

        i32 ifirst = jlo;
#ifdef MB03BD_DEBUG
        fprintf(stderr, "MB03BD: jlo=%d, ifirst=%d, active window=%d\n", jlo, ifirst, ilast - ifirst + 1);
#endif

        if (ilast - ifirst + 1 == 1) {
            goto deflate_single;
        }

        iiter++;

        if (!lschr) {
            ifrstm = ifirst;
        }

        if (iiter > NITER * 10) {
            goto failed;
        }

        f64 cs1, sn1, cs2, sn2;

        // Special case: 2x2 block
        if (ifirst + 1 == ilast) {
            i32 j = ilast - 1;
#ifdef MB03BD_DEBUG
            fprintf(stderr, "MB03BD: 2x2 block at ifirst=%d, ilast=%d, j=%d, titer=%d\n", ifirst, ilast, j, titer);
#endif

            if (titer < 2) {
                titer++;
                // Try to deflate the 2x2 problem
                // Copy 2x2 blocks to workspace in physical factor order
                for (i32 l = 0; l < k; l++) {
                    dwork[pfree + l * 4]     = a[(j - 1) + (j - 1) * lda1 + l * ldas];
                    dwork[pfree + l * 4 + 1] = a[j + (j - 1) * lda1 + l * ldas];
                    dwork[pfree + l * 4 + 2] = a[(j - 1) + j * lda1 + l * ldas];
                    dwork[pfree + l * 4 + 3] = a[j + j * lda1 + l * ldas];
                }
#ifdef MB03BD_DEBUG
                fprintf(stderr, "MB03BD: 2x2 block before mb03bf (k=%d, h=%d):\n", k, h);
                for (i32 l = 0; l < k; l++) {
                    fprintf(stderr, "  factor %d: [%.4e %.4e; %.4e %.4e]\n", l,
                            dwork[pfree + l*4], dwork[pfree + l*4 + 2],
                            dwork[pfree + l*4 + 1], dwork[pfree + l*4 + 3]);
                }
#endif

                // Call MB03BF to try deflation
                i32 saved_mapq = 0;
                if (sinv < 0) {
                    saved_mapq = iwork[mapq];
                    iwork[mapq] = iwork[mapa];
                }
                i32 two = 2;
                mb03bf(k, &iwork[maph], s, sinv, &dwork[pfree], two, two, ulp);
                if (sinv < 0) {
                    iwork[mapq] = saved_mapq;
                }
#ifdef MB03BD_DEBUG
                fprintf(stderr, "MB03BD: 2x2 block after mb03bf:\n");
                for (i32 l = 0; l < k; l++) {
                    fprintf(stderr, "  factor %d: [%.4e %.4e; %.4e %.4e]\n", l,
                            dwork[pfree + l*4], dwork[pfree + l*4 + 2],
                            dwork[pfree + l*4 + 1], dwork[pfree + l*4 + 3]);
                }
#endif

                // Check if deflation succeeded (subdiagonal of Hessenberg factor small)
                i32 idx = pfree + 4 * (h - 1);
                f64 maxval = fmax(fmax(fabs(dwork[idx]), fabs(dwork[idx + 2])), fabs(dwork[idx + 3]));
                if (fabs(dwork[idx + 1]) < ulp * maxval) {
#ifdef MB03BD_DEBUG
                    fprintf(stderr, "MB03BD: 2x2 real deflation succeeded, dwork[idx+1]=%.6e, ulp*maxval=%.6e\n",
                            dwork[idx + 1], ulp * maxval);
#endif
                    // Construct perfect shift polynomial

                    cs1 = ONE;
                    sn1 = ONE;
                    for (i32 l = k - 1; l >= 1; l--) {
                        i32 ai = iwork[mapa + l] - 1;
                        f64 temp = dwork[pfree + ai * 4 + 3];  // (2,2) element after deflation
                        f64 diag = a[(j - 1) + (j - 1) * lda1 + ai * ldas];
                        if (s[ai] == sinv) {
                            f64 r;
                            SLC_DLARTG(&(f64){cs1 * diag}, &(f64){sn1 * temp}, &cs1, &sn1, &r);
                        } else {
                            f64 r;
                            SLC_DLARTG(&(f64){cs1 * temp}, &(f64){sn1 * diag}, &cs1, &sn1, &r);
                        }
                    }
                    i32 ai = iwork[mapa] - 1;
                    f64 temp = dwork[pfree + ai * 4 + 3];
                    f64 diag = a[(j - 1) + (j - 1) * lda1 + ai * ldas];
                    f64 subdiag = a[j + (j - 1) * lda1 + ai * ldas];
                    f64 r;
                    SLC_DLARTG(&(f64){diag * cs1 - temp * sn1}, &(f64){subdiag * cs1}, &cs1, &sn1, &r);

                    // Do single-shift QZ step
                    goto single_shift_step;
                }
            }

            // Complex 2x2 block - do SVD if requested (JOB='T'), then compute eigenvalues
            if (lsvd) {
                mb03bc(k, &iwork[mapa], s, sinv, &a[(j - 1) + (j - 1) * lda1], lda1, lda2,
                       macpar, &dwork[pfree], &dwork[pfree + k], &dwork[pfree + 2 * k]);

                // Update factors and transformations
                i32 ai = iwork[mapa] - 1;
                cs2 = dwork[pfree];
                sn2 = dwork[pfree + k];
                i32 len = ilastm - ifrstm + 1;
                SLC_DROT(&len, &a[(ifrstm - 1) + (j - 1) * lda1 + ai * ldas], &(i32){1},
                         &a[(ifrstm - 1) + j * lda1 + ai * ldas], &(i32){1}, &cs2, &sn2);

                for (i32 l = 1; l < k; l++) {
                    ai = iwork[mapa + l] - 1;
                    i32 qi = -1;
                    if (lcmpq) {
                        qi = iwork[mapq + l] - 1;
                    } else if (lparq) {
                        qi = qind[iwork[mapq + l] - 1];
                        qi = (qi < 0 ? -qi : qi) - 1;
                    }
                    if (qi >= 0) {
                        f64 *q_slice = q + qi * ldqs;
                        SLC_DROT(&n, &q_slice[(j - 1) * ldq1], &(i32){1},
                                 &q_slice[j * ldq1], &(i32){1}, &cs2, &sn2);
                    }
                    cs1 = cs2;
                    sn1 = sn2;
                    cs2 = dwork[pfree + l];
                    sn2 = dwork[pfree + k + l];
                    if (s[ai] == sinv) {
                        len = ilastm - j - 1;
                        if (len > 0) {
                            SLC_DROT(&len, &a[(j - 1) + (j + 1) * lda1 + ai * ldas], &lda1,
                                     &a[j + (j + 1) * lda1 + ai * ldas], &lda1, &cs1, &sn1);
                        }
                        len = j - ifrstm;
                        if (len > 0) {
                            SLC_DROT(&len, &a[(ifrstm - 1) + (j - 1) * lda1 + ai * ldas], &(i32){1},
                                     &a[(ifrstm - 1) + j * lda1 + ai * ldas], &(i32){1}, &cs2, &sn2);
                        }
                    } else {
                        len = ilastm - j - 1;
                        if (len > 0) {
                            SLC_DROT(&len, &a[(j - 1) + (j + 1) * lda1 + ai * ldas], &lda1,
                                     &a[j + (j + 1) * lda1 + ai * ldas], &lda1, &cs2, &sn2);
                        }
                        len = j - ifrstm;
                        if (len > 0) {
                            SLC_DROT(&len, &a[(ifrstm - 1) + (j - 1) * lda1 + ai * ldas], &(i32){1},
                                     &a[(ifrstm - 1) + j * lda1 + ai * ldas], &(i32){1}, &cs1, &sn1);
                        }
                    }
                }
                i32 qi = -1;
                if (lcmpq) {
                    qi = iwork[mapq] - 1;
                } else if (lparq) {
                    qi = qind[iwork[mapq] - 1];
                    qi = (qi < 0 ? -qi : qi) - 1;
                }
                if (qi >= 0) {
                    f64 *q_slice = q + qi * ldqs;
                    SLC_DROT(&n, &q_slice[(j - 1) * ldq1], &(i32){1},
                             &q_slice[j * ldq1], &(i32){1}, &cs2, &sn2);
                }
                ai = iwork[mapa] - 1;
                len = ilastm - j + 1;
                SLC_DROT(&len, &a[(j - 1) + (j - 1) * lda1 + ai * ldas], &lda1,
                         &a[j + (j - 1) * lda1 + ai * ldas], &lda1, &cs2, &sn2);
            }

            // Compute complex eigenvalues
            f64 ar[2], ai_v[2], bt[2];
            i32 sc[2], ierr;
            mb03bb(base, lgbas, ulp, k, &iwork[mapa], s, sinv, &a[(j - 1) + (j - 1) * lda1],
                   lda1, lda2, ar, ai_v, bt, sc, &dwork[pfree], &ierr);
#ifdef MB03BD_DEBUG
            fprintf(stderr, "MB03BD: 2x2 mb03bb returned ar=[%.6e,%.6e], ai=[%.6e,%.6e], ierr=%d\n",
                    ar[0], ar[1], ai_v[0], ai_v[1], ierr);
#endif

            if (ierr == 1) {
                *iwarn = (j > *iwarn) ? j : *iwarn;
            } else if (ierr == 2) {
                if (*iwarn == 0) *iwarn = n;
            }

            // Store eigenvalues
            alphar[j - 1] = ar[0];
            alphai[j - 1] = ai_v[0];
            beta[j - 1] = bt[0];
            scal[j - 1] = sc[0];
            alphar[j] = ar[1];
            alphai[j] = ai_v[1];
            beta[j] = bt[1];
            scal[j] = sc[1];

            // Check for accuracy issues and zero/infinite eigenvalues
            for (i32 l = 0; l < k; l++) {
                i32 ai = iwork[mapa + l] - 1;
                if (ai_v[0] == ZERO && bt[0] != ZERO) {
                    if (fabs(a[(j - 1) + (j - 1) * lda1 + ai * ldas]) < dwork[pnorm + l] * toll) {
                        *iwarn = n + 1;
                        iwork[2 * k + j - 1] = -(j);
                        break;
                    }
                }
            }

            // Go to next block
            ilast = ifirst - 1;
            if (ilast < ilo) {
                goto converged;
            }
            iiter = 0;
            titer = 0;
            count = 0;
            counte = 0;
            if (!lschr) {
                ilastm = ilast;
                if (ifrstm > ilast) ifrstm = ilo;
            }
            continue;
        }

        // Normal case: block size >= 3
        if (count < NITER) {
            count++;
            if (sinv < 0) {
                i32 tmp = iwork[mapq];
                iwork[mapq] = iwork[mapa];
                mb03af("Double", k, ilast - ifirst + 1, &iwork[maph], s, sinv,
                       &a[(ifirst - 1) + (ifirst - 1) * lda1], lda1, lda2,
                       &cs1, &sn1, &cs2, &sn2);
                iwork[mapq] = tmp;
            } else {
                mb03af("Double", k, ilast - ifirst + 1, &iwork[maph], s, sinv,
                       &a[(ifirst - 1) + (ifirst - 1) * lda1], lda1, lda2,
                       &cs1, &sn1, &cs2, &sn2);
            }
        } else if (counte < MCOUNT) {
            f64 ar[2], ai_v[2], bt[2];
            i32 sc[2];
            i32 ierr;

            f64 *a_block = a + (ilast - 2) + (ilast - 2) * lda1;
            mb03bb(base, lgbas, ulp, k, &iwork[mapa], s, sinv, a_block, lda1, lda2,
                   ar, ai_v, bt, sc, &dwork[pfree], &ierr);

            if (ierr != 0) {
                if (sinv < 0) {
                    i32 tmp = iwork[mapq];
                    iwork[mapq] = iwork[mapa];
                    mb03af("Double", k, ilast - ifirst + 1, &iwork[maph], s, sinv,
                           &a[(ifirst - 1) + (ifirst - 1) * lda1], lda1, lda2,
                           &cs1, &sn1, &cs2, &sn2);
                    iwork[mapq] = tmp;
                } else {
                    mb03af("Double", k, ilast - ifirst + 1, &iwork[maph], s, sinv,
                           &a[(ifirst - 1) + (ifirst - 1) * lda1], lda1, lda2,
                           &cs1, &sn1, &cs2, &sn2);
                }
                count = 0;
                counte = 0;
            } else {
                counte++;
                char shft;
                f64 w1, w2;
                if (ai_v[0] != ZERO) {
                    shft = 'C';
                    w1 = ar[0];
                    w2 = ai_v[0];
                } else if (ar[0] == ar[1]) {
                    shft = 'D';
                    w1 = ar[0];
                    w2 = ar[1];
                } else {
                    shft = 'R';
                    w1 = ar[0];
                    w2 = ar[1];
                }
                mb03ab(&shft, k, ilast - ifirst + 1, &iwork[mapa], s, sinv,
                       &a[(ifirst - 1) + (ifirst - 1) * lda1], lda1, lda2,
                       w1, w2, &cs1, &sn1, &cs2, &sn2);
            }
        } else {
            count = 0;
            counte = 0;
            if (sinv < 0) {
                i32 tmp = iwork[mapq];
                iwork[mapq] = iwork[mapa];
                mb03af("Double", k, ilast - ifirst + 1, &iwork[maph], s, sinv,
                       &a[(ifirst - 1) + (ifirst - 1) * lda1], lda1, lda2,
                       &cs1, &sn1, &cs2, &sn2);
                iwork[mapq] = tmp;
            } else {
                mb03af("Double", k, ilast - ifirst + 1, &iwork[maph], s, sinv,
                       &a[(ifirst - 1) + (ifirst - 1) * lda1], lda1, lda2,
                       &cs1, &sn1, &cs2, &sn2);
            }
        }

        i32 j = ifirst;
        aind = iwork[mapa] - 1;
        a_h = a + aind * ldas;

        i32 in, io;
        if (k > 1) {
            in = ifirst + 1;
            io = ilast - 2;

            {
                i32 len = ilast - ifrstm + 1;
                SLC_DROT(&len, &a_h[(ifrstm - 1) + j * lda1], &(i32){1},
                         &a_h[(ifrstm - 1) + (j + 1) * lda1], &(i32){1}, &cs2, &sn2);
                SLC_DROT(&len, &a_h[(ifrstm - 1) + (j - 1) * lda1], &(i32){1},
                         &a_h[(ifrstm - 1) + j * lda1], &(i32){1}, &cs1, &sn1);
            }

            i32 qi;
            if (lcmpq) {
                qi = iwork[mapq + 1] - 1;
            } else if (lparq) {
                qi = qind[iwork[mapq + 1] - 1];
                qi = (qi < 0 ? -qi : qi) - 1;
            } else {
                qi = -1;
            }
            if (qi >= 0) {
                f64 *q_slice = q + qi * ldqs;
                SLC_DROT(&n, &q_slice[j * ldq1], &(i32){1},
                         &q_slice[(j + 1) * ldq1], &(i32){1}, &cs2, &sn2);
                SLC_DROT(&n, &q_slice[(j - 1) * ldq1], &(i32){1},
                         &q_slice[j * ldq1], &(i32){1}, &cs1, &sn1);
            }

            for (i32 l = 1; l < k; l++) {
                aind = iwork[mapa + l] - 1;
                f64 *a_l = a + aind * ldas;

                if (s[aind] == sinv) {
                    i32 len = ilastm - j + 1;
                    SLC_DROT(&len, &a_l[j + (j - 1) * lda1], &lda1,
                             &a_l[(j + 1) + (j - 1) * lda1], &lda1, &cs2, &sn2);

                    f64 temp = a_l[(j + 1) + (j + 1) * lda1];
                    f64 neg = -a_l[(j + 1) + j * lda1];
                    SLC_DLARTG(&temp, &neg, &cs2, &sn2, &a_l[(j + 1) + (j + 1) * lda1]);
                    a_l[(j + 1) + j * lda1] = ZERO;

                    len = j - ifrstm + 2;
                    SLC_DROT(&len, &a_l[(ifrstm - 1) + j * lda1], &(i32){1},
                             &a_l[(ifrstm - 1) + (j + 1) * lda1], &(i32){1}, &cs2, &sn2);

                    len = ilastm - j + 1;
                    SLC_DROT(&len, &a_l[(j - 1) + (j - 1) * lda1], &lda1,
                             &a_l[j + (j - 1) * lda1], &lda1, &cs1, &sn1);

                    temp = a_l[j + j * lda1];
                    neg = -a_l[j + (j - 1) * lda1];
                    SLC_DLARTG(&temp, &neg, &cs1, &sn1, &a_l[j + j * lda1]);
                    a_l[j + (j - 1) * lda1] = ZERO;

                    len = j - ifrstm + 1;
                    SLC_DROT(&len, &a_l[(ifrstm - 1) + (j - 1) * lda1], &(i32){1},
                             &a_l[(ifrstm - 1) + j * lda1], &(i32){1}, &cs1, &sn1);
                } else {
                    i32 len = j + 3 - ifrstm;
                    SLC_DROT(&len, &a_l[(ifrstm - 1) + j * lda1], &(i32){1},
                             &a_l[(ifrstm - 1) + (j + 1) * lda1], &(i32){1}, &cs2, &sn2);

                    f64 temp = a_l[j + j * lda1];
                    SLC_DLARTG(&temp, &a_l[(j + 1) + j * lda1], &cs2, &sn2, &a_l[j + j * lda1]);
                    a_l[(j + 1) + j * lda1] = ZERO;

                    len = ilastm - j - 1;
                    SLC_DROT(&len, &a_l[j + (j + 1) * lda1], &lda1,
                             &a_l[(j + 1) + (j + 1) * lda1], &lda1, &cs2, &sn2);

                    len = j + 2 - ifrstm;
                    SLC_DROT(&len, &a_l[(ifrstm - 1) + (j - 1) * lda1], &(i32){1},
                             &a_l[(ifrstm - 1) + j * lda1], &(i32){1}, &cs1, &sn1);

                    temp = a_l[(j - 1) + (j - 1) * lda1];
                    SLC_DLARTG(&temp, &a_l[j + (j - 1) * lda1], &cs1, &sn1, &a_l[(j - 1) + (j - 1) * lda1]);
                    a_l[j + (j - 1) * lda1] = ZERO;

                    len = ilastm - j;
                    SLC_DROT(&len, &a_l[(j - 1) + j * lda1], &lda1,
                             &a_l[j + j * lda1], &lda1, &cs1, &sn1);
                }

                if (lcmpq) {
                    qi = iwork[mapq + ((l + 1) % k)] - 1;
                } else if (lparq) {
                    qi = qind[iwork[mapq + ((l + 1) % k)] - 1];
                    qi = (qi < 0 ? -qi : qi) - 1;
                } else {
                    qi = -1;
                }
                if (qi >= 0) {
                    f64 *q_slice = q + qi * ldqs;
                    SLC_DROT(&n, &q_slice[j * ldq1], &(i32){1},
                             &q_slice[(j + 1) * ldq1], &(i32){1}, &cs2, &sn2);
                    SLC_DROT(&n, &q_slice[(j - 1) * ldq1], &(i32){1},
                             &q_slice[j * ldq1], &(i32){1}, &cs1, &sn1);
                }
            }

            aind = iwork[mapa] - 1;
            a_h = a + aind * ldas;
            {
                i32 len = ilastm - ifirst + 1;
                SLC_DROT(&len, &a_h[j + (ifirst - 1) * lda1], &lda1,
                         &a_h[(j + 1) + (ifirst - 1) * lda1], &lda1, &cs2, &sn2);
                SLC_DROT(&len, &a_h[(j - 1) + (ifirst - 1) * lda1], &lda1,
                         &a_h[j + (ifirst - 1) * lda1], &lda1, &cs1, &sn1);
            }
        } else {
            in = ifirst - 1;
            io = ilast - 3;
        }

        i32 qi;
        for (i32 j1 = in; j1 <= io; j1++) {
            aind = iwork[mapa] - 1;
            a_h = a + aind * ldas;
            if (lcmpq) {
                qi = iwork[mapq] - 1;
            } else if (lparq) {
                qi = qind[iwork[mapq] - 1];
                qi = (qi < 0 ? -qi : qi) - 1;
            } else {
                qi = -1;
            }

            if (j1 < ifirst) {
                j = j1 + 1;
                i32 len = ilastm - j + 1;
                SLC_DROT(&len, &a_h[j + (j - 1) * lda1], &lda1,
                         &a_h[(j + 1) + (j - 1) * lda1], &lda1, &cs2, &sn2);
                SLC_DROT(&len, &a_h[(j - 1) + (j - 1) * lda1], &lda1,
                         &a_h[j + (j - 1) * lda1], &lda1, &cs1, &sn1);
            } else {
                if (k == 1) {
                    j = j + 1;
                } else {
                    j = j1;
                }
                f64 temp = a_h[j + (j - 2) * lda1];
                f64 temp2;
                SLC_DLARTG(&temp, &a_h[(j + 1) + (j - 2) * lda1], &cs2, &sn2, &temp2);
                temp = a_h[(j - 1) + (j - 2) * lda1];
                SLC_DLARTG(&temp, &temp2, &cs1, &sn1, &a_h[(j - 1) + (j - 2) * lda1]);
                a_h[j + (j - 2) * lda1] = ZERO;
                a_h[(j + 1) + (j - 2) * lda1] = ZERO;
                i32 len = ilastm - j + 1;
                SLC_DROT(&len, &a_h[j + (j - 1) * lda1], &lda1,
                         &a_h[(j + 1) + (j - 1) * lda1], &lda1, &cs2, &sn2);
                SLC_DROT(&len, &a_h[(j - 1) + (j - 1) * lda1], &lda1,
                         &a_h[j + (j - 1) * lda1], &lda1, &cs1, &sn1);
            }
            if (qi >= 0) {
                f64 *q_slice = q + qi * ldqs;
                SLC_DROT(&n, &q_slice[j * ldq1], &(i32){1},
                         &q_slice[(j + 1) * ldq1], &(i32){1}, &cs2, &sn2);
                SLC_DROT(&n, &q_slice[(j - 1) * ldq1], &(i32){1},
                         &q_slice[j * ldq1], &(i32){1}, &cs1, &sn1);
            }

            for (i32 l = k - 1; l >= 1; l--) {
                aind = iwork[mapa + l] - 1;
                f64 *a_l = a + aind * ldas;

                if (s[aind] == sinv) {
                    i32 len = j + 3 - ifrstm;
                    SLC_DROT(&len, &a_l[(ifrstm - 1) + j * lda1], &(i32){1},
                             &a_l[(ifrstm - 1) + (j + 1) * lda1], &(i32){1}, &cs2, &sn2);

                    f64 temp = a_l[j + j * lda1];
                    SLC_DLARTG(&temp, &a_l[(j + 1) + j * lda1], &cs2, &sn2, &a_l[j + j * lda1]);
                    a_l[(j + 1) + j * lda1] = ZERO;

                    len = ilastm - j - 1;
                    SLC_DROT(&len, &a_l[j + (j + 1) * lda1], &lda1,
                             &a_l[(j + 1) + (j + 1) * lda1], &lda1, &cs2, &sn2);

                    len = j + 2 - ifrstm;
                    SLC_DROT(&len, &a_l[(ifrstm - 1) + (j - 1) * lda1], &(i32){1},
                             &a_l[(ifrstm - 1) + j * lda1], &(i32){1}, &cs1, &sn1);

                    temp = a_l[(j - 1) + (j - 1) * lda1];
                    SLC_DLARTG(&temp, &a_l[j + (j - 1) * lda1], &cs1, &sn1, &a_l[(j - 1) + (j - 1) * lda1]);
                    a_l[j + (j - 1) * lda1] = ZERO;

                    len = ilastm - j;
                    SLC_DROT(&len, &a_l[(j - 1) + j * lda1], &lda1,
                             &a_l[j + j * lda1], &lda1, &cs1, &sn1);
                } else {
                    i32 len = ilastm - j + 1;
                    SLC_DROT(&len, &a_l[j + (j - 1) * lda1], &lda1,
                             &a_l[(j + 1) + (j - 1) * lda1], &lda1, &cs2, &sn2);

                    f64 temp = a_l[(j + 1) + (j + 1) * lda1];
                    f64 neg = -a_l[(j + 1) + j * lda1];
                    SLC_DLARTG(&temp, &neg, &cs2, &sn2, &a_l[(j + 1) + (j + 1) * lda1]);
                    a_l[(j + 1) + j * lda1] = ZERO;

                    len = j + 2 - ifrstm;
                    SLC_DROT(&len, &a_l[(ifrstm - 1) + j * lda1], &(i32){1},
                             &a_l[(ifrstm - 1) + (j + 1) * lda1], &(i32){1}, &cs2, &sn2);

                    len = ilastm - j + 1;
                    SLC_DROT(&len, &a_l[(j - 1) + (j - 1) * lda1], &lda1,
                             &a_l[j + (j - 1) * lda1], &lda1, &cs1, &sn1);

                    temp = a_l[j + j * lda1];
                    neg = -a_l[j + (j - 1) * lda1];
                    SLC_DLARTG(&temp, &neg, &cs1, &sn1, &a_l[j + j * lda1]);
                    a_l[j + (j - 1) * lda1] = ZERO;

                    len = j + 1 - ifrstm;
                    SLC_DROT(&len, &a_l[(ifrstm - 1) + (j - 1) * lda1], &(i32){1},
                             &a_l[(ifrstm - 1) + j * lda1], &(i32){1}, &cs1, &sn1);
                }

                if (lcmpq) {
                    qi = iwork[mapq + l] - 1;
                } else if (lparq) {
                    qi = qind[iwork[mapq + l] - 1];
                    qi = (qi < 0 ? -qi : qi) - 1;
                } else {
                    qi = -1;
                }
                if (qi >= 0) {
                    f64 *q_slice = q + qi * ldqs;
                    SLC_DROT(&n, &q_slice[j * ldq1], &(i32){1},
                             &q_slice[(j + 1) * ldq1], &(i32){1}, &cs2, &sn2);
                    SLC_DROT(&n, &q_slice[(j - 1) * ldq1], &(i32){1},
                             &q_slice[j * ldq1], &(i32){1}, &cs1, &sn1);
                }
            }

            aind = iwork[mapa] - 1;
            a_h = a + aind * ldas;
            i32 lm = (j + 3 < ilastm ? j + 3 : ilastm) - ifrstm + 1;
            SLC_DROT(&lm, &a_h[(ifrstm - 1) + j * lda1], &(i32){1},
                     &a_h[(ifrstm - 1) + (j + 1) * lda1], &(i32){1}, &cs2, &sn2);
            SLC_DROT(&lm, &a_h[(ifrstm - 1) + (j - 1) * lda1], &(i32){1},
                     &a_h[(ifrstm - 1) + j * lda1], &(i32){1}, &cs1, &sn1);
        }

        j = ilast - 1;
        {
            f64 temp = a_h[(j - 1) + (j - 2) * lda1];
            SLC_DLARTG(&temp, &a_h[j + (j - 2) * lda1], &cs1, &sn1, &a_h[(j - 1) + (j - 2) * lda1]);
            a_h[j + (j - 2) * lda1] = ZERO;
        }

        aind = iwork[mapa] - 1;
        a_h = a + aind * ldas;
        {
            i32 len = ilastm - j + 1;
            SLC_DROT(&len, &a_h[(j - 1) + (j - 1) * lda1], &lda1,
                     &a_h[j + (j - 1) * lda1], &lda1, &cs1, &sn1);
        }
        if (lcmpq) {
            qi = iwork[mapq] - 1;
        } else if (lparq) {
            qi = qind[iwork[mapq] - 1];
            qi = (qi < 0 ? -qi : qi) - 1;
        } else {
            qi = -1;
        }
        if (qi >= 0) {
            f64 *q_slice = q + qi * ldqs;
            SLC_DROT(&n, &q_slice[(j - 1) * ldq1], &(i32){1},
                     &q_slice[j * ldq1], &(i32){1}, &cs1, &sn1);
        }

        for (i32 l = k - 1; l >= 1; l--) {
            aind = iwork[mapa + l] - 1;
            f64 *a_l = a + aind * ldas;

            if (s[aind] == sinv) {
                i32 len = j + 2 - ifrstm;
                SLC_DROT(&len, &a_l[(ifrstm - 1) + (j - 1) * lda1], &(i32){1},
                         &a_l[(ifrstm - 1) + j * lda1], &(i32){1}, &cs1, &sn1);

                f64 temp = a_l[(j - 1) + (j - 1) * lda1];
                SLC_DLARTG(&temp, &a_l[j + (j - 1) * lda1], &cs1, &sn1, &a_l[(j - 1) + (j - 1) * lda1]);
                a_l[j + (j - 1) * lda1] = ZERO;

                i32 len2 = ilastm - j;
                SLC_DROT(&len2, &a_l[(j - 1) + j * lda1], &lda1,
                         &a_l[j + j * lda1], &lda1, &cs1, &sn1);
            } else {
                i32 len = ilastm - j + 1;
                SLC_DROT(&len, &a_l[(j - 1) + (j - 1) * lda1], &lda1,
                         &a_l[j + (j - 1) * lda1], &lda1, &cs1, &sn1);

                f64 temp = a_l[j + j * lda1];
                f64 neg = -a_l[j + (j - 1) * lda1];
                SLC_DLARTG(&temp, &neg, &cs1, &sn1, &a_l[j + j * lda1]);
                a_l[j + (j - 1) * lda1] = ZERO;

                i32 len2 = j + 1 - ifrstm;
                SLC_DROT(&len2, &a_l[(ifrstm - 1) + (j - 1) * lda1], &(i32){1},
                         &a_l[(ifrstm - 1) + j * lda1], &(i32){1}, &cs1, &sn1);
            }

            if (lcmpq) {
                qi = iwork[mapq + l] - 1;
            } else if (lparq) {
                qi = qind[iwork[mapq + l] - 1];
                qi = (qi < 0 ? -qi : qi) - 1;
            } else {
                qi = -1;
            }
            if (qi >= 0) {
                f64 *q_slice = q + qi * ldqs;
                SLC_DROT(&n, &q_slice[(j - 1) * ldq1], &(i32){1},
                         &q_slice[j * ldq1], &(i32){1}, &cs1, &sn1);
            }
        }
        aind = iwork[mapa] - 1;
        {
            i32 len = ilastm - ifrstm + 1;
            SLC_DROT(&len, &a[(ifrstm - 1) + (j - 1) * lda1 + aind * ldas], &(i32){1},
                     &a[(ifrstm - 1) + j * lda1 + aind * ldas], &(i32){1}, &cs1, &sn1);
        }
        continue;

    single_shift_step:
        // Single-shift QZ step for 2x2 block
        {
            i32 j = ilast - 1;
            aind = iwork[mapa] - 1;
            a_h = a + aind * ldas;

            i32 len = ilastm - j + 1;
            SLC_DROT(&len, &a_h[(j - 1) + (j - 1) * lda1], &lda1,
                     &a_h[j + (j - 1) * lda1], &lda1, &cs1, &sn1);

            i32 qi = -1;
            if (lcmpq) {
                qi = iwork[mapq] - 1;
            } else if (lparq) {
                qi = qind[iwork[mapq] - 1];
                qi = (qi < 0 ? -qi : qi) - 1;
            }
            if (qi >= 0) {
                f64 *q_slice = q + qi * ldqs;
                SLC_DROT(&n, &q_slice[(j - 1) * ldq1], &(i32){1},
                         &q_slice[j * ldq1], &(i32){1}, &cs1, &sn1);
            }

            // Propagate through triangular factors
            for (i32 l = k - 1; l >= 1; l--) {
                i32 ai = iwork[mapa + l] - 1;
                f64 *a_l = a + ai * ldas;

                if (s[ai] == sinv) {
                    len = j + 2 - ifrstm;
                    SLC_DROT(&len, &a_l[(ifrstm - 1) + (j - 1) * lda1], &(i32){1},
                             &a_l[(ifrstm - 1) + j * lda1], &(i32){1}, &cs1, &sn1);

                    f64 temp = a_l[(j - 1) + (j - 1) * lda1];
                    SLC_DLARTG(&temp, &a_l[j + (j - 1) * lda1], &cs1, &sn1, &a_l[(j - 1) + (j - 1) * lda1]);
                    a_l[j + (j - 1) * lda1] = ZERO;

                    len = ilastm - j;
                    if (len > 0) {
                        SLC_DROT(&len, &a_l[(j - 1) + j * lda1], &lda1,
                                 &a_l[j + j * lda1], &lda1, &cs1, &sn1);
                    }
                } else {
                    len = ilastm - j + 1;
                    SLC_DROT(&len, &a_l[(j - 1) + (j - 1) * lda1], &lda1,
                             &a_l[j + (j - 1) * lda1], &lda1, &cs1, &sn1);

                    f64 temp = a_l[j + j * lda1];
                    f64 neg = -a_l[j + (j - 1) * lda1];
                    SLC_DLARTG(&temp, &neg, &cs1, &sn1, &a_l[j + j * lda1]);
                    a_l[j + (j - 1) * lda1] = ZERO;

                    len = j + 1 - ifrstm;
                    SLC_DROT(&len, &a_l[(ifrstm - 1) + (j - 1) * lda1], &(i32){1},
                             &a_l[(ifrstm - 1) + j * lda1], &(i32){1}, &cs1, &sn1);
                }

                if (lcmpq) {
                    qi = iwork[mapq + l] - 1;
                } else if (lparq) {
                    qi = qind[iwork[mapq + l] - 1];
                    qi = (qi < 0 ? -qi : qi) - 1;
                } else {
                    qi = -1;
                }
                if (qi >= 0) {
                    f64 *q_slice = q + qi * ldqs;
                    SLC_DROT(&n, &q_slice[(j - 1) * ldq1], &(i32){1},
                             &q_slice[j * ldq1], &(i32){1}, &cs1, &sn1);
                }
            }

            aind = iwork[mapa] - 1;
            len = ilastm - ifrstm + 1;
            SLC_DROT(&len, &a[(ifrstm - 1) + (j - 1) * lda1 + aind * ldas], &(i32){1},
                     &a[(ifrstm - 1) + j * lda1 + aind * ldas], &(i32){1}, &cs1, &sn1);
        }
        continue;

    case_ii:
        // Case II: Deflation in triangular matrix with index 1 (s[aind] == sinv)
        // Do unshifted periodic QZ step
        {
            i32 jdef = ilast;
            for (i32 jt = ilast; jt >= jlo; jt--) {
                i32 aind_l = iwork[mapa + ldef - 1] - 1;
                if (fabs(a[(jt - 1) + (jt - 1) * lda1 + aind_l * ldas]) <= tol) {
                    jdef = jt;
                    break;
                }
            }

            aind = iwork[mapa] - 1;
            i32 pdw = pfree;
            for (i32 jt = jlo; jt <= jdef - 1; jt++) {
                f64 temp = a[(jt - 1) + (jt - 1) * lda1 + aind * ldas];
                f64 cs_t, sn_t;
                SLC_DLARTG(&temp, &a[jt + (jt - 1) * lda1 + aind * ldas], &cs_t, &sn_t, &a[(jt - 1) + (jt - 1) * lda1 + aind * ldas]);
                a[jt + (jt - 1) * lda1 + aind * ldas] = ZERO;
                i32 len = ilastm - jt;
                SLC_DROT(&len, &a[(jt - 1) + jt * lda1 + aind * ldas], &lda1,
                         &a[jt + jt * lda1 + aind * ldas], &lda1, &cs_t, &sn_t);
                dwork[pdw] = cs_t;
                dwork[pdw + 1] = sn_t;
                pdw += 2;
            }
            i32 qi = -1;
            if (lcmpq) {
                qi = iwork[mapq] - 1;
            } else if (lparq) {
                qi = qind[iwork[mapq] - 1];
                qi = (qi < 0 ? -qi : qi) - 1;
            }
            if (qi >= 0) {
                pdw = pfree;
                for (i32 jt = jlo; jt <= jdef - 1; jt++) {
                    f64 cs_t = dwork[pdw];
                    f64 sn_t = dwork[pdw + 1];
                    pdw += 2;
                    f64 *q_slice = q + qi * ldqs;
                    SLC_DROT(&n, &q_slice[(jt - 1) * ldq1], &(i32){1},
                             &q_slice[jt * ldq1], &(i32){1}, &cs_t, &sn_t);
                }
            }

            // Propagate through triangular factors
            for (i32 l = k - 1; l >= 1; l--) {
                aind = iwork[mapa + l] - 1;
                i32 ntra = (l + 1 < ldef) ? jdef - 2 : jdef - 1;
                pdw = pfree;
                if (s[aind] == sinv) {
                    for (i32 jt = jlo; jt <= ntra; jt++) {
                        f64 cs_t = dwork[pdw];
                        f64 sn_t = dwork[pdw + 1];
                        i32 len = jt + 2 - ifrstm;
                        SLC_DROT(&len, &a[(ifrstm - 1) + (jt - 1) * lda1 + aind * ldas], &(i32){1},
                                 &a[(ifrstm - 1) + jt * lda1 + aind * ldas], &(i32){1}, &cs_t, &sn_t);
                        f64 temp = a[(jt - 1) + (jt - 1) * lda1 + aind * ldas];
                        SLC_DLARTG(&temp, &a[jt + (jt - 1) * lda1 + aind * ldas], &cs_t, &sn_t, &a[(jt - 1) + (jt - 1) * lda1 + aind * ldas]);
                        a[jt + (jt - 1) * lda1 + aind * ldas] = ZERO;
                        len = ilastm - jt;
                        SLC_DROT(&len, &a[(jt - 1) + jt * lda1 + aind * ldas], &lda1,
                                 &a[jt + jt * lda1 + aind * ldas], &lda1, &cs_t, &sn_t);
                        dwork[pdw] = cs_t;
                        dwork[pdw + 1] = sn_t;
                        pdw += 2;
                    }
                } else {
                    for (i32 jt = jlo; jt <= ntra; jt++) {
                        f64 cs_t = dwork[pdw];
                        f64 sn_t = dwork[pdw + 1];
                        i32 len = ilastm - jt + 1;
                        SLC_DROT(&len, &a[(jt - 1) + (jt - 1) * lda1 + aind * ldas], &lda1,
                                 &a[jt + (jt - 1) * lda1 + aind * ldas], &lda1, &cs_t, &sn_t);
                        f64 temp = a[jt + jt * lda1 + aind * ldas];
                        f64 neg = -a[jt + (jt - 1) * lda1 + aind * ldas];
                        SLC_DLARTG(&temp, &neg, &cs_t, &sn_t, &a[jt + jt * lda1 + aind * ldas]);
                        a[jt + (jt - 1) * lda1 + aind * ldas] = ZERO;
                        len = jt + 1 - ifrstm;
                        SLC_DROT(&len, &a[(ifrstm - 1) + (jt - 1) * lda1 + aind * ldas], &(i32){1},
                                 &a[(ifrstm - 1) + jt * lda1 + aind * ldas], &(i32){1}, &cs_t, &sn_t);
                        dwork[pdw] = cs_t;
                        dwork[pdw + 1] = sn_t;
                        pdw += 2;
                    }
                }
                if (lcmpq) {
                    qi = iwork[mapq + l] - 1;
                } else if (lparq) {
                    qi = qind[iwork[mapq + l] - 1];
                    qi = (qi < 0 ? -qi : qi) - 1;
                } else {
                    qi = -1;
                }
                if (qi >= 0) {
                    pdw = pfree;
                    for (i32 jt = jlo; jt <= ntra; jt++) {
                        f64 cs_t = dwork[pdw];
                        f64 sn_t = dwork[pdw + 1];
                        pdw += 2;
                        f64 *q_slice = q + qi * ldqs;
                        SLC_DROT(&n, &q_slice[(jt - 1) * ldq1], &(i32){1},
                                 &q_slice[jt * ldq1], &(i32){1}, &cs_t, &sn_t);
                    }
                }
            }

            // Apply transforms to RHS of Hessenberg
            aind = iwork[mapa] - 1;
            pdw = pfree;
            for (i32 jt = jlo; jt <= jdef - 2; jt++) {
                f64 cs_t = dwork[pdw];
                f64 sn_t = dwork[pdw + 1];
                pdw += 2;
                i32 len = jt + 2 - ifrstm;
                SLC_DROT(&len, &a[(ifrstm - 1) + (jt - 1) * lda1 + aind * ldas], &(i32){1},
                         &a[(ifrstm - 1) + jt * lda1 + aind * ldas], &(i32){1}, &cs_t, &sn_t);
            }

            // Unshifted QZ step backward
            pdw = pfree;
            for (i32 jt = ilast; jt >= jdef + 1; jt--) {
                f64 temp = a[(jt - 1) + (jt - 1) * lda1 + aind * ldas];
                f64 neg = -a[(jt - 1) + (jt - 2) * lda1 + aind * ldas];
                f64 cs_t, sn_t;
                SLC_DLARTG(&temp, &neg, &cs_t, &sn_t, &a[(jt - 1) + (jt - 1) * lda1 + aind * ldas]);
                a[(jt - 1) + (jt - 2) * lda1 + aind * ldas] = ZERO;
                i32 len = jt - ifrstm;
                SLC_DROT(&len, &a[(ifrstm - 1) + (jt - 2) * lda1 + aind * ldas], &(i32){1},
                         &a[(ifrstm - 1) + (jt - 1) * lda1 + aind * ldas], &(i32){1}, &cs_t, &sn_t);
                dwork[pdw] = cs_t;
                dwork[pdw + 1] = sn_t;
                pdw += 2;
            }
            if (lcmpq) {
                qi = iwork[mapq + 1] - 1;
            } else if (lparq) {
                qi = qind[iwork[mapq + 1] - 1];
                qi = (qi < 0 ? -qi : qi) - 1;
            } else {
                qi = -1;
            }
            if (qi >= 0) {
                pdw = pfree;
                for (i32 jt = ilast; jt >= jdef + 1; jt--) {
                    f64 cs_t = dwork[pdw];
                    f64 sn_t = dwork[pdw + 1];
                    pdw += 2;
                    f64 *q_slice = q + qi * ldqs;
                    SLC_DROT(&n, &q_slice[(jt - 2) * ldq1], &(i32){1},
                             &q_slice[(jt - 1) * ldq1], &(i32){1}, &cs_t, &sn_t);
                }
            }

            // Propagate through triangular factors
            for (i32 l = 1; l < k; l++) {
                aind = iwork[mapa + l] - 1;
                i32 ntra = (l + 1 > ldef) ? jdef + 2 : jdef + 1;
                pdw = pfree;
                if (s[aind] != sinv) {
                    for (i32 jt = ilast; jt >= ntra; jt--) {
                        f64 cs_t = dwork[pdw];
                        f64 sn_t = dwork[pdw + 1];
                        i32 len = jt + 1 - ifrstm;
                        SLC_DROT(&len, &a[(ifrstm - 1) + (jt - 2) * lda1 + aind * ldas], &(i32){1},
                                 &a[(ifrstm - 1) + (jt - 1) * lda1 + aind * ldas], &(i32){1}, &cs_t, &sn_t);
                        f64 temp = a[(jt - 2) + (jt - 2) * lda1 + aind * ldas];
                        SLC_DLARTG(&temp, &a[(jt - 1) + (jt - 2) * lda1 + aind * ldas], &cs_t, &sn_t, &a[(jt - 2) + (jt - 2) * lda1 + aind * ldas]);
                        a[(jt - 1) + (jt - 2) * lda1 + aind * ldas] = ZERO;
                        len = ilastm - jt + 1;
                        SLC_DROT(&len, &a[(jt - 2) + (jt - 1) * lda1 + aind * ldas], &lda1,
                                 &a[(jt - 1) + (jt - 1) * lda1 + aind * ldas], &lda1, &cs_t, &sn_t);
                        dwork[pdw] = cs_t;
                        dwork[pdw + 1] = sn_t;
                        pdw += 2;
                    }
                } else {
                    for (i32 jt = ilast; jt >= ntra; jt--) {
                        f64 cs_t = dwork[pdw];
                        f64 sn_t = dwork[pdw + 1];
                        i32 len = ilastm - jt + 2;
                        SLC_DROT(&len, &a[(jt - 2) + (jt - 2) * lda1 + aind * ldas], &lda1,
                                 &a[(jt - 1) + (jt - 2) * lda1 + aind * ldas], &lda1, &cs_t, &sn_t);
                        f64 temp = a[(jt - 1) + (jt - 1) * lda1 + aind * ldas];
                        f64 neg = -a[(jt - 1) + (jt - 2) * lda1 + aind * ldas];
                        SLC_DLARTG(&temp, &neg, &cs_t, &sn_t, &a[(jt - 1) + (jt - 1) * lda1 + aind * ldas]);
                        a[(jt - 1) + (jt - 2) * lda1 + aind * ldas] = ZERO;
                        len = jt - ifrstm;
                        SLC_DROT(&len, &a[(ifrstm - 1) + (jt - 2) * lda1 + aind * ldas], &(i32){1},
                                 &a[(ifrstm - 1) + (jt - 1) * lda1 + aind * ldas], &(i32){1}, &cs_t, &sn_t);
                        dwork[pdw] = cs_t;
                        dwork[pdw + 1] = sn_t;
                        pdw += 2;
                    }
                }
                i32 lm = (l + 1) % k;
                if (lcmpq) {
                    qi = iwork[mapq + lm] - 1;
                } else if (lparq) {
                    qi = qind[iwork[mapq + lm] - 1];
                    qi = (qi < 0 ? -qi : qi) - 1;
                } else {
                    qi = -1;
                }
                if (qi >= 0) {
                    pdw = pfree;
                    for (i32 jt = ilast; jt >= ntra; jt--) {
                        f64 cs_t = dwork[pdw];
                        f64 sn_t = dwork[pdw + 1];
                        pdw += 2;
                        f64 *q_slice = q + qi * ldqs;
                        SLC_DROT(&n, &q_slice[(jt - 2) * ldq1], &(i32){1},
                                 &q_slice[(jt - 1) * ldq1], &(i32){1}, &cs_t, &sn_t);
                    }
                }
            }

            // Apply transforms to LHS of Hessenberg
            aind = iwork[mapa] - 1;
            pdw = pfree;
            for (i32 jt = ilast; jt >= jdef + 2; jt--) {
                f64 cs_t = dwork[pdw];
                f64 sn_t = dwork[pdw + 1];
                pdw += 2;
                i32 len = ilastm - jt + 2;
                SLC_DROT(&len, &a[(jt - 2) + (jt - 2) * lda1 + aind * ldas], &lda1,
                         &a[(jt - 1) + (jt - 2) * lda1 + aind * ldas], &lda1, &cs_t, &sn_t);
            }
        }
        continue;

    case_iii:
        // Case III: Deflation in triangular matrix with index -1 (s[aind] != sinv)
        {
            i32 jdef = jdef_val;
            i32 j_c3 = jdef;

            if (jdef > (ilast - jlo + 1) / 2) {
                // Chase the zero downwards to the last position
                for (i32 j1 = jdef; j1 <= ilast - 1; j1++) {
                    j_c3 = j1;
                    aind = iwork[mapa + ldef - 1] - 1;
                    f64 temp = a[(j_c3 - 1) + j_c3 * lda1 + aind * ldas];
                    f64 cs_t, sn_t;
                    SLC_DLARTG(&temp, &a[j_c3 + j_c3 * lda1 + aind * ldas], &cs_t, &sn_t,
                               &a[(j_c3 - 1) + j_c3 * lda1 + aind * ldas]);
                    a[j_c3 + j_c3 * lda1 + aind * ldas] = ZERO;
                    i32 len = ilastm - j_c3 - 1;
                    if (len > 0) {
                        SLC_DROT(&len, &a[(j_c3 - 1) + (j_c3 + 1) * lda1 + aind * ldas], &lda1,
                                 &a[j_c3 + (j_c3 + 1) * lda1 + aind * ldas], &lda1, &cs_t, &sn_t);
                    }
                    i32 lm = ldef % k + 1;
                    i32 qi = -1;
                    if (lcmpq) {
                        qi = iwork[mapq + lm - 1] - 1;
                    } else if (lparq) {
                        qi = qind[iwork[mapq + lm - 1] - 1];
                        qi = (qi < 0 ? -qi : qi) - 1;
                    }
                    if (qi >= 0) {
                        f64 *q_slice = q + qi * ldqs;
                        SLC_DROT(&n, &q_slice[(j_c3 - 1) * ldq1], &(i32){1},
                                 &q_slice[j_c3 * ldq1], &(i32){1}, &cs_t, &sn_t);
                    }
                    for (i32 lc = 1; lc <= k - 1; lc++) {
                        aind = iwork[mapa + lm - 1] - 1;
                        if (lm == 1) {
                            // Hessenberg factor
                            len = ilastm - j_c3 + 2;
                            SLC_DROT(&len, &a[(j_c3 - 1) + (j_c3 - 2) * lda1 + aind * ldas], &lda1,
                                     &a[j_c3 + (j_c3 - 2) * lda1 + aind * ldas], &lda1, &cs_t, &sn_t);
                            temp = a[j_c3 + (j_c3 - 1) * lda1 + aind * ldas];
                            f64 neg = -a[j_c3 + (j_c3 - 2) * lda1 + aind * ldas];
                            SLC_DLARTG(&temp, &neg, &cs_t, &sn_t,
                                       &a[j_c3 + (j_c3 - 1) * lda1 + aind * ldas]);
                            a[j_c3 + (j_c3 - 2) * lda1 + aind * ldas] = ZERO;
                            len = j_c3 - ifrstm + 1;
                            SLC_DROT(&len, &a[(ifrstm - 1) + (j_c3 - 2) * lda1 + aind * ldas], &(i32){1},
                                     &a[(ifrstm - 1) + (j_c3 - 1) * lda1 + aind * ldas], &(i32){1}, &cs_t, &sn_t);
                            j_c3 = j_c3 - 1;
                        } else if (s[aind] == sinv) {
                            len = ilastm - j_c3 + 1;
                            SLC_DROT(&len, &a[(j_c3 - 1) + (j_c3 - 1) * lda1 + aind * ldas], &lda1,
                                     &a[j_c3 + (j_c3 - 1) * lda1 + aind * ldas], &lda1, &cs_t, &sn_t);
                            temp = a[j_c3 + j_c3 * lda1 + aind * ldas];
                            f64 neg = -a[j_c3 + (j_c3 - 1) * lda1 + aind * ldas];
                            SLC_DLARTG(&temp, &neg, &cs_t, &sn_t,
                                       &a[j_c3 + j_c3 * lda1 + aind * ldas]);
                            a[j_c3 + (j_c3 - 1) * lda1 + aind * ldas] = ZERO;
                            len = j_c3 - ifrstm + 1;
                            SLC_DROT(&len, &a[(ifrstm - 1) + (j_c3 - 1) * lda1 + aind * ldas], &(i32){1},
                                     &a[(ifrstm - 1) + j_c3 * lda1 + aind * ldas], &(i32){1}, &cs_t, &sn_t);
                        } else {
                            len = j_c3 - ifrstm + 2;
                            SLC_DROT(&len, &a[(ifrstm - 1) + (j_c3 - 1) * lda1 + aind * ldas], &(i32){1},
                                     &a[(ifrstm - 1) + j_c3 * lda1 + aind * ldas], &(i32){1}, &cs_t, &sn_t);
                            temp = a[(j_c3 - 1) + (j_c3 - 1) * lda1 + aind * ldas];
                            SLC_DLARTG(&temp, &a[j_c3 + (j_c3 - 1) * lda1 + aind * ldas], &cs_t, &sn_t,
                                       &a[(j_c3 - 1) + (j_c3 - 1) * lda1 + aind * ldas]);
                            a[j_c3 + (j_c3 - 1) * lda1 + aind * ldas] = ZERO;
                            len = ilastm - j_c3;
                            if (len > 0) {
                                SLC_DROT(&len, &a[(j_c3 - 1) + j_c3 * lda1 + aind * ldas], &lda1,
                                         &a[j_c3 + j_c3 * lda1 + aind * ldas], &lda1, &cs_t, &sn_t);
                            }
                        }
                        lm = lm % k + 1;
                        qi = -1;
                        if (lcmpq) {
                            qi = iwork[mapq + lm - 1] - 1;
                        } else if (lparq) {
                            qi = qind[iwork[mapq + lm - 1] - 1];
                            qi = (qi < 0 ? -qi : qi) - 1;
                        }
                        if (qi >= 0) {
                            f64 *q_slice = q + qi * ldqs;
                            SLC_DROT(&n, &q_slice[(j_c3 - 1) * ldq1], &(i32){1},
                                     &q_slice[j_c3 * ldq1], &(i32){1}, &cs_t, &sn_t);
                        }
                    }
                    aind = iwork[mapa + ldef - 1] - 1;
                    i32 len2 = j_c3 - ifrstm + 1;
                    SLC_DROT(&len2, &a[(ifrstm - 1) + (j_c3 - 1) * lda1 + aind * ldas], &(i32){1},
                             &a[(ifrstm - 1) + j_c3 * lda1 + aind * ldas], &(i32){1}, &cs_t, &sn_t);
                }

                // Deflate the last element in the Hessenberg matrix
                aind = iwork[mapa] - 1;
                j_c3 = ilast;
                f64 temp = a[(j_c3 - 1) + (j_c3 - 1) * lda1 + aind * ldas];
                f64 neg = -a[(j_c3 - 1) + (j_c3 - 2) * lda1 + aind * ldas];
                f64 cs_t, sn_t;
                SLC_DLARTG(&temp, &neg, &cs_t, &sn_t,
                           &a[(j_c3 - 1) + (j_c3 - 1) * lda1 + aind * ldas]);
                a[(j_c3 - 1) + (j_c3 - 2) * lda1 + aind * ldas] = ZERO;
                i32 len = j_c3 - ifrstm;
                if (len > 0) {
                    SLC_DROT(&len, &a[(ifrstm - 1) + (j_c3 - 2) * lda1 + aind * ldas], &(i32){1},
                             &a[(ifrstm - 1) + (j_c3 - 1) * lda1 + aind * ldas], &(i32){1}, &cs_t, &sn_t);
                }
                i32 qi = -1;
                if (lcmpq) {
                    qi = iwork[mapq + 1] - 1;
                } else if (lparq) {
                    qi = qind[iwork[mapq + 1] - 1];
                    qi = (qi < 0 ? -qi : qi) - 1;
                }
                if (qi >= 0) {
                    f64 *q_slice = q + qi * ldqs;
                    SLC_DROT(&n, &q_slice[(j_c3 - 2) * ldq1], &(i32){1},
                             &q_slice[(j_c3 - 1) * ldq1], &(i32){1}, &cs_t, &sn_t);
                }
                for (i32 lc = 2; lc <= ldef - 1; lc++) {
                    aind = iwork[mapa + lc - 1] - 1;
                    if (s[aind] != sinv) {
                        len = j_c3 + 1 - ifrstm;
                        SLC_DROT(&len, &a[(ifrstm - 1) + (j_c3 - 2) * lda1 + aind * ldas], &(i32){1},
                                 &a[(ifrstm - 1) + (j_c3 - 1) * lda1 + aind * ldas], &(i32){1}, &cs_t, &sn_t);
                        temp = a[(j_c3 - 2) + (j_c3 - 2) * lda1 + aind * ldas];
                        SLC_DLARTG(&temp, &a[(j_c3 - 1) + (j_c3 - 2) * lda1 + aind * ldas], &cs_t, &sn_t,
                                   &a[(j_c3 - 2) + (j_c3 - 2) * lda1 + aind * ldas]);
                        a[(j_c3 - 1) + (j_c3 - 2) * lda1 + aind * ldas] = ZERO;
                        len = ilastm - j_c3 + 1;
                        SLC_DROT(&len, &a[(j_c3 - 2) + (j_c3 - 1) * lda1 + aind * ldas], &lda1,
                                 &a[(j_c3 - 1) + (j_c3 - 1) * lda1 + aind * ldas], &lda1, &cs_t, &sn_t);
                    } else {
                        len = ilastm - j_c3 + 2;
                        SLC_DROT(&len, &a[(j_c3 - 2) + (j_c3 - 2) * lda1 + aind * ldas], &lda1,
                                 &a[(j_c3 - 1) + (j_c3 - 2) * lda1 + aind * ldas], &lda1, &cs_t, &sn_t);
                        temp = a[(j_c3 - 1) + (j_c3 - 1) * lda1 + aind * ldas];
                        neg = -a[(j_c3 - 1) + (j_c3 - 2) * lda1 + aind * ldas];
                        SLC_DLARTG(&temp, &neg, &cs_t, &sn_t,
                                   &a[(j_c3 - 1) + (j_c3 - 1) * lda1 + aind * ldas]);
                        a[(j_c3 - 1) + (j_c3 - 2) * lda1 + aind * ldas] = ZERO;
                        len = j_c3 - ifrstm;
                        if (len > 0) {
                            SLC_DROT(&len, &a[(ifrstm - 1) + (j_c3 - 2) * lda1 + aind * ldas], &(i32){1},
                                     &a[(ifrstm - 1) + (j_c3 - 1) * lda1 + aind * ldas], &(i32){1}, &cs_t, &sn_t);
                        }
                    }
                    i32 lm = lc + 1;
                    qi = -1;
                    if (lcmpq) {
                        qi = iwork[mapq + lm - 1] - 1;
                    } else if (lparq) {
                        qi = qind[iwork[mapq + lm - 1] - 1];
                        qi = (qi < 0 ? -qi : qi) - 1;
                    }
                    if (qi >= 0) {
                        f64 *q_slice = q + qi * ldqs;
                        SLC_DROT(&n, &q_slice[(j_c3 - 2) * ldq1], &(i32){1},
                                 &q_slice[(j_c3 - 1) * ldq1], &(i32){1}, &cs_t, &sn_t);
                    }
                }
                aind = iwork[mapa + ldef - 1] - 1;
                len = j_c3 + 1 - ifrstm;
                SLC_DROT(&len, &a[(ifrstm - 1) + (j_c3 - 2) * lda1 + aind * ldas], &(i32){1},
                         &a[(ifrstm - 1) + (j_c3 - 1) * lda1 + aind * ldas], &(i32){1}, &cs_t, &sn_t);
            } else {
                // Chase the zero upwards to the first position
                for (i32 j1 = jdef; j1 >= jlo + 1; j1--) {
                    j_c3 = j1;
                    aind = iwork[mapa + ldef - 1] - 1;
                    f64 temp = a[(j_c3 - 2) + (j_c3 - 1) * lda1 + aind * ldas];
                    f64 neg = -a[(j_c3 - 2) + (j_c3 - 2) * lda1 + aind * ldas];
                    f64 cs_t, sn_t;
                    SLC_DLARTG(&temp, &neg, &cs_t, &sn_t,
                               &a[(j_c3 - 2) + (j_c3 - 1) * lda1 + aind * ldas]);
                    a[(j_c3 - 2) + (j_c3 - 2) * lda1 + aind * ldas] = ZERO;
                    i32 len = j_c3 - ifrstm - 1;
                    if (len > 0) {
                        SLC_DROT(&len, &a[(ifrstm - 1) + (j_c3 - 2) * lda1 + aind * ldas], &(i32){1},
                                 &a[(ifrstm - 1) + (j_c3 - 1) * lda1 + aind * ldas], &(i32){1}, &cs_t, &sn_t);
                    }
                    i32 qi = -1;
                    if (lcmpq) {
                        qi = iwork[mapq + ldef - 1] - 1;
                    } else if (lparq) {
                        qi = qind[iwork[mapq + ldef - 1] - 1];
                        qi = (qi < 0 ? -qi : qi) - 1;
                    }
                    if (qi >= 0) {
                        f64 *q_slice = q + qi * ldqs;
                        SLC_DROT(&n, &q_slice[(j_c3 - 2) * ldq1], &(i32){1},
                                 &q_slice[(j_c3 - 1) * ldq1], &(i32){1}, &cs_t, &sn_t);
                    }
                    i32 lm = ldef - 1;
                    for (i32 lc = 1; lc <= k - 1; lc++) {
                        aind = iwork[mapa + lm - 1] - 1;
                        if (lm == 1) {
                            // Hessenberg factor
                            len = j_c3 - ifrstm + 2;
                            SLC_DROT(&len, &a[(ifrstm - 1) + (j_c3 - 2) * lda1 + aind * ldas], &(i32){1},
                                     &a[(ifrstm - 1) + (j_c3 - 1) * lda1 + aind * ldas], &(i32){1}, &cs_t, &sn_t);
                            temp = a[(j_c3 - 1) + (j_c3 - 2) * lda1 + aind * ldas];
                            SLC_DLARTG(&temp, &a[j_c3 + (j_c3 - 2) * lda1 + aind * ldas], &cs_t, &sn_t,
                                       &a[(j_c3 - 1) + (j_c3 - 2) * lda1 + aind * ldas]);
                            a[j_c3 + (j_c3 - 2) * lda1 + aind * ldas] = ZERO;
                            len = ilastm - j_c3 + 1;
                            SLC_DROT(&len, &a[(j_c3 - 1) + (j_c3 - 1) * lda1 + aind * ldas], &lda1,
                                     &a[j_c3 + (j_c3 - 1) * lda1 + aind * ldas], &lda1, &cs_t, &sn_t);
                            j_c3 = j_c3 + 1;
                        } else if (s[aind] != sinv) {
                            len = ilastm - j_c3 + 2;
                            SLC_DROT(&len, &a[(j_c3 - 2) + (j_c3 - 2) * lda1 + aind * ldas], &lda1,
                                     &a[(j_c3 - 1) + (j_c3 - 2) * lda1 + aind * ldas], &lda1, &cs_t, &sn_t);
                            temp = a[(j_c3 - 1) + (j_c3 - 1) * lda1 + aind * ldas];
                            neg = -a[(j_c3 - 1) + (j_c3 - 2) * lda1 + aind * ldas];
                            SLC_DLARTG(&temp, &neg, &cs_t, &sn_t,
                                       &a[(j_c3 - 1) + (j_c3 - 1) * lda1 + aind * ldas]);
                            a[(j_c3 - 1) + (j_c3 - 2) * lda1 + aind * ldas] = ZERO;
                            len = j_c3 - ifrstm;
                            if (len > 0) {
                                SLC_DROT(&len, &a[(ifrstm - 1) + (j_c3 - 2) * lda1 + aind * ldas], &(i32){1},
                                         &a[(ifrstm - 1) + (j_c3 - 1) * lda1 + aind * ldas], &(i32){1}, &cs_t, &sn_t);
                            }
                        } else {
                            len = j_c3 - ifrstm + 1;
                            SLC_DROT(&len, &a[(ifrstm - 1) + (j_c3 - 2) * lda1 + aind * ldas], &(i32){1},
                                     &a[(ifrstm - 1) + (j_c3 - 1) * lda1 + aind * ldas], &(i32){1}, &cs_t, &sn_t);
                            temp = a[(j_c3 - 2) + (j_c3 - 2) * lda1 + aind * ldas];
                            SLC_DLARTG(&temp, &a[(j_c3 - 1) + (j_c3 - 2) * lda1 + aind * ldas], &cs_t, &sn_t,
                                       &a[(j_c3 - 2) + (j_c3 - 2) * lda1 + aind * ldas]);
                            a[(j_c3 - 1) + (j_c3 - 2) * lda1 + aind * ldas] = ZERO;
                            len = ilastm - j_c3 + 1;
                            SLC_DROT(&len, &a[(j_c3 - 2) + (j_c3 - 1) * lda1 + aind * ldas], &lda1,
                                     &a[(j_c3 - 1) + (j_c3 - 1) * lda1 + aind * ldas], &lda1, &cs_t, &sn_t);
                        }
                        qi = -1;
                        if (lcmpq) {
                            qi = iwork[mapq + lm - 1] - 1;
                        } else if (lparq) {
                            qi = qind[iwork[mapq + lm - 1] - 1];
                            qi = (qi < 0 ? -qi : qi) - 1;
                        }
                        if (qi >= 0) {
                            f64 *q_slice = q + qi * ldqs;
                            SLC_DROT(&n, &q_slice[(j_c3 - 2) * ldq1], &(i32){1},
                                     &q_slice[(j_c3 - 1) * ldq1], &(i32){1}, &cs_t, &sn_t);
                        }
                        lm = lm - 1;
                        if (lm <= 0) lm = k;
                    }
                    aind = iwork[mapa + ldef - 1] - 1;
                    len = ilastm - j_c3 + 1;
                    SLC_DROT(&len, &a[(j_c3 - 2) + (j_c3 - 1) * lda1 + aind * ldas], &lda1,
                             &a[(j_c3 - 1) + (j_c3 - 1) * lda1 + aind * ldas], &lda1, &cs_t, &sn_t);
                }

                // Deflate the first element in the Hessenberg matrix
                aind = iwork[mapa] - 1;
                j_c3 = jlo;
                f64 temp = a[(j_c3 - 1) + (j_c3 - 1) * lda1 + aind * ldas];
                f64 cs_t, sn_t;
                SLC_DLARTG(&temp, &a[j_c3 + (j_c3 - 1) * lda1 + aind * ldas], &cs_t, &sn_t,
                           &a[(j_c3 - 1) + (j_c3 - 1) * lda1 + aind * ldas]);
                a[j_c3 + (j_c3 - 1) * lda1 + aind * ldas] = ZERO;
                i32 len = ilastm - j_c3;
                if (len > 0) {
                    SLC_DROT(&len, &a[(j_c3 - 1) + j_c3 * lda1 + aind * ldas], &lda1,
                             &a[j_c3 + j_c3 * lda1 + aind * ldas], &lda1, &cs_t, &sn_t);
                }
                i32 qi = -1;
                if (lcmpq) {
                    qi = iwork[mapq] - 1;
                } else if (lparq) {
                    qi = qind[iwork[mapq] - 1];
                    qi = (qi < 0 ? -qi : qi) - 1;
                }
                if (qi >= 0) {
                    f64 *q_slice = q + qi * ldqs;
                    SLC_DROT(&n, &q_slice[(j_c3 - 1) * ldq1], &(i32){1},
                             &q_slice[j_c3 * ldq1], &(i32){1}, &cs_t, &sn_t);
                }
                for (i32 lc = k; lc >= ldef + 1; lc--) {
                    aind = iwork[mapa + lc - 1] - 1;
                    if (s[aind] == sinv) {
                        len = j_c3 + 2 - ifrstm;
                        SLC_DROT(&len, &a[(ifrstm - 1) + (j_c3 - 1) * lda1 + aind * ldas], &(i32){1},
                                 &a[(ifrstm - 1) + j_c3 * lda1 + aind * ldas], &(i32){1}, &cs_t, &sn_t);
                        temp = a[(j_c3 - 1) + (j_c3 - 1) * lda1 + aind * ldas];
                        SLC_DLARTG(&temp, &a[j_c3 + (j_c3 - 1) * lda1 + aind * ldas], &cs_t, &sn_t,
                                   &a[(j_c3 - 1) + (j_c3 - 1) * lda1 + aind * ldas]);
                        a[j_c3 + (j_c3 - 1) * lda1 + aind * ldas] = ZERO;
                        len = ilastm - j_c3;
                        if (len > 0) {
                            SLC_DROT(&len, &a[(j_c3 - 1) + j_c3 * lda1 + aind * ldas], &lda1,
                                     &a[j_c3 + j_c3 * lda1 + aind * ldas], &lda1, &cs_t, &sn_t);
                        }
                    } else {
                        len = ilastm - j_c3 + 1;
                        SLC_DROT(&len, &a[(j_c3 - 1) + (j_c3 - 1) * lda1 + aind * ldas], &lda1,
                                 &a[j_c3 + (j_c3 - 1) * lda1 + aind * ldas], &lda1, &cs_t, &sn_t);
                        temp = a[j_c3 + j_c3 * lda1 + aind * ldas];
                        f64 neg = -a[j_c3 + (j_c3 - 1) * lda1 + aind * ldas];
                        SLC_DLARTG(&temp, &neg, &cs_t, &sn_t,
                                   &a[j_c3 + j_c3 * lda1 + aind * ldas]);
                        a[j_c3 + (j_c3 - 1) * lda1 + aind * ldas] = ZERO;
                        len = j_c3 + 1 - ifrstm;
                        SLC_DROT(&len, &a[(ifrstm - 1) + (j_c3 - 1) * lda1 + aind * ldas], &(i32){1},
                                 &a[(ifrstm - 1) + j_c3 * lda1 + aind * ldas], &(i32){1}, &cs_t, &sn_t);
                    }
                    qi = -1;
                    if (lcmpq) {
                        qi = iwork[mapq + lc - 1] - 1;
                    } else if (lparq) {
                        qi = qind[iwork[mapq + lc - 1] - 1];
                        qi = (qi < 0 ? -qi : qi) - 1;
                    }
                    if (qi >= 0) {
                        f64 *q_slice = q + qi * ldqs;
                        SLC_DROT(&n, &q_slice[(j_c3 - 1) * ldq1], &(i32){1},
                                 &q_slice[j_c3 * ldq1], &(i32){1}, &cs_t, &sn_t);
                    }
                }
                aind = iwork[mapa + ldef - 1] - 1;
                len = ilastm - j_c3;
                if (len > 0) {
                    SLC_DROT(&len, &a[(j_c3 - 1) + j_c3 * lda1 + aind * ldas], &lda1,
                             &a[j_c3 + j_c3 * lda1 + aind * ldas], &lda1, &cs_t, &sn_t);
                }
            }
        }
        continue;

    deflate_single:
#ifdef MB03BD_DEBUG
        fprintf(stderr, "MB03BD: deflate_single at ilast=%d\n", ilast);
#endif
        ma01bd(base, lgbas, k, s, &a[(ilast - 1) + (ilast - 1) * lda1], ldas,
               &alphar[ilast - 1], &beta[ilast - 1], &scal[ilast - 1]);
        alphai[ilast - 1] = ZERO;
        ilast--;
        if (ilast < ilo) {
            goto converged;
        }
        iiter = 0;
        continue;

#ifdef MB03BD_DEBUG
        fprintf(stderr, "MB03BD: deflate_double at ilast=%d\n", ilast);
#endif
        {
            f64 ar[2], ai_v[2], bt[2];
            i32 sc[2], ierr;

            f64 *a_block = a + (ilast - 2) + (ilast - 2) * lda1;
            mb03bb(base, lgbas, ulp, k, &iwork[mapa], s, sinv, a_block, lda1, lda2,
                   ar, ai_v, bt, sc, &dwork[pfree], &ierr);

#ifdef MB03BD_DEBUG
            fprintf(stderr, "MB03BD: mb03bb returned ar=[%.6e,%.6e], ai=[%.6e,%.6e]\n",
                    ar[0], ar[1], ai_v[0], ai_v[1]);
#endif

            alphar[ilast - 2] = ar[0];
            alphai[ilast - 2] = ai_v[0];
            beta[ilast - 2] = bt[0];
            scal[ilast - 2] = sc[0];
            alphar[ilast - 1] = ar[1];
            alphai[ilast - 1] = ai_v[1];
            beta[ilast - 1] = bt[1];
            scal[ilast - 1] = sc[1];

            ilast -= 2;
            if (ilast < ilo) {
                goto converged;
            }
            iiter = 0;
        }
        continue;
    }

failed:
    *info = ilast;
    goto done;

converged:
    for (i32 j = 0; j < ilo - 1; j++) {
        ma01bd(base, lgbas, k, s, &a[j + j * lda1], ldas, &alphar[j], &beta[j], &scal[j]);
        alphai[j] = ZERO;
    }

done:
    for (i32 l = k; l >= 1; l--) {
        dwork[pnorm + l] = dwork[pnorm + l - 1];
    }

    dwork[0] = (f64)optdw;
    iwork[0] = optiw;
}
