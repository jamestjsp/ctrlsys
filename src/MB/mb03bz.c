// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

static bool lsame(char a, char b) {
    return (a == b) || (a + 32 == b) || (a - 32 == b);
}

void mb03bz(const char *job, const char *compq, i32 k, i32 n, i32 ilo, i32 ihi,
            const i32 *s, c128 *a, i32 lda1, i32 lda2, c128 *q, i32 ldq1,
            i32 ldq2, c128 *alpha, c128 *beta, i32 *scal, f64 *dwork,
            i32 ldwork, c128 *zwork, i32 lzwork, i32 *info) {
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CONE = 1.0 + 0.0 * I;
    const c128 CZERO = 0.0 + 0.0 * I;

    bool lschr = lsame(job[0], 'S');
    bool liniq = lsame(compq[0], 'I');
    bool wantq = lsame(compq[0], 'V') || liniq;

    *info = 0;

    if (!lschr && !lsame(job[0], 'E')) {
        *info = -1;
    } else if (!wantq && !lsame(compq[0], 'N')) {
        *info = -2;
    } else if (k < 1) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (ilo < 1) {
        *info = -5;
    } else if (ihi > n || ihi < ilo - 1) {
        *info = -6;
    } else {
        bool sok = (s[0] == 1);
        for (i32 l = 1; l < k; l++) {
            sok = sok && (s[l] == 1 || s[l] == -1);
        }
        if (!sok) {
            *info = -7;
        } else if (lda1 < (n > 1 ? n : 1)) {
            *info = -9;
        } else if (lda2 < (n > 1 ? n : 1)) {
            *info = -10;
        } else if (ldq1 < 1 || (wantq && ldq1 < n)) {
            *info = -12;
        } else if (ldq2 < 1 || (wantq && ldq2 < n)) {
            *info = -13;
        } else if (ldwork < (n > 1 ? n : 1)) {
            *info = -18;
        } else if (lzwork < (n > 1 ? n : 1)) {
            *info = -20;
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        dwork[0] = ONE;
        zwork[0] = CONE;
        return;
    }

    i32 lda12 = lda1 * lda2;
    i32 ldq12 = ldq1 * ldq2;

    if (liniq) {
        for (i32 l = 0; l < k; l++) {
            c128 *ql = &q[l * ldq12];
            i32 mn = n;
            SLC_ZLASET("Full", &mn, &mn, &CZERO, &CONE, ql, &ldq1);
        }
    }

    i32 in = ihi - ilo + 1;
    f64 safmin = SLC_DLAMCH("SafeMinimum");
    f64 safmax = ONE / safmin;
    f64 ulp = SLC_DLAMCH("Precision");
    SLC_DLABAD(&safmin, &safmax);
    f64 smlnum = safmin * ((f64)in / ulp);
    f64 base = SLC_DLAMCH("Base");

    i32 ziter;
    f64 log_underflow = log(SLC_DLAMCH("Underflow"));
    f64 log_ulp = log(ulp);
    if (k >= (i32)(log_underflow / log_ulp)) {
        ziter = -1;
    } else {
        ziter = 0;
    }

    for (i32 j = ihi; j < n; j++) {
        c128 *aj = &a[j + j * lda1];
        ma01bz(base, k, s, aj, lda12, &alpha[j], &beta[j], &scal[j]);
    }

    if (ihi < ilo) {
        goto L460;
    }

    i32 ilast = ihi - 1;
    i32 ifrstm, ilastm;
    if (lschr) {
        ifrstm = 0;
        ilastm = n - 1;
    } else {
        ifrstm = ilo - 1;
        ilastm = ihi - 1;
    }

    i32 iiter = 0;
    i32 iseed[4] = {1, 0, 0, 1};
    i32 maxit = 30 * in;

    i32 ifirst, jlo, ldef, jdef, ln, ntra, j1;
    f64 cs, tol, abst;
    c128 sn, temp;
    c128 rnd[4];

    for (i32 jiter = 0; jiter < maxit; jiter++) {
        if (ilast == ilo - 1) {
            goto L390;
        }

        jlo = ilo - 1;
        for (i32 j = ilast; j >= ilo - 1; j--) {
            if (j == ilo - 1)
                break;
            i32 j0 = j - 1;
            i32 j1_ = j;
            c128 *a1 = &a[0];
            tol = cabs(a1[j0 + j0 * lda1]) + cabs(a1[j1_ + j1_ * lda1]);
            if (tol == ZERO) {
                i32 jmilo = j - (ilo - 1) + 1;
                tol = SLC_ZLANHS("1", &jmilo, &a1[(ilo - 1) + (ilo - 1) * lda1], &lda1, dwork);
            }
            tol = fmax(ulp * tol, smlnum);
            if (cabs(a1[j1_ + j0 * lda1]) <= tol) {
                a1[j1_ + j0 * lda1] = CZERO;
                jlo = j;
                if (j == ilast) {
                    goto L390;
                }
                break;
            }
        }

        for (i32 l = 1; l < k; l++) {
            if (s[l] == 1) {
                c128 *al = &a[l * lda12];
                for (i32 j = ilast; j >= jlo; j--) {
                    if (j == ilast) {
                        tol = cabs(al[(j - 1) + j * lda1]);
                    } else if (j == jlo) {
                        tol = cabs(al[j + (j + 1) * lda1]);
                    } else {
                        tol = cabs(al[(j - 1) + j * lda1]) + cabs(al[j + (j + 1) * lda1]);
                    }
                    if (tol == ZERO) {
                        i32 jmjlo = j - jlo + 1;
                        tol = SLC_ZLANTR("1", "Upper", "Non-unit", &jmjlo, &jmjlo,
                                         &al[jlo + jlo * lda1], &lda1, dwork);
                    }
                    tol = fmax(ulp * tol, smlnum);
                    if (cabs(al[j + j * lda1]) <= tol) {
                        al[j + j * lda1] = CZERO;
                        ldef = l;
                        jdef = j;
                        goto L170;
                    }
                }
            }
        }

        for (i32 l = 1; l < k; l++) {
            if (s[l] == -1) {
                c128 *al = &a[l * lda12];
                for (i32 j = ilast; j >= jlo; j--) {
                    if (j == ilast) {
                        tol = cabs(al[(j - 1) + j * lda1]);
                    } else if (j == jlo) {
                        tol = cabs(al[j + (j + 1) * lda1]);
                    } else {
                        tol = cabs(al[(j - 1) + j * lda1]) + cabs(al[j + (j + 1) * lda1]);
                    }
                    if (tol == ZERO) {
                        i32 jmjlo = j - jlo + 1;
                        tol = SLC_ZLANTR("1", "Upper", "Non-unit", &jmjlo, &jmjlo,
                                         &al[jlo + jlo * lda1], &lda1, dwork);
                    }
                    tol = fmax(ulp * tol, smlnum);
                    if (cabs(al[j + j * lda1]) <= tol) {
                        al[j + j * lda1] = CZERO;
                        ldef = l;
                        jdef = j;
                        goto L320;
                    }
                }
            }
        }

        if (ziter >= 7 || ziter < 0) {
            c128 *a1 = &a[0];
            for (i32 j = jlo; j < ilast; j++) {
                temp = a1[j + j * lda1];
                SLC_ZLARTG(&temp, &a1[(j + 1) + j * lda1], &cs, &sn, &a1[j + j * lda1]);
                a1[(j + 1) + j * lda1] = CZERO;
                i32 n_rot = ilastm - j;
                SLC_ZROT(&n_rot, &a1[j + (j + 1) * lda1], &lda1,
                         &a1[(j + 1) + (j + 1) * lda1], &lda1, &cs, &sn);
                dwork[j] = cs;
                zwork[j] = sn;
            }
            if (wantq) {
                c128 *q1 = &q[0];
                for (i32 j = jlo; j < ilast; j++) {
                    c128 snc = conj(zwork[j]);
                    SLC_ZROT(&n, &q1[j * ldq1], &(i32){1}, &q1[(j + 1) * ldq1],
                             &(i32){1}, &dwork[j], &snc);
                }
            }

            for (i32 l = k - 1; l >= 1; l--) {
                c128 *al = &a[l * lda12];
                if (s[l] == 1) {
                    for (i32 j = jlo; j < ilast; j++) {
                        sn = zwork[j];
                        if (sn != CZERO) {
                            cs = dwork[j];
                            i32 n_rot = j + 2 - ifrstm;
                            c128 snc = conj(sn);
                            SLC_ZROT(&n_rot, &al[ifrstm + j * lda1], &(i32){1},
                                     &al[ifrstm + (j + 1) * lda1], &(i32){1}, &cs, &snc);

                            tol = cabs(al[j + j * lda1]) + cabs(al[(j + 1) + (j + 1) * lda1]);
                            if (tol == ZERO) {
                                i32 jmjlo2 = j - jlo + 2;
                                tol = SLC_ZLANHS("1", &jmjlo2, &al[jlo + jlo * lda1], &lda1, dwork);
                            }
                            tol = fmax(ulp * tol, smlnum);
                            if (cabs(al[(j + 1) + j * lda1]) <= tol) {
                                cs = ONE;
                                sn = CZERO;
                                al[(j + 1) + j * lda1] = CZERO;
                            } else {
                                temp = al[j + j * lda1];
                                SLC_ZLARTG(&temp, &al[(j + 1) + j * lda1], &cs, &sn, &al[j + j * lda1]);
                                al[(j + 1) + j * lda1] = CZERO;
                                i32 n_rot2 = ilastm - j;
                                SLC_ZROT(&n_rot2, &al[j + (j + 1) * lda1], &lda1,
                                         &al[(j + 1) + (j + 1) * lda1], &lda1, &cs, &sn);
                            }
                            dwork[j] = cs;
                            zwork[j] = sn;
                        }
                    }
                } else {
                    for (i32 j = jlo; j < ilast; j++) {
                        sn = zwork[j];
                        if (sn != CZERO) {
                            cs = dwork[j];
                            i32 n_rot = ilastm - j + 1;
                            SLC_ZROT(&n_rot, &al[j + j * lda1], &lda1,
                                     &al[(j + 1) + j * lda1], &lda1, &cs, &sn);

                            tol = cabs(al[j + j * lda1]) + cabs(al[(j + 1) + (j + 1) * lda1]);
                            if (tol == ZERO) {
                                i32 jmjlo2 = j - jlo + 2;
                                tol = SLC_ZLANHS("1", &jmjlo2, &al[jlo + jlo * lda1], &lda1, dwork);
                            }
                            tol = fmax(ulp * tol, smlnum);
                            if (cabs(al[(j + 1) + j * lda1]) <= tol) {
                                cs = ONE;
                                sn = CZERO;
                                al[(j + 1) + j * lda1] = CZERO;
                            } else {
                                temp = al[(j + 1) + (j + 1) * lda1];
                                SLC_ZLARTG(&temp, &al[(j + 1) + j * lda1], &cs, &sn, &al[(j + 1) + (j + 1) * lda1]);
                                al[(j + 1) + j * lda1] = CZERO;
                                i32 n_rot2 = j + 1 - ifrstm;
                                SLC_ZROT(&n_rot2, &al[ifrstm + (j + 1) * lda1], &(i32){1},
                                         &al[ifrstm + j * lda1], &(i32){1}, &cs, &sn);
                            }
                            dwork[j] = cs;
                            zwork[j] = -sn;
                        }
                    }
                }

                if (wantq) {
                    c128 *ql = &q[l * ldq12];
                    for (i32 j = jlo; j < ilast; j++) {
                        c128 snc = conj(zwork[j]);
                        SLC_ZROT(&n, &ql[j * ldq1], &(i32){1}, &ql[(j + 1) * ldq1],
                                 &(i32){1}, &dwork[j], &snc);
                    }
                }
            }

            ziter = 0;
            for (i32 j = jlo; j < ilast; j++) {
                c128 *a1_fin = &a[0];
                cs = dwork[j];
                sn = zwork[j];
                i32 n_rot = j + 2 - ifrstm;
                c128 snc = conj(sn);
                SLC_ZROT(&n_rot, &a1_fin[ifrstm + j * lda1], &(i32){1},
                         &a1_fin[ifrstm + (j + 1) * lda1], &(i32){1}, &cs, &snc);
                if (sn == CZERO) {
                    ziter = 1;
                }
            }

            goto L440;
        }

        ifirst = jlo;
        goto L400;

    L170:
        {
            c128 *a1 = &a[0];
            for (i32 j = jlo; j < jdef; j++) {
                temp = a1[j + j * lda1];
                SLC_ZLARTG(&temp, &a1[(j + 1) + j * lda1], &cs, &sn, &a1[j + j * lda1]);
                a1[(j + 1) + j * lda1] = CZERO;
                i32 n_rot = ilastm - j;
                SLC_ZROT(&n_rot, &a1[j + (j + 1) * lda1], &lda1,
                         &a1[(j + 1) + (j + 1) * lda1], &lda1, &cs, &sn);
                dwork[j] = cs;
                zwork[j] = sn;
            }
            if (wantq) {
                c128 *q1 = &q[0];
                for (i32 j = jlo; j < jdef; j++) {
                    c128 snc = conj(zwork[j]);
                    SLC_ZROT(&n, &q1[j * ldq1], &(i32){1}, &q1[(j + 1) * ldq1],
                             &(i32){1}, &dwork[j], &snc);
                }
            }

            for (i32 l = k - 1; l >= 1; l--) {
                c128 *al = &a[l * lda12];
                if (l < ldef) {
                    ntra = jdef - 2;
                } else {
                    ntra = jdef - 1;
                }
                if (s[l] == 1) {
                    for (i32 j = jlo; j <= ntra; j++) {
                        i32 n_rot = j + 2 - ifrstm;
                        c128 snc = conj(zwork[j]);
                        SLC_ZROT(&n_rot, &al[ifrstm + j * lda1], &(i32){1},
                                 &al[ifrstm + (j + 1) * lda1], &(i32){1}, &dwork[j], &snc);
                        temp = al[j + j * lda1];
                        SLC_ZLARTG(&temp, &al[(j + 1) + j * lda1], &cs, &sn, &al[j + j * lda1]);
                        al[(j + 1) + j * lda1] = CZERO;
                        i32 n_rot2 = ilastm - j;
                        SLC_ZROT(&n_rot2, &al[j + (j + 1) * lda1], &lda1,
                                 &al[(j + 1) + (j + 1) * lda1], &lda1, &cs, &sn);
                        dwork[j] = cs;
                        zwork[j] = sn;
                    }
                } else {
                    for (i32 j = jlo; j <= ntra; j++) {
                        i32 n_rot = ilastm - j + 1;
                        SLC_ZROT(&n_rot, &al[j + j * lda1], &lda1,
                                 &al[(j + 1) + j * lda1], &lda1, &dwork[j], &zwork[j]);
                        temp = al[(j + 1) + (j + 1) * lda1];
                        SLC_ZLARTG(&temp, &al[(j + 1) + j * lda1], &cs, &sn, &al[(j + 1) + (j + 1) * lda1]);
                        al[(j + 1) + j * lda1] = CZERO;
                        i32 n_rot2 = j + 1 - ifrstm;
                        SLC_ZROT(&n_rot2, &al[ifrstm + (j + 1) * lda1], &(i32){1},
                                 &al[ifrstm + j * lda1], &(i32){1}, &cs, &sn);
                        dwork[j] = cs;
                        zwork[j] = -sn;
                    }
                }
                if (wantq) {
                    c128 *ql = &q[l * ldq12];
                    for (i32 j = jlo; j <= ntra; j++) {
                        c128 snc = conj(zwork[j]);
                        SLC_ZROT(&n, &ql[j * ldq1], &(i32){1}, &ql[(j + 1) * ldq1],
                                 &(i32){1}, &dwork[j], &snc);
                    }
                }
            }

            for (i32 j = jlo; j <= jdef - 2; j++) {
                i32 n_rot = j + 2 - ifrstm;
                c128 snc = conj(zwork[j]);
                SLC_ZROT(&n_rot, &a1[ifrstm + j * lda1], &(i32){1},
                         &a1[ifrstm + (j + 1) * lda1], &(i32){1}, &dwork[j], &snc);
            }

            for (i32 j = ilast; j >= jdef + 1; j--) {
                temp = a1[j + j * lda1];
                SLC_ZLARTG(&temp, &a1[j + (j - 1) * lda1], &cs, &sn, &a1[j + j * lda1]);
                a1[j + (j - 1) * lda1] = CZERO;
                i32 n_rot = j - ifrstm;
                SLC_ZROT(&n_rot, &a1[ifrstm + j * lda1], &(i32){1},
                         &a1[ifrstm + (j - 1) * lda1], &(i32){1}, &cs, &sn);
                dwork[j] = cs;
                zwork[j] = -sn;
            }
            if (wantq) {
                c128 *q2 = &q[1 * ldq12];
                for (i32 j = ilast; j >= jdef + 1; j--) {
                    c128 snc = conj(zwork[j]);
                    SLC_ZROT(&n, &q2[(j - 1) * ldq1], &(i32){1}, &q2[j * ldq1],
                             &(i32){1}, &dwork[j], &snc);
                }
            }

            for (i32 l = 1; l < k; l++) {
                c128 *al = &a[l * lda12];
                if (l > ldef) {
                    ntra = jdef + 2;
                } else {
                    ntra = jdef + 1;
                }
                if (s[l] == -1) {
                    for (i32 j = ilast; j >= ntra; j--) {
                        cs = dwork[j];
                        sn = zwork[j];
                        i32 n_rot = j + 1 - ifrstm;
                        c128 snc = conj(sn);
                        SLC_ZROT(&n_rot, &al[ifrstm + (j - 1) * lda1], &(i32){1},
                                 &al[ifrstm + j * lda1], &(i32){1}, &cs, &snc);
                        temp = al[(j - 1) + (j - 1) * lda1];
                        SLC_ZLARTG(&temp, &al[j + (j - 1) * lda1], &cs, &sn, &al[(j - 1) + (j - 1) * lda1]);
                        al[j + (j - 1) * lda1] = CZERO;
                        i32 n_rot2 = ilastm - j + 1;
                        SLC_ZROT(&n_rot2, &al[(j - 1) + j * lda1], &lda1,
                                 &al[j + j * lda1], &lda1, &cs, &sn);
                        dwork[j] = cs;
                        zwork[j] = sn;
                    }
                } else {
                    for (i32 j = ilast; j >= ntra; j--) {
                        i32 n_rot = ilastm - j + 2;
                        SLC_ZROT(&n_rot, &al[(j - 1) + (j - 1) * lda1], &lda1,
                                 &al[j + (j - 1) * lda1], &lda1, &dwork[j], &zwork[j]);
                        temp = al[j + j * lda1];
                        SLC_ZLARTG(&temp, &al[j + (j - 1) * lda1], &cs, &sn, &al[j + j * lda1]);
                        al[j + (j - 1) * lda1] = CZERO;
                        i32 n_rot2 = j - ifrstm;
                        SLC_ZROT(&n_rot2, &al[ifrstm + j * lda1], &(i32){1},
                                 &al[ifrstm + (j - 1) * lda1], &(i32){1}, &cs, &sn);
                        dwork[j] = cs;
                        zwork[j] = -sn;
                    }
                }
                if (wantq) {
                    i32 ln_q;
                    if (l == k - 1) {
                        ln_q = 0;
                    } else {
                        ln_q = l + 1;
                    }
                    c128 *ql = &q[ln_q * ldq12];
                    for (i32 j = ilast; j >= ntra; j--) {
                        c128 snc = conj(zwork[j]);
                        SLC_ZROT(&n, &ql[(j - 1) * ldq1], &(i32){1}, &ql[j * ldq1],
                                 &(i32){1}, &dwork[j], &snc);
                    }
                }
            }

            for (i32 j = ilast; j >= jdef + 2; j--) {
                i32 n_rot = ilastm - j + 2;
                SLC_ZROT(&n_rot, &a1[(j - 1) + (j - 1) * lda1], &lda1,
                         &a1[j + (j - 1) * lda1], &lda1, &dwork[j], &zwork[j]);
            }
        }

        goto L440;

    L320:
        {
            if (jdef > (ilast - jlo + 1) / 2) {
                c128 *aldef = &a[ldef * lda12];
                for (j1 = jdef; j1 < ilast; j1++) {
                    i32 j = j1;
                    temp = aldef[j + (j + 1) * lda1];
                    SLC_ZLARTG(&temp, &aldef[(j + 1) + (j + 1) * lda1], &cs, &sn, &aldef[j + (j + 1) * lda1]);
                    aldef[(j + 1) + (j + 1) * lda1] = CZERO;
                    i32 n_rot = ilastm - j - 1;
                    if (n_rot > 0) {
                        SLC_ZROT(&n_rot, &aldef[j + (j + 2) * lda1], &lda1,
                                 &aldef[(j + 1) + (j + 2) * lda1], &lda1, &cs, &sn);
                    }
                    ln = (ldef == k - 1) ? 0 : ldef + 1;
                    if (wantq) {
                        c128 *ql = &q[ln * ldq12];
                        c128 snc = conj(sn);
                        SLC_ZROT(&n, &ql[j * ldq1], &(i32){1}, &ql[(j + 1) * ldq1],
                                 &(i32){1}, &cs, &snc);
                    }
                    for (i32 lp = 0; lp < k - 1; lp++) {
                        c128 *aln = &a[ln * lda12];
                        if (ln == 0) {
                            i32 n_rot2 = ilastm - j + 2;
                            SLC_ZROT(&n_rot2, &aln[j + (j - 1) * lda1], &lda1,
                                     &aln[(j + 1) + (j - 1) * lda1], &lda1, &cs, &sn);
                            temp = aln[(j + 1) + j * lda1];
                            SLC_ZLARTG(&temp, &aln[(j + 1) + (j - 1) * lda1], &cs, &sn, &aln[(j + 1) + j * lda1]);
                            aln[(j + 1) + (j - 1) * lda1] = CZERO;
                            i32 n_rot3 = j - ifrstm + 1;
                            SLC_ZROT(&n_rot3, &aln[ifrstm + j * lda1], &(i32){1},
                                     &aln[ifrstm + (j - 1) * lda1], &(i32){1}, &cs, &sn);
                            sn = -sn;
                            j = j - 1;
                        } else if (s[ln] == 1) {
                            i32 n_rot2 = ilastm - j + 1;
                            SLC_ZROT(&n_rot2, &aln[j + j * lda1], &lda1,
                                     &aln[(j + 1) + j * lda1], &lda1, &cs, &sn);
                            temp = aln[(j + 1) + (j + 1) * lda1];
                            SLC_ZLARTG(&temp, &aln[(j + 1) + j * lda1], &cs, &sn, &aln[(j + 1) + (j + 1) * lda1]);
                            aln[(j + 1) + j * lda1] = CZERO;
                            i32 n_rot3 = j - ifrstm + 1;
                            SLC_ZROT(&n_rot3, &aln[ifrstm + (j + 1) * lda1], &(i32){1},
                                     &aln[ifrstm + j * lda1], &(i32){1}, &cs, &sn);
                            sn = -sn;
                        } else {
                            i32 n_rot2 = j - ifrstm + 2;
                            c128 snc = conj(sn);
                            SLC_ZROT(&n_rot2, &aln[ifrstm + j * lda1], &(i32){1},
                                     &aln[ifrstm + (j + 1) * lda1], &(i32){1}, &cs, &snc);
                            temp = aln[j + j * lda1];
                            SLC_ZLARTG(&temp, &aln[(j + 1) + j * lda1], &cs, &sn, &aln[j + j * lda1]);
                            aln[(j + 1) + j * lda1] = CZERO;
                            i32 n_rot3 = ilastm - j;
                            SLC_ZROT(&n_rot3, &aln[j + (j + 1) * lda1], &lda1,
                                     &aln[(j + 1) + (j + 1) * lda1], &lda1, &cs, &sn);
                        }
                        ln = ln + 1;
                        if (ln >= k) ln = 0;
                        if (wantq) {
                            c128 *ql = &q[ln * ldq12];
                            c128 snc = conj(sn);
                            SLC_ZROT(&n, &ql[j * ldq1], &(i32){1}, &ql[(j + 1) * ldq1],
                                     &(i32){1}, &cs, &snc);
                        }
                    }
                    c128 snc = conj(sn);
                    i32 n_rot_last = j - ifrstm + 1;
                    SLC_ZROT(&n_rot_last, &aldef[ifrstm + j * lda1], &(i32){1},
                             &aldef[ifrstm + (j + 1) * lda1], &(i32){1}, &cs, &snc);
                }

                c128 *a1 = &a[0];
                i32 j = ilast;
                temp = a1[j + j * lda1];
                SLC_ZLARTG(&temp, &a1[j + (j - 1) * lda1], &cs, &sn, &a1[j + j * lda1]);
                a1[j + (j - 1) * lda1] = CZERO;
                i32 n_rot = j - ifrstm;
                SLC_ZROT(&n_rot, &a1[ifrstm + j * lda1], &(i32){1},
                         &a1[ifrstm + (j - 1) * lda1], &(i32){1}, &cs, &sn);
                sn = -sn;
                if (wantq) {
                    c128 *q2 = &q[1 * ldq12];
                    c128 snc = conj(sn);
                    SLC_ZROT(&n, &q2[(j - 1) * ldq1], &(i32){1}, &q2[j * ldq1],
                             &(i32){1}, &cs, &snc);
                }
                for (i32 l = 1; l < ldef; l++) {
                    c128 *al = &a[l * lda12];
                    if (s[l] == -1) {
                        i32 n_rot2 = j + 1 - ifrstm;
                        c128 snc = conj(sn);
                        SLC_ZROT(&n_rot2, &al[ifrstm + (j - 1) * lda1], &(i32){1},
                                 &al[ifrstm + j * lda1], &(i32){1}, &cs, &snc);
                        temp = al[(j - 1) + (j - 1) * lda1];
                        SLC_ZLARTG(&temp, &al[j + (j - 1) * lda1], &cs, &sn, &al[(j - 1) + (j - 1) * lda1]);
                        al[j + (j - 1) * lda1] = CZERO;
                        i32 n_rot3 = ilastm - j + 1;
                        SLC_ZROT(&n_rot3, &al[(j - 1) + j * lda1], &lda1,
                                 &al[j + j * lda1], &lda1, &cs, &sn);
                    } else {
                        i32 n_rot2 = ilastm - j + 2;
                        SLC_ZROT(&n_rot2, &al[(j - 1) + (j - 1) * lda1], &lda1,
                                 &al[j + (j - 1) * lda1], &lda1, &cs, &sn);
                        temp = al[j + j * lda1];
                        SLC_ZLARTG(&temp, &al[j + (j - 1) * lda1], &cs, &sn, &al[j + j * lda1]);
                        al[j + (j - 1) * lda1] = CZERO;
                        i32 n_rot3 = j - ifrstm;
                        SLC_ZROT(&n_rot3, &al[ifrstm + j * lda1], &(i32){1},
                                 &al[ifrstm + (j - 1) * lda1], &(i32){1}, &cs, &sn);
                        sn = -sn;
                    }
                    if (wantq) {
                        i32 ln_q;
                        if (l == k - 1) {
                            ln_q = 0;
                        } else {
                            ln_q = l + 1;
                        }
                        c128 *ql = &q[ln_q * ldq12];
                        c128 snc = conj(sn);
                        SLC_ZROT(&n, &ql[(j - 1) * ldq1], &(i32){1}, &ql[j * ldq1],
                                 &(i32){1}, &cs, &snc);
                    }
                }
                c128 snc = conj(sn);
                i32 n_rot_final = j + 1 - ifrstm;
                SLC_ZROT(&n_rot_final, &aldef[ifrstm + (j - 1) * lda1], &(i32){1},
                         &aldef[ifrstm + j * lda1], &(i32){1}, &cs, &snc);
            } else {
                c128 *aldef = &a[ldef * lda12];
                for (j1 = jdef; j1 >= jlo + 1; j1--) {
                    i32 j = j1;
                    temp = aldef[(j - 1) + j * lda1];
                    SLC_ZLARTG(&temp, &aldef[(j - 1) + (j - 1) * lda1], &cs, &sn, &aldef[(j - 1) + j * lda1]);
                    aldef[(j - 1) + (j - 1) * lda1] = CZERO;
                    i32 n_rot = j - ifrstm - 1;
                    if (n_rot > 0) {
                        SLC_ZROT(&n_rot, &aldef[ifrstm + j * lda1], &(i32){1},
                                 &aldef[ifrstm + (j - 1) * lda1], &(i32){1}, &cs, &sn);
                    }
                    sn = -sn;
                    if (wantq) {
                        c128 *ql = &q[ldef * ldq12];
                        c128 snc = conj(sn);
                        SLC_ZROT(&n, &ql[(j - 1) * ldq1], &(i32){1}, &ql[j * ldq1],
                                 &(i32){1}, &cs, &snc);
                    }
                    ln = ldef - 1;
                    for (i32 lp = 0; lp < k - 1; lp++) {
                        c128 *aln = &a[ln * lda12];
                        if (ln == 0) {
                            i32 n_rot2 = j - ifrstm + 2;
                            c128 snc = conj(sn);
                            SLC_ZROT(&n_rot2, &aln[ifrstm + (j - 1) * lda1], &(i32){1},
                                     &aln[ifrstm + j * lda1], &(i32){1}, &cs, &snc);
                            temp = aln[j + (j - 1) * lda1];
                            SLC_ZLARTG(&temp, &aln[(j + 1) + (j - 1) * lda1], &cs, &sn, &aln[j + (j - 1) * lda1]);
                            aln[(j + 1) + (j - 1) * lda1] = CZERO;
                            i32 n_rot3 = ilastm - j + 1;
                            SLC_ZROT(&n_rot3, &aln[j + j * lda1], &lda1,
                                     &aln[(j + 1) + j * lda1], &lda1, &cs, &sn);
                            j = j + 1;
                        } else if (s[ln] == -1) {
                            i32 n_rot2 = ilastm - j + 2;
                            SLC_ZROT(&n_rot2, &aln[(j - 1) + (j - 1) * lda1], &lda1,
                                     &aln[j + (j - 1) * lda1], &lda1, &cs, &sn);
                            temp = aln[j + j * lda1];
                            SLC_ZLARTG(&temp, &aln[j + (j - 1) * lda1], &cs, &sn, &aln[j + j * lda1]);
                            aln[j + (j - 1) * lda1] = CZERO;
                            i32 n_rot3 = j - ifrstm;
                            SLC_ZROT(&n_rot3, &aln[ifrstm + j * lda1], &(i32){1},
                                     &aln[ifrstm + (j - 1) * lda1], &(i32){1}, &cs, &sn);
                            sn = -sn;
                        } else {
                            i32 n_rot2 = j - ifrstm + 1;
                            c128 snc = conj(sn);
                            SLC_ZROT(&n_rot2, &aln[ifrstm + (j - 1) * lda1], &(i32){1},
                                     &aln[ifrstm + j * lda1], &(i32){1}, &cs, &snc);
                            temp = aln[(j - 1) + (j - 1) * lda1];
                            SLC_ZLARTG(&temp, &aln[j + (j - 1) * lda1], &cs, &sn, &aln[(j - 1) + (j - 1) * lda1]);
                            aln[j + (j - 1) * lda1] = CZERO;
                            i32 n_rot3 = ilastm - j + 1;
                            SLC_ZROT(&n_rot3, &aln[(j - 1) + j * lda1], &lda1,
                                     &aln[j + j * lda1], &lda1, &cs, &sn);
                        }
                        if (wantq) {
                            c128 *ql = &q[ln * ldq12];
                            c128 snc = conj(sn);
                            SLC_ZROT(&n, &ql[(j - 1) * ldq1], &(i32){1}, &ql[j * ldq1],
                                     &(i32){1}, &cs, &snc);
                        }
                        ln = ln - 1;
                        if (ln < 0) ln = k - 1;
                    }
                    i32 n_rot_last = ilastm - j + 1;
                    SLC_ZROT(&n_rot_last, &aldef[(j - 1) + j * lda1], &lda1,
                             &aldef[j + j * lda1], &lda1, &cs, &sn);
                }

                c128 *a1 = &a[0];
                i32 j = jlo;
                temp = a1[j + j * lda1];
                SLC_ZLARTG(&temp, &a1[(j + 1) + j * lda1], &cs, &sn, &a1[j + j * lda1]);
                a1[(j + 1) + j * lda1] = CZERO;
                i32 n_rot = ilastm - j;
                SLC_ZROT(&n_rot, &a1[j + (j + 1) * lda1], &lda1,
                         &a1[(j + 1) + (j + 1) * lda1], &lda1, &cs, &sn);
                if (wantq) {
                    c128 *q1 = &q[0];
                    c128 snc = conj(sn);
                    SLC_ZROT(&n, &q1[j * ldq1], &(i32){1}, &q1[(j + 1) * ldq1],
                             &(i32){1}, &cs, &snc);
                }
                for (i32 l = k - 1; l >= ldef + 1; l--) {
                    c128 *al = &a[l * lda12];
                    if (s[l] == 1) {
                        i32 n_rot2 = j + 2 - ifrstm;
                        c128 snc = conj(sn);
                        SLC_ZROT(&n_rot2, &al[ifrstm + j * lda1], &(i32){1},
                                 &al[ifrstm + (j + 1) * lda1], &(i32){1}, &cs, &snc);
                        temp = al[j + j * lda1];
                        SLC_ZLARTG(&temp, &al[(j + 1) + j * lda1], &cs, &sn, &al[j + j * lda1]);
                        al[(j + 1) + j * lda1] = CZERO;
                        i32 n_rot3 = ilastm - j;
                        SLC_ZROT(&n_rot3, &al[j + (j + 1) * lda1], &lda1,
                                 &al[(j + 1) + (j + 1) * lda1], &lda1, &cs, &sn);
                    } else {
                        i32 n_rot2 = ilastm - j + 1;
                        SLC_ZROT(&n_rot2, &al[j + j * lda1], &lda1,
                                 &al[(j + 1) + j * lda1], &lda1, &cs, &sn);
                        temp = al[(j + 1) + (j + 1) * lda1];
                        SLC_ZLARTG(&temp, &al[(j + 1) + j * lda1], &cs, &sn, &al[(j + 1) + (j + 1) * lda1]);
                        al[(j + 1) + j * lda1] = CZERO;
                        i32 n_rot3 = j + 1 - ifrstm;
                        SLC_ZROT(&n_rot3, &al[ifrstm + (j + 1) * lda1], &(i32){1},
                                 &al[ifrstm + j * lda1], &(i32){1}, &cs, &sn);
                        sn = -sn;
                    }
                    if (wantq) {
                        c128 *ql = &q[l * ldq12];
                        c128 snc = conj(sn);
                        SLC_ZROT(&n, &ql[j * ldq1], &(i32){1}, &ql[(j + 1) * ldq1],
                                 &(i32){1}, &cs, &snc);
                    }
                }
                i32 n_rot_final = ilastm - j;
                SLC_ZROT(&n_rot_final, &aldef[j + (j + 1) * lda1], &lda1,
                         &aldef[(j + 1) + (j + 1) * lda1], &lda1, &cs, &sn);
            }
        }

        goto L440;

    L390:
        {
            c128 *aj = &a[ilast + ilast * lda1];
            ma01bz(base, k, s, aj, lda12, &alpha[ilast], &beta[ilast], &scal[ilast]);
        }

        ilast = ilast - 1;
        if (ilast < ilo - 1) {
            goto L460;
        }

        iiter = 0;
        if (ziter != -1) {
            ziter = 0;
        }
        if (!lschr) {
            ilastm = ilast;
            if (ifrstm > ilast) {
                ifrstm = ilo - 1;
            }
        }

        goto L440;

    L400:
        iiter = iiter + 1;
        ziter = ziter + 1;
        if (!lschr) {
            ifrstm = ifirst;
        }

        if (iiter % 10 == 0) {
            i32 idist = 2;
            i32 nrand = 2;
            SLC_ZLARNV(&idist, iseed, &nrand, rnd);
            SLC_ZLARTG(&rnd[0], &rnd[1], &cs, &sn, &temp);
        } else {
            SLC_ZLARTG(&CONE, &CONE, &cs, &sn, &temp);
            for (i32 l = k - 1; l >= 1; l--) {
                c128 *al = &a[l * lda12];
                if (s[l] == 1) {
                    c128 arg1 = al[ifirst + ifirst * lda1] * cs;
                    c128 arg2 = al[ilast + ilast * lda1] * conj(sn);
                    SLC_ZLARTG(&arg1, &arg2, &cs, &sn, &temp);
                } else {
                    c128 arg1 = al[ilast + ilast * lda1] * cs;
                    c128 arg2 = -al[ifirst + ifirst * lda1] * conj(sn);
                    SLC_ZLARTG(&arg1, &arg2, &cs, &sn, &temp);
                    sn = -sn;
                }
            }
            c128 *a1 = &a[0];
            c128 arg1 = a1[ifirst + ifirst * lda1] * cs - a1[ilast + ilast * lda1] * conj(sn);
            c128 arg2 = a1[(ifirst + 1) + ifirst * lda1] * cs;
            SLC_ZLARTG(&arg1, &arg2, &cs, &sn, &temp);
        }

        {
            c128 *a1 = &a[0];
            for (j1 = ifirst - 1; j1 <= ilast - 2; j1++) {
                i32 j = j1 + 1;

                if (j1 >= ifirst) {
                    temp = a1[j + (j - 1) * lda1];
                    SLC_ZLARTG(&temp, &a1[(j + 1) + (j - 1) * lda1], &cs, &sn, &a1[j + (j - 1) * lda1]);
                    a1[(j + 1) + (j - 1) * lda1] = CZERO;
                }
                i32 n_rot = ilastm - j + 1;
                SLC_ZROT(&n_rot, &a1[j + j * lda1], &lda1,
                         &a1[(j + 1) + j * lda1], &lda1, &cs, &sn);
                if (wantq) {
                    c128 *q1 = &q[0];
                    c128 snc = conj(sn);
                    SLC_ZROT(&n, &q1[j * ldq1], &(i32){1}, &q1[(j + 1) * ldq1],
                             &(i32){1}, &cs, &snc);
                }

                for (i32 l = k - 1; l >= 1; l--) {
                    c128 *al = &a[l * lda12];
                    if (s[l] == 1) {
                        i32 n_rot2 = j + 2 - ifrstm;
                        c128 snc = conj(sn);
                        SLC_ZROT(&n_rot2, &al[ifrstm + j * lda1], &(i32){1},
                                 &al[ifrstm + (j + 1) * lda1], &(i32){1}, &cs, &snc);
                        temp = al[j + j * lda1];
                        SLC_ZLARTG(&temp, &al[(j + 1) + j * lda1], &cs, &sn, &al[j + j * lda1]);
                        al[(j + 1) + j * lda1] = CZERO;
                        i32 n_rot3 = ilastm - j;
                        SLC_ZROT(&n_rot3, &al[j + (j + 1) * lda1], &lda1,
                                 &al[(j + 1) + (j + 1) * lda1], &lda1, &cs, &sn);
                    } else {
                        i32 n_rot2 = ilastm - j + 1;
                        SLC_ZROT(&n_rot2, &al[j + j * lda1], &lda1,
                                 &al[(j + 1) + j * lda1], &lda1, &cs, &sn);
                        temp = al[(j + 1) + (j + 1) * lda1];
                        SLC_ZLARTG(&temp, &al[(j + 1) + j * lda1], &cs, &sn, &al[(j + 1) + (j + 1) * lda1]);
                        al[(j + 1) + j * lda1] = CZERO;
                        i32 n_rot3 = j + 1 - ifrstm;
                        SLC_ZROT(&n_rot3, &al[ifrstm + (j + 1) * lda1], &(i32){1},
                                 &al[ifrstm + j * lda1], &(i32){1}, &cs, &sn);
                        sn = -sn;
                    }
                    if (wantq) {
                        c128 *ql = &q[l * ldq12];
                        c128 snc = conj(sn);
                        SLC_ZROT(&n, &ql[j * ldq1], &(i32){1}, &ql[(j + 1) * ldq1],
                                 &(i32){1}, &cs, &snc);
                    }
                }
                i32 min_j2_ilastm = (j + 2 < ilastm) ? j + 2 : ilastm;
                i32 n_rot_final = min_j2_ilastm - ifrstm + 1;
                c128 snc = conj(sn);
                SLC_ZROT(&n_rot_final, &a1[ifrstm + j * lda1], &(i32){1},
                         &a1[ifrstm + (j + 1) * lda1], &(i32){1}, &cs, &snc);
            }
        }

    L440:
        continue;
    }

    *info = ilast + 1;
    goto L540;

L460:
    for (i32 j = 0; j < ilo - 1; j++) {
        c128 *aj = &a[j + j * lda1];
        ma01bz(base, k, s, aj, lda12, &alpha[j], &beta[j], &scal[j]);
    }

    if (lschr) {
        for (i32 l = k - 1; l >= 1; l--) {
            c128 *al = &a[l * lda12];
            if (s[l] == 1) {
                for (i32 j = 0; j < n; j++) {
                    abst = cabs(al[j + j * lda1]);
                    if (abst > safmin) {
                        temp = conj(al[j + j * lda1] / abst);
                        al[j + j * lda1] = abst;
                        if (j < n - 1) {
                            i32 n_scal = n - j - 1;
                            SLC_ZSCAL(&n_scal, &temp, &al[j + (j + 1) * lda1], &lda1);
                        }
                    } else {
                        temp = CONE;
                    }
                    zwork[j] = temp;
                }
            } else {
                for (i32 j = 0; j < n; j++) {
                    abst = cabs(al[j + j * lda1]);
                    if (abst > safmin) {
                        temp = conj(al[j + j * lda1] / abst);
                        al[j + j * lda1] = abst;
                        i32 n_scal = j;
                        SLC_ZSCAL(&n_scal, &temp, &al[j * lda1], &(i32){1});
                    } else {
                        temp = CONE;
                    }
                    zwork[j] = conj(temp);
                }
            }
            if (wantq) {
                c128 *ql = &q[l * ldq12];
                for (i32 j = 0; j < n; j++) {
                    c128 sc = conj(zwork[j]);
                    SLC_ZSCAL(&n, &sc, &ql[j * ldq1], &(i32){1});
                }
            }
            c128 *alm1 = &a[(l - 1) * lda12];
            if (s[l - 1] == 1) {
                for (i32 j = 0; j < n; j++) {
                    c128 sc = conj(zwork[j]);
                    i32 n_scal = j + 1;
                    SLC_ZSCAL(&n_scal, &sc, &alm1[j * lda1], &(i32){1});
                }
            } else {
                for (i32 j = 0; j < n; j++) {
                    i32 n_scal = n - j;
                    SLC_ZSCAL(&n_scal, &zwork[j], &alm1[j + j * lda1], &lda1);
                }
            }
        }
    }

L540:
    dwork[0] = (f64)n;
    zwork[0] = (c128)n;
}
