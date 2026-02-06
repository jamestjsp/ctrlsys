// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void mb03xd(const char *balanc, const char *job, const char *jobu,
            const char *jobv, i32 n, f64 *a, i32 lda, f64 *qg, i32 ldqg,
            f64 *t, i32 ldt, f64 *u1, i32 ldu1, f64 *u2, i32 ldu2,
            f64 *v1, i32 ldv1, f64 *v2, i32 ldv2,
            f64 *wr, f64 *wi, i32 *ilo, f64 *scale,
            f64 *dwork, i32 ldwork, i32 *info) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 NEGONE = -1.0;

    bool lperm = (balanc[0] == 'P' || balanc[0] == 'p' ||
                  balanc[0] == 'B' || balanc[0] == 'b');
    bool lscal = (balanc[0] == 'S' || balanc[0] == 's' ||
                  balanc[0] == 'B' || balanc[0] == 'b');
    bool wants = (job[0] == 'S' || job[0] == 's' ||
                  job[0] == 'G' || job[0] == 'g');
    bool wantg = (job[0] == 'G' || job[0] == 'g');
    bool wantu = (jobu[0] == 'U' || jobu[0] == 'u');
    bool wantv = (jobv[0] == 'V' || jobv[0] == 'v');

    *info = 0;

    i32 wrkmin;
    if (wantg) {
        if (wantu && wantv) {
            wrkmin = (2 > 7 * n + n * n) ? 2 : 7 * n + n * n;
        } else if (!wantu && !wantv) {
            i32 v1 = 7 * n + n * n;
            i32 v2 = 2 * n + 3 * n * n;
            wrkmin = 2;
            if (v1 > wrkmin) wrkmin = v1;
            if (v2 > wrkmin) wrkmin = v2;
        } else {
            wrkmin = (2 > 7 * n + 2 * n * n) ? 2 : 7 * n + 2 * n * n;
        }
    } else {
        if (!wantu && !wantv) {
            wrkmin = (2 > 7 * n + n * n) ? 2 : 7 * n + n * n;
        } else {
            wrkmin = (2 > 8 * n) ? 2 : 8 * n;
        }
    }

    bool balanc_n = (balanc[0] == 'N' || balanc[0] == 'n');
    bool job_e = (job[0] == 'E' || job[0] == 'e');
    bool jobu_n = (jobu[0] == 'N' || jobu[0] == 'n');
    bool jobv_n = (jobv[0] == 'N' || jobv[0] == 'n');

    if (!lperm && !lscal && !balanc_n) {
        *info = -1;
    } else if (!wants && !job_e) {
        *info = -2;
    } else if (!wantu && !jobu_n) {
        *info = -3;
    } else if (!wantv && !jobv_n) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -7;
    } else if (ldqg < (1 > n ? 1 : n)) {
        *info = -9;
    } else if (ldt < (1 > n ? 1 : n)) {
        *info = -11;
    } else if (ldu1 < 1 || (wantu && ldu1 < n)) {
        *info = -13;
    } else if (ldu2 < 1 || (wantu && ldu2 < n)) {
        *info = -15;
    } else if (ldv1 < 1 || (wantv && ldv1 < n)) {
        *info = -17;
    } else if (ldv2 < 1 || (wantv && ldv2 < n)) {
        *info = -19;
    } else {
        bool lquery = (ldwork == -1);
        if (lquery) {
            i32 wrkopt = wrkmin;
            dwork[0] = (f64)wrkopt;
            return;
        } else if (ldwork < wrkmin) {
            dwork[0] = (f64)wrkmin;
            *info = -25;
        }
    }

    if (*info != 0) {
        return;
    }

    *ilo = 0;
    if (n == 0) {
        dwork[0] = TWO;
        dwork[1] = ZERO;
        return;
    }

    f64 eps = SLC_DLAMCH("P");
    f64 smlnum = SLC_DLAMCH("S");
    f64 bignum = ONE / smlnum;
    SLC_DLABAD(&smlnum, &bignum);
    smlnum = sqrt(smlnum) / eps;
    bignum = ONE / smlnum;

    f64 hnrm = ma02id("H", "M", n, a, lda, qg, ldqg, dwork);
    bool scaleh = false;
    f64 cscale = ONE;
    if (hnrm > ZERO && hnrm < smlnum) {
        scaleh = true;
        cscale = smlnum;
    } else if (hnrm > bignum) {
        scaleh = true;
        cscale = bignum;
    }
    if (scaleh) {
        i32 izero = 0;
        SLC_DLASCL("G", &izero, &izero, &hnrm, &cscale, &n, &n, a, &lda, info);
        i32 np1 = n + 1;
        SLC_DLASCL("G", &izero, &izero, &hnrm, &cscale, &n, &np1, qg, &ldqg, info);
    }

    i32 ierr;
    if (lperm || lscal) {
        mb04dd(balanc, n, a, lda, qg, ldqg, ilo, scale, &ierr);
    } else {
        *ilo = 1;
    }
    f64 hnr1 = ma02id("H", "1", n, a, lda, qg, ldqg, dwork);

    SLC_DLACPY("A", &n, &n, a, &lda, t, &ldt);
    i32 izero = 0;
    SLC_DLASCL("G", &izero, &izero, &ONE, &NEGONE, &n, &n, a, &lda, &ierr);

    i32 pcsl = 0;
    i32 pcsr = pcsl + 2 * n;
    i32 ptaul = pcsr + 2 * n;
    i32 ptaur = ptaul + n;
    i32 pdw = ptaur + n;

    i32 pq = 0;
    if (!wantu && !wantv) {
        pq = pdw;
        pdw = pdw + n * n;
        for (i32 j = 0; j < n; j++) {
            i32 k = pq + (n + 1) * j;
            i32 l = k;
            dwork[k] = qg[j + j * ldqg];
            for (i32 i = j + 1; i < n; i++) {
                k++;
                l += n;
                f64 temp = qg[i + j * ldqg];
                dwork[k] = temp;
                dwork[l] = temp;
            }
        }
    } else if (wantu) {
        for (i32 j = 0; j < n; j++) {
            u2[j + j * ldu2] = qg[j + j * ldqg];
            for (i32 i = j + 1; i < n; i++) {
                f64 temp = qg[i + j * ldqg];
                u2[i + j * ldu2] = temp;
                u2[j + i * ldu2] = temp;
            }
        }
    } else {
        for (i32 j = 0; j < n; j++) {
            v2[j + j * ldv2] = qg[j + j * ldqg];
            for (i32 i = j + 1; i < n; i++) {
                f64 temp = qg[i + j * ldqg];
                v2[i + j * ldv2] = temp;
                v2[j + i * ldv2] = temp;
            }
        }
    }

    for (i32 j = 0; j < n; j++) {
        for (i32 i = j + 1; i < n; i++) {
            qg[i + (j + 1) * ldqg] = qg[j + (i + 1) * ldqg];
        }
    }

    i32 ilo_c = *ilo - 1;
    i32 ldwork_rem = ldwork - pdw;

    if (!wantu && !wantv) {
        mb04tb("N", "T", n, *ilo, t, ldt, a, lda, &qg[ldqg], ldqg,
               &dwork[pq], n, &dwork[pcsl], &dwork[pcsr],
               &dwork[ptaul], &dwork[ptaur], &dwork[pdw], ldwork_rem, &ierr);
    } else if (wantu) {
        mb04tb("N", "T", n, *ilo, t, ldt, a, lda, &qg[ldqg], ldqg,
               u2, ldu2, &dwork[pcsl], &dwork[pcsr],
               &dwork[ptaul], &dwork[ptaur], &dwork[pdw], ldwork_rem, &ierr);
    } else {
        mb04tb("N", "T", n, *ilo, t, ldt, a, lda, &qg[ldqg], ldqg,
               v2, ldv2, &dwork[pcsl], &dwork[pcsr],
               &dwork[ptaul], &dwork[ptaur], &dwork[pdw], ldwork_rem, &ierr);
    }

    if (wantu && !wantv && !wantg) {
        if (n > 1) {
            i32 nm1 = n - 1;
            SLC_DLACPY("L", &nm1, &nm1, &t[1], &ldt, &qg[1], &ldqg);
        }
    } else if (!wantu && wantv && !wantg) {
        if (n > 1) {
            i32 nm1 = n - 1;
            SLC_DLACPY("L", &nm1, &nm1, &a[1], &lda, &qg[1], &ldqg);
            SLC_DLACPY("U", &nm1, &nm1, &v2[ldv2], &ldv2, &qg[ldqg], &ldqg);
        }
    } else if (wantu && wantv && !wantg) {
        if (n > 1) {
            i32 nm1 = n - 1;
            SLC_DLACPY("L", &nm1, &nm1, &t[1], &ldt, &v2[1], &ldv2);
            SLC_DLACPY("L", &nm1, &nm1, &a[1], &lda, &qg[1], &ldqg);
        }
    } else if (wantu && !wantv && wantg) {
        if (n > 1) {
            i32 nm1 = n - 1;
            SLC_DLACPY("L", &nm1, &nm1, &t[1], &ldt, &dwork[pdw + n * n + n], &nm1);
        }
    } else if (!wantu && wantv && wantg) {
        if (n > 2) {
            i32 nm2 = n - 2;
            SLC_DLACPY("L", &nm2, &nm2, &a[2], &lda, &dwork[pdw + n * n + n], &nm2);
        }
    } else if (wantu && wantv && wantg) {
        if (n > 1) {
            i32 nm1 = n - 1;
            SLC_DLACPY("L", &nm1, &nm1, &t[1], &ldt, &dwork[pdw + n], &nm1);
        }
        if (n > 2) {
            i32 nm2 = n - 2;
            SLC_DLACPY("L", &nm2, &nm2, &a[2], &lda, &v2[2], &ldv2);
        }
    }

    i32 pbeta;
    if (!wantu && !wantv) {
        pbeta = 0;
    } else {
        pbeta = pdw;
    }

    if (n > 2) {
        i32 nm2 = n - 2;
        SLC_DLASET("L", &nm2, &nm2, &ZERO, &ZERO, &a[2], &lda);
    }
    if (n > 1) {
        i32 nm1 = n - 1;
        SLC_DLASET("L", &nm1, &nm1, &ZERO, &ZERO, &t[1], &ldt);
    }

    if (!wantg) {
        pdw = pbeta + n;
        const char *uchar = wantu ? "I" : "N";
        const char *vchar = wantv ? "I" : "N";

        mb03xp(job, vchar, uchar, n, *ilo, n, a, lda, t, ldt, v1, ldv1,
               u1, ldu1, wr, wi, &dwork[pbeta], &dwork[pdw], ldwork - pdw, info);
        if (*info != 0) goto label_90;
    } else if (!wantu && !wantv) {
        pq = pbeta + n;
        i32 pz = pq + n * n;
        pdw = pz + n * n;

        mb03xp("S", "I", "I", n, *ilo, n, a, lda, t, ldt, &dwork[pq], n,
               &dwork[pz], n, wr, wi, &dwork[pbeta], &dwork[pdw], ldwork - pdw, info);
        if (*info != 0) goto label_90;

        SLC_DGEMM("T", "N", &n, &n, &n, &ONE, &dwork[pz], &n, &qg[ldqg], &ldqg,
                  &ZERO, &dwork[pdw], &n);
        SLC_DGEMM("N", "N", &n, &n, &n, &ONE, &dwork[pdw], &n, &dwork[pq], &n,
                  &ZERO, &qg[ldqg], &ldqg);
    } else if (wantu && !wantv) {
        pq = pbeta + n;
        pdw = pq + n * n;

        mb03xp("S", "I", "I", n, *ilo, n, a, lda, t, ldt, &dwork[pq], n,
               u1, ldu1, wr, wi, &dwork[pbeta], &dwork[pdw + (n - 1) * (n - 1)],
               ldwork - pdw - (n - 1) * (n - 1), info);
        if (*info != 0) goto label_90;

        if (n > 1) {
            i32 nm1 = n - 1;
            SLC_DLACPY("L", &nm1, &nm1, &dwork[pdw], &nm1, &t[1], &ldt);
        }

        SLC_DGEMM("T", "N", &n, &n, &n, &ONE, u1, &ldu1, &qg[ldqg], &ldqg,
                  &ZERO, &dwork[pdw], &n);
        SLC_DGEMM("N", "N", &n, &n, &n, &ONE, &dwork[pdw], &n, &dwork[pq], &n,
                  &ZERO, &qg[ldqg], &ldqg);
    } else if (!wantu && wantv) {
        i32 pz = pbeta + n;
        pdw = pz + n * n;

        mb03xp("S", "I", "I", n, *ilo, n, a, lda, t, ldt, v1, ldv1,
               &dwork[pz], n, wr, wi, &dwork[pbeta], &dwork[pdw + (n - 1) * (n - 1)],
               ldwork - pdw - (n - 1) * (n - 1), info);
        if (*info != 0) goto label_90;

        if (n > 2) {
            i32 nm2 = n - 2;
            SLC_DLACPY("L", &nm2, &nm2, &dwork[pdw], &nm2, &a[2], &lda);
        }

        SLC_DGEMM("T", "N", &n, &n, &n, &ONE, &dwork[pz], &n, &qg[ldqg], &ldqg,
                  &ZERO, &dwork[pdw], &n);
        SLC_DGEMM("N", "N", &n, &n, &n, &ONE, &dwork[pdw], &n, v1, &ldv1,
                  &ZERO, &qg[ldqg], &ldqg);
    } else {
        pdw = pbeta + n;

        mb03xp("S", "I", "I", n, *ilo, n, a, lda, t, ldt, v1, ldv1,
               u1, ldu1, wr, wi, &dwork[pbeta], &dwork[pdw + (n - 1) * (n - 1)],
               ldwork - pdw - (n - 1) * (n - 1), info);
        if (*info != 0) goto label_90;

        if (n > 1) {
            i32 nm1 = n - 1;
            SLC_DLACPY("L", &nm1, &nm1, &dwork[pdw], &nm1, &t[1], &ldt);
        }
        if (n > 2) {
            i32 nm2 = n - 2;
            SLC_DLACPY("L", &nm2, &nm2, &v2[2], &ldv2, &a[2], &lda);
        }

        SLC_DGEMM("T", "N", &n, &n, &n, &ONE, u1, &ldu1, &qg[ldqg], &ldqg,
                  &ZERO, &dwork[pdw], &n);
        SLC_DGEMM("N", "N", &n, &n, &n, &ONE, &dwork[pdw], &n, v1, &ldv1,
                  &ZERO, &qg[ldqg], &ldqg);
    }

label_90:
    for (i32 i = *info; i < n; ) {
        f64 tempr = wr[i];
        f64 tempi = wi[i];
        f64 temp = dwork[pbeta + i];
        if (temp > ZERO) tempr = -tempr;
        temp = fabs(temp);

        if (tempi == ZERO) {
            if (tempr < ZERO) {
                wr[i] = ZERO;
                wi[i] = sqrt(temp) * sqrt(-tempr);
            } else {
                wr[i] = sqrt(temp) * sqrt(tempr);
                wi[i] = ZERO;
            }
            i++;
        } else {
            ma01ad(tempr, tempi, &wr[i], &wi[i]);
            wr[i] = wr[i] * sqrt(temp);
            if (temp > ZERO) {
                wi[i] = wi[i] * sqrt(temp);
            } else {
                wi[i] = ZERO;
            }
            wr[i + 1] = -wr[i];
            wi[i + 1] = wi[i];
            i += 2;
        }
    }

    if (scaleh) {
        i32 izero = 0;
        SLC_DLASCL("H", &izero, &izero, &cscale, &hnrm, &n, &n, a, &lda, &ierr);
        SLC_DLASCL("U", &izero, &izero, &cscale, &hnrm, &n, &n, t, &ldt, &ierr);
        if (wantg) {
            SLC_DLASCL("G", &izero, &izero, &cscale, &hnrm, &n, &n, &qg[ldqg], &ldqg, &ierr);
        }
        i32 one = 1;
        SLC_DLASCL("G", &izero, &izero, &cscale, &hnrm, &n, &one, wr, &n, &ierr);
        SLC_DLASCL("G", &izero, &izero, &cscale, &hnrm, &n, &one, wi, &n, &ierr);
        hnr1 = hnr1 * hnrm / cscale;
    }

    if (*info != 0) return;

    if (*ilo > n) {
        dwork[0] = (f64)ldwork;
        dwork[1] = hnr1;
        return;
    }

    if (wantu) {
        i32 inc2 = 2;
        SLC_DSCAL(&n, &NEGONE, &dwork[pcsl + 1], &inc2);
    }
    if (wantv && n > 1) {
        i32 nm1 = n - 1;
        i32 inc2 = 2;
        SLC_DSCAL(&nm1, &NEGONE, &dwork[pcsr + 1], &inc2);
    }

    i32 ilo1 = (*ilo + 1 < n) ? *ilo + 1 : n;

    if (wantu && !wantv && !wantg) {
        pdw = ptaur;
        i32 int1 = 1;
        i32 ldtp1 = ldt + 1;
        SLC_DCOPY(&n, t, &ldtp1, &dwork[pdw], &int1);
        SLC_DLACPY("L", &n, &n, u2, &ldu2, t, &ldt);
        SLC_DLASET("A", &n, &n, &ZERO, &ZERO, u2, &ldu2);

        i32 nilo = n - *ilo + 1;
        mb04qb("N", "N", "N", "C", "C", nilo, n, nilo,
               &qg[(*ilo - 1) + (*ilo - 1) * ldqg], ldqg,
               &t[(*ilo - 1) + (*ilo - 1) * ldt], ldt,
               &u1[(*ilo - 1)], ldu1, &u2[(*ilo - 1)], ldu2,
               &dwork[pcsl + 2 * (*ilo - 1)], &dwork[ptaul + *ilo - 1],
               &dwork[pdw + n], ldwork - pdw - n, &ierr);

        SLC_DCOPY(&n, &dwork[pdw], &int1, t, &ldtp1);
        if (n > 1) {
            i32 nm1 = n - 1;
            SLC_DLASET("L", &nm1, &nm1, &ZERO, &ZERO, &t[1], &ldt);
        }
    } else if (!wantu && wantv && !wantg) {
        pdw = ptaur + n;
        SLC_DLASET("A", &n, &n, &ZERO, &ZERO, v2, &ldv2);

        i32 nilo = n - *ilo;
        if (nilo > 0) {
            mb04qb("N", "N", "N", "C", "R", nilo, n, nilo,
                   &qg[(ilo1 - 1) + (*ilo - 1) * ldqg], ldqg,
                   &qg[(*ilo - 1) + (ilo1 - 1) * ldqg], ldqg,
                   &v1[(ilo1 - 1)], ldv1, &v2[(ilo1 - 1)], ldv2,
                   &dwork[pcsr + 2 * (*ilo - 1)], &dwork[ptaur + *ilo - 1],
                   &dwork[pdw], ldwork - pdw, &ierr);
        }
    } else if (wantu && wantv && !wantg) {
        pdw = ptaur + n;
        i32 int1 = 1;
        i32 ldtp1 = ldt + 1;
        SLC_DCOPY(&n, t, &ldtp1, &dwork[pdw], &int1);
        SLC_DLACPY("L", &n, &n, v2, &ldv2, t, &ldt);
        SLC_DLASET("A", &n, &n, &ZERO, &ZERO, v2, &ldv2);

        i32 nilo = n - *ilo;
        if (nilo > 0) {
            mb04qb("N", "N", "N", "C", "R", nilo, n, nilo,
                   &qg[(ilo1 - 1) + (*ilo - 1) * ldqg], ldqg,
                   &u2[(*ilo - 1) + (ilo1 - 1) * ldu2], ldu2,
                   &v1[(ilo1 - 1)], ldv1, &v2[(ilo1 - 1)], ldv2,
                   &dwork[pcsr + 2 * (*ilo - 1)], &dwork[ptaur + *ilo - 1],
                   &dwork[pdw + n], ldwork - pdw - n, &ierr);
        }

        SLC_DLACPY("L", &n, &n, u2, &ldu2, qg, &ldqg);
        SLC_DLASET("A", &n, &n, &ZERO, &ZERO, u2, &ldu2);

        i32 niloq = n - *ilo + 1;
        mb04qb("N", "N", "N", "C", "C", niloq, n, niloq,
               &t[(*ilo - 1) + (*ilo - 1) * ldt], ldt,
               &qg[(*ilo - 1) + (*ilo - 1) * ldqg], ldqg,
               &u1[(*ilo - 1)], ldu1, &u2[(*ilo - 1)], ldu2,
               &dwork[pcsl + 2 * (*ilo - 1)], &dwork[ptaul + *ilo - 1],
               &dwork[pdw + n], ldwork - pdw - n, &ierr);

        SLC_DCOPY(&n, &dwork[pdw], &int1, t, &ldtp1);
        if (n > 1) {
            i32 nm1 = n - 1;
            SLC_DLASET("L", &nm1, &nm1, &ZERO, &ZERO, &t[1], &ldt);
        }
    } else if (wantu && !wantv && wantg) {
        pq = ptaur;
        pdw = pq + n * n;
        SLC_DLACPY("L", &n, &n, u2, &ldu2, &dwork[pq], &n);
        SLC_DLASET("A", &n, &n, &ZERO, &ZERO, u2, &ldu2);

        i32 niloq = n - *ilo + 1;
        mb04qb("N", "N", "N", "C", "C", niloq, n, niloq,
               &t[(*ilo - 1) + (*ilo - 1) * ldt], ldt,
               &dwork[pq + (*ilo - 1) * (n + 1)], n,
               &u1[(*ilo - 1)], ldu1, &u2[(*ilo - 1)], ldu2,
               &dwork[pcsl + 2 * (*ilo - 1)], &dwork[ptaul + *ilo - 1],
               &dwork[pdw], ldwork - pdw, &ierr);

        if (n > 1) {
            i32 nm1 = n - 1;
            SLC_DLASET("L", &nm1, &nm1, &ZERO, &ZERO, &t[1], &ldt);
        }
    } else if (!wantu && wantv && wantg) {
        pq = ptaur + n;
        pdw = pq + n * n;
        SLC_DLACPY("U", &n, &n, v2, &ldv2, &dwork[pq], &n);
        SLC_DLASET("A", &n, &n, &ZERO, &ZERO, v2, &ldv2);

        i32 nilo = n - *ilo;
        if (nilo > 0) {
            mb04qb("N", "N", "N", "C", "R", nilo, n, nilo,
                   &a[(ilo1 - 1) + (*ilo - 1) * lda], lda,
                   &dwork[pq + (*ilo) * n + (*ilo - 1)], n,
                   &v1[(ilo1 - 1)], ldv1, &v2[(ilo1 - 1)], ldv2,
                   &dwork[pcsr + 2 * (*ilo - 1)], &dwork[ptaur + *ilo - 1],
                   &dwork[pdw + n], ldwork - pdw - n, &ierr);
        }

        if (n > 2) {
            i32 nm2 = n - 2;
            SLC_DLASET("L", &nm2, &nm2, &ZERO, &ZERO, &a[2], &lda);
        }
    } else if (wantu && wantv && wantg) {
        pdw = ptaur + n;
        SLC_DLASET("A", &n, &n, &ZERO, &ZERO, v2, &ldv2);

        i32 nilo = n - *ilo;
        if (nilo > 0) {
            mb04qb("N", "N", "N", "C", "R", nilo, n, nilo,
                   &a[(ilo1 - 1) + (*ilo - 1) * lda], lda,
                   &u2[(*ilo - 1) + (ilo1 - 1) * ldu2], ldu2,
                   &v1[(ilo1 - 1)], ldv1, &v2[(ilo1 - 1)], ldv2,
                   &dwork[pcsr + 2 * (*ilo - 1)], &dwork[ptaur + *ilo - 1],
                   &dwork[pdw], ldwork - pdw, &ierr);
        }

        pq = ptaur;
        pdw = pq + n * n;
        SLC_DLACPY("L", &n, &n, u2, &ldu2, &dwork[pq], &n);
        SLC_DLASET("A", &n, &n, &ZERO, &ZERO, u2, &ldu2);

        i32 niloq = n - *ilo + 1;
        mb04qb("N", "N", "N", "C", "C", niloq, n, niloq,
               &t[(*ilo - 1) + (*ilo - 1) * ldt], ldt,
               &dwork[pq + (*ilo - 1) * (n + 1)], n,
               &u1[(*ilo - 1)], ldu1, &u2[(*ilo - 1)], ldu2,
               &dwork[pcsl + 2 * (*ilo - 1)], &dwork[ptaul + *ilo - 1],
               &dwork[pdw], ldwork - pdw, &ierr);

        if (n > 2) {
            i32 nm2 = n - 2;
            SLC_DLASET("L", &nm2, &nm2, &ZERO, &ZERO, &a[2], &lda);
        }
        if (n > 1) {
            i32 nm1 = n - 1;
            SLC_DLASET("L", &nm1, &nm1, &ZERO, &ZERO, &t[1], &ldt);
        }
    }

    dwork[0] = (f64)ldwork;
    dwork[1] = hnr1;
}
