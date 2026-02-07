/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>
#include <string.h>

void mb03xz(const char *balanc, const char *job, const char *jobu, i32 n,
            c128 *a, i32 lda, c128 *qg, i32 ldqg,
            c128 *u1, i32 ldu1, c128 *u2, i32 ldu2,
            f64 *wr, f64 *wi, i32 *ilo, f64 *scale,
            f64 *dwork, i32 ldwork, c128 *zwork, i32 lzwork,
            bool *bwork, i32 *info)
{
    const f64 ZERO = 0.0, ONE = 1.0, TWO = 2.0;
    const c128 CZERO = 0.0 + 0.0*I;
    const c128 CONE  = 1.0 + 0.0*I;

    char balanc_up = (char)toupper((unsigned char)balanc[0]);
    char job_up    = (char)toupper((unsigned char)job[0]);
    char jobu_up   = (char)toupper((unsigned char)jobu[0]);

    bool lscal = (balanc_up == 'P') || (balanc_up == 'S') || (balanc_up == 'B');
    bool wantg = (job_up == 'G');
    bool wants = (job_up == 'S') || wantg;
    bool wantu = (jobu_up == 'U');
    bool wantus = wants && wantu;

    i32 nn  = n * n;
    i32 n2  = 2 * n;
    i32 nn2 = n2 * n2;

    i32 k = wants ? n2 : n;

    i32 mindw, minzw, mindb;

    if (n == 0) {
        mindw = 2;
    } else if (wantu) {
        mindb = 4 * nn2 + n2;
        if (wants) {
            i32 m1 = 20 * nn + 12 * n;
            mindw = m1 > 2 ? m1 : 2;
        } else {
            mindw = 20 * nn + 12 * n + 2;
        }
    } else {
        mindb = 2 * nn2 + n2;
        if (wants) {
            i32 m1 = 12 * nn + 4 * n;
            i32 m2 = 8 * nn + 12 * n;
            i32 mx = m1 > m2 ? m1 : m2;
            mindw = mx > 2 ? mx : 2;
        } else {
            i32 m1 = 12 * nn + 4 * n;
            i32 m2 = 8 * nn + 12 * n;
            mindw = (m1 > m2 ? m1 : m2) + 2;
        }
    }

    if (wantg || wantu) {
        minzw = 12 * n - 2 > 1 ? 12 * n - 2 : 1;
    } else if (wants) {
        minzw = 12 * n - 6 > 1 ? 12 * n - 6 : 1;
    } else {
        minzw = 1;
    }

    bool lquery = (ldwork == -1) || (lzwork == -1);

    *info = 0;

    if (!lscal && balanc_up != 'N') {
        *info = -1;
    } else if (!wants && job_up != 'E') {
        *info = -2;
    } else if (!wantu && jobu_up != 'N') {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (lda < (k > 1 ? k : 1)) {
        *info = -6;
    } else if (ldqg < (k > 1 ? k : 1)) {
        *info = -8;
    } else if (ldu1 < 1 || (wantus && ldu1 < n2)) {
        *info = -10;
    } else if (ldu2 < 1 || (wantus && ldu2 < n2)) {
        *info = -12;
    } else if (!lquery) {
        if (ldwork < mindw) {
            dwork[0] = (f64)mindw;
            *info = -18;
        } else if (lzwork < minzw) {
            zwork[0] = (c128)minzw;
            *info = -20;
        }
    }

    if (*info != 0) {
        i32 neg_info = -(*info);
        SLC_XERBLA("MB03XZ", &neg_info);
        return;
    }

    if (n > 0) {
        i32 optzw = minzw;

        if (lquery) {
            i32 mb03xs_info;
            i32 lwork_query = -1;
            f64 work_query;
            mb03xs(jobu, n2, dwork, n2, dwork, n2, dwork, n2, dwork, n2,
                   wi, wr, &work_query, lwork_query, &mb03xs_info);
            i32 optdw = (i32)work_query + mindb;
            if (optdw < mindw) optdw = mindw;
            dwork[0] = (f64)optdw;
            zwork[0] = (c128)optzw;
            return;
        }
    }

    *ilo = 1;
    if (n == 0) {
        dwork[0] = TWO;
        dwork[1] = ZERO;
        zwork[0] = CONE;
        return;
    }

    f64 eps = SLC_DLAMCH("P");
    f64 smlnum = SLC_DLAMCH("S");
    f64 bignum = ONE / smlnum;
    SLC_DLABAD(&smlnum, &bignum);
    smlnum = sqrt(smlnum) / eps;
    bignum = ONE / smlnum;

    f64 hnrm = ma02iz("Hamiltonian", "MaxElement", n, a, lda, qg, ldqg, dwork);

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
        i32 izero = 0, ierr;
        SLC_ZLASCL("General", &izero, &izero, &hnrm, &cscale, &n, &n, a, &lda, &ierr);
        i32 np1 = n + 1;
        SLC_ZLASCL("General", &izero, &izero, &hnrm, &cscale, &n, &np1, qg, &ldqg, &ierr);
    }

    f64 hnr1;
    i32 ilo_bal;
    if (lscal) {
        mb04dz(balanc, n, a, lda, qg, ldqg, &ilo_bal, scale, info);
        *ilo = ilo_bal;
    } else {
        *ilo = 1;
    }

    hnr1 = ma02iz("Hamiltonian", "1-norm", n, a, lda, qg, ldqg, dwork);

    i32 ia, iqg_ofs, iu1_ofs, iu2_ofs, iwrk;
    if (wants) {
        ia = 0;
    } else {
        ia = 2;
    }
    iqg_ofs = ia + nn2;

    if (wantu) {
        iu1_ofs = iqg_ofs + nn2 + n2;
        iu2_ofs = iu1_ofs + nn2;
        iwrk = iu2_ofs + nn2;
    } else {
        iu1_ofs = iqg_ofs;
        iu2_ofs = iqg_ofs;
        iwrk = iqg_ofs + nn2 + n2;
    }

    i32 iw = ia;
    i32 is = iw + n2 * n;

    for (i32 j = 0; j < n; j++) {
        i32 iw1 = iw;
        for (i32 i = 0; i < n; i++) {
            dwork[iw] = cimag(a[i + j * lda]);
            iw++;
        }
        for (i32 i = 0; i < n; i++) {
            dwork[iw] = -creal(a[i + j * lda]);
            dwork[is] = -dwork[iw];
            iw++;
            is++;
        }
        i32 int1 = 1;
        SLC_DCOPY(&n, &dwork[iw1], &int1, &dwork[is], &int1);
        is += n;
    }

    iw = iqg_ofs;
    for (i32 j = 0; j < n + 1; j++) {
        for (i32 i = 0; i < n; i++) {
            dwork[iw] = cimag(qg[i + j * ldqg]);
            iw++;
        }
        iw += j;
        is = iw;
        for (i32 i = j; i < n; i++) {
            dwork[iw] = -creal(qg[i + j * ldqg]);
            dwork[is] = dwork[iw];
            iw++;
            is += n2;
        }
    }

    i32 iw1 = iw;
    i32 i1 = iw;
    for (i32 j = 1; j < n + 1; j++) {
        is = i1;
        i1++;
        for (i32 i = 0; i < j; i++) {
            dwork[iw] = creal(qg[i + j * ldqg]);
            dwork[is] = dwork[iw];
            iw++;
            is += n2;
        }
        iw += n2 - j;
    }

    i32 int1 = 1;
    i32 np1 = n + 1;
    SLC_DLACPY("Full", &n, &np1, &dwork[iqg_ofs], &n2, &dwork[iw1 - n], &n2);

    i32 mb03xs_info;
    mb03xs(jobu, n2, &dwork[ia], n2, &dwork[iqg_ofs], n2,
           &dwork[iu1_ofs], n2, &dwork[iu2_ofs], n2, wi, wr,
           &dwork[iwrk], ldwork - iwrk, &mb03xs_info);

    if (mb03xs_info != 0) {
        *info = mb03xs_info;
        return;
    }

    i32 optdw = (i32)dwork[iwrk] + iwrk;
    if (optdw < mindw) optdw = mindw;

    if (!wants) {
        if (scaleh) {
            i32 izero = 0, ierr;
            SLC_DLASCL("Hessenberg", &izero, &izero, &cscale, &hnrm, &n2, &n2,
                       &dwork[ia], &n2, &ierr);
            if (wantg) {
                i32 iqg_g = iqg_ofs + n2;
                SLC_DLASCL("General", &izero, &izero, &cscale, &hnrm, &n2, &n2,
                           &dwork[iqg_g], &n2, &ierr);
            }
            SLC_DLASCL("General", &izero, &izero, &cscale, &hnrm, &n2, &int1, wr, &n2, &ierr);
            SLC_DLASCL("General", &izero, &izero, &cscale, &hnrm, &n2, &int1, wi, &n2, &ierr);
            hnr1 = hnr1 * hnrm / cscale;
        }
        goto exit_label;
    }

    iw = ia;
    for (i32 j = 0; j < n2; j++) {
        i32 limit = (j + 1 < n2) ? j + 1 : n2 - 1;
        for (i32 i = 0; i <= limit; i++) {
            a[i + j * lda] = ZERO * I + dwork[iw];
            a[i + j * lda] *= I;
            iw++;
        }
        iw += n2 - limit - 1;
    }

    if (wantg) {
        iw = iqg_ofs + n2;
        for (i32 j = 0; j < n2; j++) {
            for (i32 i = 0; i < j; i++) {
                qg[i + j * ldqg] = ZERO * I + dwork[iw];
                qg[i + j * ldqg] *= I;
                iw++;
            }
            qg[j + j * ldqg] = CZERO;
            iw += n2 - j;
        }
    }

    if (wantu) {
        iw = iu1_ofs;
        for (i32 j = 0; j < n2; j++) {
            for (i32 i = 0; i < n2; i++) {
                u1[i + j * ldu1] = (c128)dwork[iw];
                iw++;
            }
        }

        for (i32 j = 0; j < n2; j++) {
            for (i32 i = 0; i < n2; i++) {
                u2[i + j * ldu2] = (c128)dwork[iw];
                iw++;
            }
        }
    }

    i32 nb = 2;
    if (wants) {
        i32 lwork_query = -1;
        c128 work_query;
        i32 qr_info;
        SLC_ZGEQRF(&n2, &n2, zwork, &n2, zwork, &work_query, &lwork_query, &qr_info);
        i32 opt_qr = (i32)creal(work_query);
        if (opt_qr / n2 > 2) {
            nb = opt_qr / n2;
        }
    }

    i32 iev = 0;
    i32 iu = 2;
    i32 izwrk = iu + 4 * (n2 - 1);

    i32 j = 0;
    i32 j2 = (n2 < nb) ? n2 : nb;

    while (j < n2 - 1) {
        f64 nrmb = cabs(a[j + j * lda]) + cabs(a[j + 1 + (j + 1) * lda]);
        if (cabs(a[j + 1 + j * lda]) > nrmb * eps) {
            i32 nc = j2 - j - 2 > 0 ? j2 - j - 2 : 0;
            i32 nc1 = j2 - j > 0 ? j2 - j : 0;
            i32 jm1 = j > 0 ? j : 1;
            i32 jp2 = (j + 2 < n2) ? j + 2 : n2;

            i32 int2 = 2;
            SLC_ZLASET("Full", &int2, &int2, &CZERO, &CONE, &zwork[iu], &int2);

            i32 wantt_int = 1, wantz_int = 1;
            i32 ilo_hqr = 1, ihi_hqr = 2;
            i32 zlahqr_info;
            SLC_ZLAHQR(&wantt_int, &wantz_int, &int2, &ilo_hqr, &ihi_hqr,
                       &a[j + j * lda], &lda, &zwork[iev],
                       &ilo_hqr, &ihi_hqr, &zwork[iu], &int2, &zlahqr_info);

            if (zlahqr_info > 0) {
                *info = n2 + 1;
                return;
            }

            if (j > 0) {
                SLC_ZGEMM("N", "N", &j, &int2, &int2,
                          &CONE, &a[0 + j * lda], &lda, &zwork[iu], &int2,
                          &CZERO, &zwork[izwrk], &jm1);
                SLC_ZLACPY("Full", &j, &int2, &zwork[izwrk], &jm1, &a[0 + j * lda], &lda);
            }

            if (nc > 0) {
                SLC_ZGEMM("C", "N", &int2, &nc, &int2,
                          &CONE, &zwork[iu], &int2, &a[j + jp2 * lda], &lda,
                          &CZERO, &zwork[izwrk], &int2);
                SLC_ZLACPY("Full", &int2, &nc, &zwork[izwrk], &int2, &a[j + jp2 * lda], &lda);
            }

            if (wantg) {
                c128 tmp = qg[j + 1 + j * ldqg];
                qg[j + 1 + j * ldqg] = -qg[j + (j + 1) * ldqg];

                i32 jp1 = j + 1;
                SLC_ZGEMM("N", "N", &jp1, &int2, &int2,
                          &CONE, &qg[0 + j * ldqg], &ldqg, &zwork[iu], &int2,
                          &CZERO, &zwork[izwrk], &jp1);
                SLC_ZLACPY("Full", &jp1, &int2, &zwork[izwrk], &jp1, &qg[0 + j * ldqg], &ldqg);

                if (nc1 > 0) {
                    SLC_ZGEMM("C", "N", &int2, &nc1, &int2,
                              &CONE, &zwork[iu], &int2, &qg[j + j * ldqg], &ldqg,
                              &CZERO, &zwork[izwrk], &int2);
                    SLC_ZLACPY("Full", &int2, &nc1, &zwork[izwrk], &int2, &qg[j + j * ldqg], &ldqg);
                }

                qg[j + 1 + j * ldqg] = tmp;
            }

            if (wantu) {
                SLC_ZGEMM("N", "N", &n2, &int2, &int2,
                          &CONE, &u1[0 + j * ldu1], &ldu1, &zwork[iu], &int2,
                          &CZERO, &zwork[izwrk], &n2);
                SLC_ZLACPY("Full", &n2, &int2, &zwork[izwrk], &n2, &u1[0 + j * ldu1], &ldu1);

                SLC_ZGEMM("N", "N", &n2, &int2, &int2,
                          &CONE, &u2[0 + j * ldu2], &ldu2, &zwork[iu], &int2,
                          &CZERO, &zwork[izwrk], &n2);
                SLC_ZLACPY("Full", &n2, &int2, &zwork[izwrk], &n2, &u2[0 + j * ldu2], &ldu2);
            }

            bwork[j] = true;
            j += 2;
            iu += 4;
        } else {
            bwork[j] = false;
            a[j + 1 + j * lda] = CZERO;
            j++;
        }

        if (j >= j2 && j < n2 - 1) {
            i32 j1 = j2;
            j2 = (n2 < j1 + nb) ? n2 : j1 + nb;
            i32 nc_block = j2 - j1;

            i32 i_block = 0;
            i32 iub = 2;
            while (i_block < j) {
                if (bwork[i_block]) {
                    i32 int2 = 2;
                    SLC_ZGEMM("C", "N", &int2, &nc_block, &int2,
                              &CONE, &zwork[iub], &int2, &a[i_block + j1 * lda], &lda,
                              &CZERO, &zwork[izwrk], &int2);
                    SLC_ZLACPY("Full", &int2, &nc_block, &zwork[izwrk], &int2,
                               &a[i_block + j1 * lda], &lda);

                    if (wantg) {
                        SLC_ZGEMM("C", "N", &int2, &nc_block, &int2,
                                  &CONE, &zwork[iub], &int2, &qg[i_block + j1 * ldqg], &ldqg,
                                  &CZERO, &zwork[izwrk], &int2);
                        SLC_ZLACPY("Full", &int2, &nc_block, &zwork[izwrk], &int2,
                                   &qg[i_block + j1 * ldqg], &ldqg);
                    }

                    iub += 4;
                    i_block += 2;
                } else {
                    i_block++;
                }
            }
        }
    }

    if (scaleh) {
        i32 izero = 0, ierr;
        SLC_ZLASCL("Hessenberg", &izero, &izero, &cscale, &hnrm, &n2, &n2, a, &lda, &ierr);
        if (wantg) {
            SLC_ZLASCL("General", &izero, &izero, &cscale, &hnrm, &n2, &n2, &qg[0 + 1 * ldqg], &ldqg, &ierr);
        }
        SLC_DLASCL("General", &izero, &izero, &cscale, &hnrm, &n2, &int1, wr, &n2, &ierr);
        SLC_DLASCL("General", &izero, &izero, &cscale, &hnrm, &n2, &int1, wi, &n2, &ierr);
        hnr1 = hnr1 * hnrm / cscale;
    }

exit_label:
    dwork[0] = (f64)optdw;
    dwork[1] = hnr1;
    zwork[0] = (c128)minzw;
}
