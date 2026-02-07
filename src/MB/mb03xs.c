/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * MB03XS - Eigenvalues and real skew-Hamiltonian Schur form
 *
 * Computes eigenvalues and real skew-Hamiltonian Schur form of a
 * skew-Hamiltonian matrix W = [[A, G], [Q, A^T]] where G, Q are
 * skew-symmetric.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <string.h>

void mb03xs(const char *jobu, i32 n, f64 *a, i32 lda,
            f64 *qg, i32 ldqg,
            f64 *u1, i32 ldu1, f64 *u2, i32 ldu2,
            f64 *wr, f64 *wi,
            f64 *dwork, i32 ldwork, i32 *info) {
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    bool compu = (jobu[0] == 'U' || jobu[0] == 'u');
    bool lquery = (ldwork == -1);

    i32 nn = n * n;
    i32 wrkmin, wrkopt;
    i32 ilo = 1;
    i32 ierr = 0;

    *info = 0;

    if (compu) {
        wrkmin = (nn + 5 * n > 1) ? nn + 5 * n : 1;
    } else {
        i32 tmp1 = 5 * n;
        i32 tmp2 = nn + n;
        wrkmin = (tmp1 > tmp2) ? tmp1 : tmp2;
        if (wrkmin < 1) wrkmin = 1;
    }
    wrkopt = wrkmin;

    if (!compu && !(jobu[0] == 'N' || jobu[0] == 'n')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    } else if (ldqg < (n > 1 ? n : 1)) {
        *info = -6;
    } else if (ldu1 < 1 || (compu && ldu1 < n)) {
        *info = -8;
    } else if (ldu2 < 1 || (compu && ldu2 < n)) {
        *info = -10;
    } else if (!lquery && ldwork < wrkmin) {
        dwork[0] = (f64)wrkmin;
        *info = -14;
    }

    if (*info == 0 && lquery) {
        if (n == 0) {
            dwork[0] = ONE;
        } else {
            i32 itmp = 4 * n;

            mb04rb(n, 1, a, lda, qg, ldqg, dwork, dwork, dwork, -1, &ierr);
            i32 opt_mb04rb = (i32)dwork[0] + itmp;
            wrkopt = (wrkopt > opt_mb04rb) ? wrkopt : opt_mb04rb;

            SLC_DHSEQR("S", "I", &n, &(i32){1}, &n, a, &lda, wr, wi, dwork, &n, dwork, &(i32){-1}, &ierr);
            if (compu) {
                itmp = itmp + nn;
                i32 opt_dhseqr = (i32)dwork[0] + itmp;
                wrkopt = (wrkopt > opt_dhseqr) ? wrkopt : opt_dhseqr;

                mb04qs("N", "N", "N", n, n, 1, dwork, n, qg, ldqg,
                       u1, ldu1, u2, ldu2, dwork, dwork, dwork, -1, &ierr);
            } else {
                itmp = nn;
                i32 opt_tmp = 2 * nn - n;
                wrkopt = (wrkopt > opt_tmp) ? wrkopt : opt_tmp;
            }
            i32 opt_final = (i32)dwork[0] + itmp;
            wrkopt = (wrkopt > opt_final) ? wrkopt : opt_final;
            wrkopt = (wrkopt > wrkmin) ? wrkopt : wrkmin;
            dwork[0] = (f64)wrkopt;
        }
        return;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        dwork[0] = ONE;
        return;
    }

    f64 eps = SLC_DLAMCH("P");
    f64 smlnum = SLC_DLAMCH("S");
    f64 bignum = ONE / smlnum;
    SLC_DLABAD(&smlnum, &bignum);
    smlnum = sqrt(smlnum) / eps;
    bignum = ONE / smlnum;

    f64 wnrm = ma02id("Skew-Hamiltonian", "Max-Norm", n, a, lda, qg, ldqg, dwork);
    bool scalew = false;
    f64 cscale = ONE;

    if (wnrm > ZERO && wnrm < smlnum) {
        scalew = true;
        cscale = smlnum;
    } else if (wnrm > bignum) {
        scalew = true;
        cscale = bignum;
    }

    if (scalew) {
        SLC_DLASCL("G", &(i32){0}, &(i32){0}, &wnrm, &cscale, &n, &n, a, &lda, &ierr);
        if (n > 1) {
            i32 nm1 = n - 1;
            SLC_DLASCL("L", &(i32){0}, &(i32){0}, &wnrm, &cscale, &nm1, &nm1, &qg[1], &ldqg, &ierr);
            SLC_DLASCL("U", &(i32){0}, &(i32){0}, &wnrm, &cscale, &nm1, &nm1, &qg[0 + 2 * ldqg], &ldqg, &ierr);
        }
    }

    i32 pbal = 0;
    mb04ds("Permute", n, a, lda, qg, ldqg, &ilo, &dwork[pbal], &ierr);

    i32 pcs = n + pbal;
    i32 ptau = 2 * n + pcs;
    i32 pdw = n + ptau;
    i32 ldwork_remaining = ldwork - pdw;
    mb04rb(n, ilo, a, lda, qg, ldqg, &dwork[pcs], &dwork[ptau], &dwork[pdw], ldwork_remaining, &ierr);
    i32 opt_tmp = (i32)dwork[pdw] + pdw;
    wrkopt = (wrkopt > opt_tmp) ? wrkopt : opt_tmp;

    if (compu) {
        i32 pho = pdw;
        pdw = pdw + nn;
        SLC_DLACPY("L", &n, &n, a, &lda, &dwork[pho], &n);

        i32 ilo_min = (ilo < n) ? ilo : n;
        ldwork_remaining = ldwork - pdw;
        SLC_DHSEQR("S", "I", &n, &ilo_min, &n, a, &lda, wr, wi, u1, &ldu1, &dwork[pdw], &ldwork_remaining, info);
        opt_tmp = (i32)dwork[pdw] + pdw;
        wrkopt = (wrkopt > opt_tmp) ? wrkopt : opt_tmp;

        mb01ld("Upper", "Transpose", n, n, ZERO, ONE, &qg[ldqg], ldqg, u1, ldu1, &qg[ldqg], ldqg, u2, nn, &ierr);

        if (n > 1) {
            for (i32 i = 1; i < n; i += 2) {
                dwork[pcs + i] = -dwork[pcs + i];
            }
        }

        SLC_DLASET("A", &n, &n, &ZERO, &ZERO, u2, &ldu2);

        ldwork_remaining = ldwork - pdw;
        mb04qs("N", "N", "N", n, n, ilo, &dwork[pho], n, qg, ldqg,
               u1, ldu1, u2, ldu2, &dwork[pcs], &dwork[ptau], &dwork[pdw], ldwork_remaining, &ierr);
        opt_tmp = (i32)dwork[pdw] + pdw;
        wrkopt = (wrkopt > opt_tmp) ? wrkopt : opt_tmp;

        if (n > 1) {
            i32 nm1 = n - 1;
            SLC_DLASET("L", &nm1, &nm1, &ZERO, &ZERO, &qg[1], &ldqg);
        }

        mb04di("Permute", "Positive", n, ilo, &dwork[pbal], n, u1, ldu1, u2, ldu2, &ierr);
    } else {
        i32 pdv = 0;
        pdw = nn;
        i32 ilo_min = (ilo < n) ? ilo : n;
        ldwork_remaining = ldwork - pdw;
        SLC_DHSEQR("S", "I", &n, &ilo_min, &n, a, &lda, wr, wi, &dwork[pdv], &n, &dwork[pdw], &ldwork_remaining, info);
        opt_tmp = (i32)dwork[pdw] + pdw;
        wrkopt = (wrkopt > opt_tmp) ? wrkopt : opt_tmp;

        ldwork_remaining = ldwork - pdw;
        mb01ld("Upper", "Transpose", n, n, ZERO, ONE, &qg[ldqg], ldqg, &dwork[pdv], n, &qg[ldqg], ldqg, &dwork[pdw], ldwork_remaining, &ierr);
        i32 opt_2nn = 2 * nn - n;
        wrkopt = (wrkopt > opt_2nn) ? wrkopt : opt_2nn;

        if (n > 1) {
            i32 nm1 = n - 1;
            SLC_DLASET("L", &nm1, &nm1, &ZERO, &ZERO, &qg[1], &ldqg);
        }
    }

    if (scalew) {
        SLC_DLASCL("H", &(i32){0}, &(i32){0}, &cscale, &wnrm, &n, &n, a, &lda, &ierr);
        if (n > 1) {
            i32 nm1 = n - 1;
            SLC_DLASCL("U", &(i32){0}, &(i32){0}, &cscale, &wnrm, &nm1, &nm1, &qg[0 + 2 * ldqg], &ldqg, &ierr);
        }

        i32 ldap1 = lda + 1;
        SLC_DCOPY(&n, a, &ldap1, wr, &(i32){1});

        if (cscale == smlnum) {
            i32 i1, i2, inxt;
            if (*info > 0) {
                i1 = *info + 1;
                if (ilo > 1) {
                    i32 ilo_m1 = ilo - 1;
                    i32 ld_wi = (ilo_m1 > 1) ? ilo_m1 : 1;
                    SLC_DLASCL("G", &(i32){0}, &(i32){0}, &cscale, &wnrm, &ilo_m1, &(i32){1}, wi, &ld_wi, &ierr);
                }
            } else {
                i1 = ilo;
            }
            i2 = n - 1;
            inxt = i1 - 1;

            for (i32 i = i1 - 1; i < i2; i++) {
                if (i < inxt) continue;

                if (wi[i] == ZERO) {
                    inxt = i + 1;
                } else {
                    if (a[(i + 1) + i * lda] == ZERO) {
                        wi[i] = ZERO;
                        wi[i + 1] = ZERO;
                    } else if (a[i + (i + 1) * lda] == ZERO) {
                        wi[i] = ZERO;
                        wi[i + 1] = ZERO;

                        if (i > 0) {
                            SLC_DSWAP(&i, &a[0 + i * lda], &(i32){1}, &a[0 + (i + 1) * lda], &(i32){1});
                        }
                        if (n > i + 2) {
                            i32 nm_i_2 = n - i - 2;
                            SLC_DSWAP(&nm_i_2, &a[i + (i + 2) * lda], &lda, &a[(i + 1) + (i + 2) * lda], &lda);
                        }

                        a[i + (i + 1) * lda] = a[(i + 1) + i * lda];
                        a[(i + 1) + i * lda] = ZERO;

                        if (i > 0) {
                            SLC_DSWAP(&i, &qg[0 + (i + 2) * ldqg], &(i32){1}, &qg[0 + (i + 1) * ldqg], &(i32){1});
                        }
                        if (n > i + 2) {
                            i32 nm_i_2 = n - i - 2;
                            SLC_DSWAP(&nm_i_2, &qg[(i + 1) + (i + 3) * ldqg], &ldqg, &qg[i + (i + 3) * ldqg], &ldqg);
                        }
                        qg[i + (i + 2) * ldqg] = -qg[i + (i + 2) * ldqg];

                        if (compu) {
                            SLC_DSWAP(&n, &u1[0 + i * ldu1], &(i32){1}, &u1[0 + (i + 1) * ldu1], &(i32){1});
                            SLC_DSWAP(&n, &u2[0 + i * ldu2], &(i32){1}, &u2[0 + (i + 1) * ldu2], &(i32){1});
                        }
                    }
                    inxt = i + 2;
                }
            }
        }

        i32 n_info = n - *info;
        i32 ld_wi = (n_info > 1) ? n_info : 1;
        SLC_DLASCL("G", &(i32){0}, &(i32){0}, &cscale, &wnrm, &n_info, &(i32){1}, &wi[*info], &ld_wi, &ierr);
    }

    dwork[0] = (f64)wrkopt;
}
