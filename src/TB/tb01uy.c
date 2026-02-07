/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>

static bool lsame(char ca, char cb) {
    if (ca >= 'a' && ca <= 'z') ca -= 32;
    if (cb >= 'a' && cb <= 'z') cb -= 32;
    return ca == cb;
}

static i32 imax(i32 a, i32 b) {
    return a > b ? a : b;
}

static i32 imin(i32 a, i32 b) {
    return a < b ? a : b;
}

void tb01uy(const char* jobz, i32 n, i32 m1, i32 m2, i32 p,
            f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            i32* ncont, i32* indcon, i32* nblk,
            f64* z, i32 ldz, f64* tau, f64 tol,
            i32* iwork, f64* dwork, i32 ldwork, i32* info) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const i32 one_i = 1;

    bool b1red, ljobf, ljobi, ljobz, lquery;
    i32 iqr, itau, j, jb2, jqr, m, mcrt, mcrt1, mcrt2;
    i32 minwrk, ncrt, ni, nj, rank, wrkopt;
    f64 anorm, bnorm, fnrm, fnrm2, fnrma, toldef;
    f64 sval[3];

    *info = 0;
    ljobf = lsame(*jobz, 'F');
    ljobi = lsame(*jobz, 'I');
    ljobz = ljobf || ljobi;
    m = m1 + m2;

    if (!ljobz && !lsame(*jobz, 'N')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (m1 < 0) {
        *info = -3;
    } else if (m2 < 0) {
        *info = -4;
    } else if (p < 0) {
        *info = -5;
    } else if (lda < imax(1, n)) {
        *info = -7;
    } else if (ldb < imax(1, n)) {
        *info = -9;
    } else if (ldc < imax(1, p)) {
        *info = -11;
    } else if (ldz < 1 || (ljobz && ldz < imax(1, n))) {
        *info = -16;
    } else {
        if (imin(n, m) == 0) {
            minwrk = 1;
        } else {
            minwrk = imax(n, imax(3 * imax(m1, m2), p));
        }

        lquery = ldwork < 0;
        if (lquery) {
            mcrt = imax(m1, m2);
            rank = imin(n, mcrt);
            i32 n_i = n, mcrt_i = mcrt, rank_i = rank;
            i32 query = -1;
            SLC_DORMQR("Left", "Transpose", &n_i, &mcrt_i, &rank_i, b, &ldb,
                       tau, b, &ldb, dwork, &query, info);
            wrkopt = imax(minwrk, (i32)dwork[0]);
            SLC_DORMQR("Left", "Transpose", &n_i, &n_i, &rank_i, b, &ldb,
                       tau, a, &lda, dwork, &query, info);
            wrkopt = imax(wrkopt, (i32)dwork[0]);
            i32 p_i = p;
            SLC_DORMQR("Right", "No transpose", &p_i, &n_i, &rank_i, b, &ldb,
                       tau, c, &ldc, dwork, &query, info);
            wrkopt = imax(wrkopt, (i32)dwork[0]);
            if (ljobi && n > 0) {
                i32 n_m1 = n - 1;
                SLC_DORGQR(&n_i, &n_i, &n_m1, z, &ldz, tau, dwork, &query, info);
                wrkopt = imax(wrkopt, (i32)dwork[0]);
            }
        } else if (ldwork < minwrk) {
            *info = -21;
        }
    }

    if (*info != 0) {
        return;
    } else if (lquery) {
        dwork[0] = (f64)wrkopt;
        return;
    }

    *ncont = 0;
    *indcon = 0;

    if (n == 0) {
        dwork[0] = ONE;
        return;
    }

    anorm = SLC_DLANGE("M", &n, &n, a, &lda, dwork);
    bnorm = SLC_DLANGE("M", &n, &m, b, &ldb, dwork);

    if (bnorm == ZERO) {
        if (ljobi) {
            SLC_DLASET("Full", &n, &n, &ZERO, &ONE, z, &ldz);
        } else if (ljobf && n > 1) {
            i32 n_m1 = n - 1;
            SLC_DLASET("Lower", &n_m1, &n_m1, &ZERO, &ZERO, &z[1], &ldz);
            i32 min_nm = imin(n, m);
            SLC_DLASET("Full", &min_nm, &one_i, &ZERO, &ZERO, tau, &n);
        }
        dwork[0] = ONE;
        return;
    }

    mb01pd("S", "G", n, n, 0, 0, anorm, 0, nblk, a, lda, info);
    mb01pd("S", "G", n, m, 0, 0, bnorm, 0, nblk, b, ldb, info);

    fnrm = SLC_DLANGE("F", &n, &m1, b, &ldb, dwork);
    fnrma = SLC_DLANGE("F", &n, &n, a, &lda, dwork);
    if (m2 > 0) {
        i32 m2_i = m2;
        fnrm2 = SLC_DLANGE("F", &n, &m2_i, &b[m1 * ldb], &ldb, dwork);
    } else {
        fnrm2 = ZERO;
    }

    toldef = tol;
    if (toldef <= ZERO) {
        toldef = (f64)(n * n) * SLC_DLAMCH("Epsilon");
    }

    wrkopt = 1;
    ni = 0;
    nj = 0;
    itau = 0;
    ncrt = n;
    mcrt1 = m1;
    mcrt2 = m2;
    mcrt = mcrt1;
    iqr = 0;
    jqr = 0;
    b1red = true;
    jb2 = imin(m1, m - 1);

    while (1) {
        i32 ncrt_i = ncrt;
        mb03oy(ncrt, mcrt, &b[iqr + jqr * ldb], ldb, toldef, fnrm, &rank, sval, iwork, &tau[itau], dwork, info);

        if (rank == 0) {
            if (b1red) {
                if (mcrt2 > 0) {
                    b1red = !b1red;
                    (*indcon)++;
                    nblk[*indcon - 1] = 0;
                    if (*indcon == 1) {
                        fnrm = fnrm2;
                    } else if (*indcon > 2) {
                        nj += mcrt;
                    }
                    mcrt1 = 0;
                    mcrt = mcrt2;
                    jqr = jb2;
                    if (*indcon >= 2) {
                        if (*indcon == 2)
                            fnrm = fnrma;
                        i32 mcrt_loc = mcrt;
                        SLC_DLACPY("G", &ncrt_i, &mcrt_loc, &a[*ncont + nj * lda], &lda, &b[iqr + jqr * ldb], &ldb);
                        SLC_DLASET("G", &ncrt_i, &mcrt_loc, &ZERO, &ZERO, &a[*ncont + nj * lda], &lda);
                    }
                    continue;
                }
            } else {
                if (mcrt1 > 0) {
                    b1red = !b1red;
                    (*indcon)++;
                    nblk[*indcon - 1] = 0;
                    mcrt2 = 0;
                    mcrt = mcrt1;
                    jqr = 0;
                    if (*indcon >= 2) {
                        if (*indcon == 2) {
                            fnrm = fnrma;
                        } else if (*indcon > 2) {
                            nj += mcrt;
                        }
                        i32 mcrt_loc = mcrt;
                        SLC_DLACPY("G", &ncrt_i, &mcrt_loc, &a[*ncont + nj * lda], &lda, &b[iqr + jqr * ldb], &ldb);
                        SLC_DLASET("G", &ncrt_i, &mcrt_loc, &ZERO, &ZERO, &a[*ncont + nj * lda], &lda);
                    }
                    continue;
                }
                (*indcon)--;
            }
        } else {
            ni = *ncont;
            *ncont = *ncont + rank;
            (*indcon)++;
            nblk[*indcon - 1] = rank;

            if (*indcon < 2) {
                fnrm = fnrm2;
                i32 mcrt2_i = mcrt2, rank_i = rank;
                SLC_DORMQR("Left", "Transpose", &ncrt_i, &mcrt2_i, &rank_i,
                           &b[iqr + jqr * ldb], &ldb, &tau[itau],
                           &b[ni + jb2 * ldb], &ldb, dwork, &ldwork, info);
                wrkopt = imax(wrkopt, (i32)dwork[0]);
            }

            i32 n_minus_nj = n - nj;
            i32 rank_i = rank;
            SLC_DORMQR("Left", "Transpose", &ncrt_i, &n_minus_nj, &rank_i,
                       &b[iqr + jqr * ldb], &ldb, &tau[itau], &a[ni + nj * lda], &lda,
                       dwork, &ldwork, info);
            wrkopt = imax(wrkopt, (i32)dwork[0]);

            SLC_DORMQR("Right", "No transpose", &n, &ncrt_i, &rank_i,
                       &b[iqr + jqr * ldb], &ldb, &tau[itau], &a[ni * lda], &lda,
                       dwork, &ldwork, info);
            wrkopt = imax(wrkopt, (i32)dwork[0]);

            i32 p_i = p;
            SLC_DORMQR("Right", "No transpose", &p_i, &ncrt_i, &rank_i,
                       &b[iqr + jqr * ldb], &ldb, &tau[itau], &c[ni * ldc], &ldc,
                       dwork, &ldwork, info);
            wrkopt = imax(wrkopt, (i32)dwork[0]);

            if (ljobz && ncrt > 1) {
                i32 ncrt_m1 = ncrt - 1;
                i32 min_rank_ncrt_m1 = imin(rank, ncrt - 1);
                SLC_DLACPY("L", &ncrt_m1, &min_rank_ncrt_m1,
                           &b[iqr + 1 + jqr * ldb], &ldb, &z[ni + 1 + itau * ldz], &ldz);
            }

            if (rank > 1) {
                i32 rank_m1 = rank - 1;
                SLC_DLASET("L", &rank_m1, &rank_m1, &ZERO, &ZERO, &b[iqr + 1 + jqr * ldb], &ldb);
            }

            if (*indcon <= 2) {
                if (*indcon == 2)
                    fnrm = fnrma;
                i32 forwrd = 0;
                i32 rank_i2 = rank, mcrt_i2 = mcrt;
                SLC_DLAPMT(&forwrd, &rank_i2, &mcrt_i2, &b[iqr + jqr * ldb], &ldb, iwork);
                iqr = iqr + rank;
            } else {
                for (j = 0; j < mcrt; j++) {
                    i32 ipiv_j = iwork[j] - 1;
                    if (ipiv_j >= 0 && ipiv_j < mcrt) {
                        SLC_DCOPY(&rank_i, &b[iqr + (jqr + j) * ldb], &one_i, &a[ni + (nj + ipiv_j) * lda], &one_i);
                    }
                }
            }

            itau = itau + rank;
            if (rank != ncrt) {
                if (*indcon > 2)
                    nj += mcrt;
                if (b1red) {
                    mcrt1 = rank;
                    mcrt = mcrt2;
                    jqr = jb2;
                } else {
                    mcrt2 = rank;
                    mcrt = mcrt1;
                    jqr = 0;
                }
                ncrt = ncrt - rank;
                if (*indcon >= 2) {
                    i32 mcrt_loc = mcrt;
                    SLC_DLACPY("G", &ncrt, &mcrt_loc, &a[*ncont + nj * lda], &lda, &b[iqr + jqr * ldb], &ldb);
                    SLC_DLASET("G", &ncrt, &mcrt_loc, &ZERO, &ZERO, &a[*ncont + nj * lda], &lda);
                }
                b1red = !b1red;
                continue;
            } else {
                if (b1red) {
                    (*indcon)++;
                    nblk[*indcon - 1] = 0;
                }
            }
        }
        break;
    }

    if (ljobi) {
        i32 itau_m1 = imax(1, itau);
        SLC_DORGQR(&n, &n, &itau_m1, z, &ldz, tau, dwork, &ldwork, info);
        wrkopt = imax(wrkopt, (i32)dwork[0]);
    }

    i32 nblk0 = nblk[0];
    if (nblk0 < n) {
        i32 n_minus_nblk0 = n - nblk0;
        i32 m1_i = m1;
        SLC_DLASET("G", &n_minus_nblk0, &m1_i, &ZERO, &ZERO, &b[nblk0], &ldb);
    }
    if (iqr < n) {
        i32 n_minus_iqr = n - iqr;
        i32 m2_i = m2;
        SLC_DLASET("G", &n_minus_iqr, &m2_i, &ZERO, &ZERO, &b[iqr + jb2 * ldb], &ldb);
    }

    if (ljobf) {
        for (j = itau; j < imin(n, m); j++) {
            tau[j] = ZERO;
        }
    }

    mb01pd("U", "G", n, n, 0, 0, anorm, 0, nblk, a, lda, info);
    i32 nblk_sum = nblk[0] + ((*indcon > 1) ? nblk[1] : 0);
    mb01pd("U", "G", nblk_sum, m, 0, 0, bnorm, 0, nblk, b, ldb, info);

    dwork[0] = (f64)wrkopt;
}
