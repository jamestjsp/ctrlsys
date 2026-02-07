/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <math.h>

static inline i32 max_i32(i32 a, i32 b) {
    return a > b ? a : b;
}

static inline i32 min_i32(i32 a, i32 b) {
    return a < b ? a : b;
}

void mb02qd(
    const char* job,
    const char* iniper,
    const i32 m,
    const i32 n,
    const i32 nrhs,
    const f64 rcond,
    const f64 svlmax,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    const f64* y,
    i32* jpvt,
    i32* rank,
    f64* sval,
    f64* dwork,
    const i32 ldwork,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 DONE = ZERO;
    const f64 NTDONE = ONE;

    bool leasts, permut;
    i32 i, iascl, ibscl, j, k, maxwrk, minwrk, mn;
    f64 anrm, bignum, bnrm, smlnum, t1, t2;
    i32 int0 = 0;
    sl_int info_lapack;

    mn = min_i32(m, n);
    leasts = (*job == 'L' || *job == 'l');
    permut = (*iniper == 'P' || *iniper == 'p');

    *info = 0;
    minwrk = max_i32(mn + 3*n + 1, 2*mn + nrhs);

    if (!leasts && !(*job == 'F' || *job == 'f')) {
        *info = -1;
    } else if (!permut && !(*iniper == 'N' || *iniper == 'n')) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (nrhs < 0) {
        *info = -5;
    } else if (rcond < ZERO || rcond > ONE) {
        *info = -6;
    } else if (svlmax < ZERO) {
        *info = -7;
    } else if (lda < max_i32(1, m)) {
        *info = -9;
    } else if (ldb < max_i32(1, max_i32(m, n))) {
        *info = -11;
    } else if (ldwork < minwrk) {
        *info = -17;
    }

    if (*info != 0) {
        return;
    }

    if (mn == 0) {
        *rank = 0;
        dwork[0] = ONE;
        return;
    }

    smlnum = SLC_DLAMCH("Safe minimum") / SLC_DLAMCH("Precision");
    bignum = ONE / smlnum;
    SLC_DLABAD(&smlnum, &bignum);

    sl_int m_sl = m, n_sl = n, lda_sl = lda, ldb_sl = ldb;

    anrm = SLC_DLANGE("M", &m_sl, &n_sl, a, &lda_sl, dwork);
    iascl = 0;

    if (anrm > ZERO && anrm < smlnum) {
        SLC_DLASCL("G", &int0, &int0, &anrm, &smlnum, &m_sl, &n_sl, a, &lda_sl, &info_lapack);
        iascl = 1;
    } else if (anrm > bignum) {
        SLC_DLASCL("G", &int0, &int0, &anrm, &bignum, &m_sl, &n_sl, a, &lda_sl, &info_lapack);
        iascl = 2;
    } else if (anrm == ZERO) {
        if (nrhs > 0) {
            sl_int mn_max = max_i32(m, n);
            sl_int nrhs_sl = nrhs;
            SLC_DLASET("Full", &mn_max, &nrhs_sl, &ZERO, &ZERO, b, &ldb_sl);
        }
        *rank = 0;
        dwork[0] = ONE;
        return;
    }

    if (nrhs > 0) {
        sl_int nrhs_sl = nrhs;
        bnrm = SLC_DLANGE("M", &m_sl, &nrhs_sl, b, &ldb_sl, dwork);
        ibscl = 0;
        if (bnrm > ZERO && bnrm < smlnum) {
            SLC_DLASCL("G", &int0, &int0, &bnrm, &smlnum, &m_sl, &nrhs_sl, b, &ldb_sl, &info_lapack);
            ibscl = 1;
        } else if (bnrm > bignum) {
            SLC_DLASCL("G", &int0, &int0, &bnrm, &bignum, &m_sl, &nrhs_sl, b, &ldb_sl, &info_lapack);
            ibscl = 2;
        }
    } else {
        bnrm = ZERO;
        ibscl = 0;
    }

    maxwrk = minwrk;
    if (permut) {
        mb03od("Q", m, n, a, lda, jpvt, rcond, svlmax, &dwork[0], rank, sval,
               &dwork[mn], ldwork - mn, info);
        if (*info != 0) return;
        maxwrk = max_i32(maxwrk, (i32)dwork[mn] + mn);
    } else {
        mb03oy(m, n, a, lda, rcond, svlmax, rank, sval, jpvt, &dwork[0], &dwork[mn], info);
        if (*info != 0) return;
    }

    if (*rank < n) {
        sl_int rank_sl = *rank;
        sl_int ldwork_sl = ldwork - 2*mn;
        SLC_DTZRZF(&rank_sl, &n_sl, a, &lda_sl, &dwork[mn], &dwork[2*mn], &ldwork_sl, &info_lapack);
        maxwrk = max_i32(maxwrk, (i32)dwork[2*mn] + 2*mn);
    }

    if (nrhs > 0) {
        sl_int nrhs_sl = nrhs;
        sl_int ldwork_sl = ldwork - 2*mn;
        SLC_DORMQR("Left", "Transpose", &m_sl, &nrhs_sl, &mn, a, &lda_sl,
                   &dwork[0], b, &ldb_sl, &dwork[2*mn], &ldwork_sl, &info_lapack);
        maxwrk = max_i32(maxwrk, (i32)dwork[2*mn] + 2*mn);

        sl_int rank_sl = *rank;
        SLC_DTRSM("Left", "Upper", "No transpose", "Non-unit", &rank_sl,
                  &nrhs_sl, &ONE, a, &lda_sl, b, &ldb_sl);

        if (*rank < n) {
            sl_int n_minus_rank = n - *rank;
            if (leasts) {
                SLC_DLASET("Full", &n_minus_rank, &nrhs_sl, &ZERO, &ZERO,
                           &b[*rank], &ldb_sl);
            } else {
                SLC_DLACPY("Full", &n_minus_rank, &nrhs_sl, y, &n_minus_rank,
                           &b[*rank], &ldb_sl);
            }

            ldwork_sl = ldwork - 2*mn;
            SLC_DORMRZ("Left", "Transpose", &n_sl, &nrhs_sl, &rank_sl, &n_minus_rank,
                       a, &lda_sl, &dwork[mn], b, &ldb_sl, &dwork[2*mn], &ldwork_sl, &info_lapack);
            maxwrk = max_i32(maxwrk, (i32)dwork[2*mn] + 2*mn);
        }

        for (j = 0; j < nrhs; j++) {
            for (i = 0; i < n; i++) {
                dwork[2*mn + i] = NTDONE;
            }
            for (i = 0; i < n; i++) {
                if (dwork[2*mn + i] == NTDONE) {
                    i32 jpvt_i = jpvt[i] - 1;
                    if (jpvt_i < 0 || jpvt_i >= n) continue;
                    if (jpvt_i != i) {
                        k = i;
                        t1 = b[k + j*ldb];
                        i32 jpvt_k = jpvt[k] - 1;
                        if (jpvt_k < 0 || jpvt_k >= n) continue;
                        t2 = b[jpvt_k + j*ldb];

                        while (1) {
                            jpvt_k = jpvt[k] - 1;
                            if (jpvt_k < 0 || jpvt_k >= n) break;
                            b[jpvt_k + j*ldb] = t1;
                            dwork[2*mn + k] = DONE;
                            t1 = t2;
                            k = jpvt_k;
                            jpvt_k = jpvt[k] - 1;
                            if (jpvt_k < 0 || jpvt_k >= n) break;
                            t2 = b[jpvt_k + j*ldb];
                            if (jpvt_k == i)
                                break;
                        }
                        b[i + j*ldb] = t1;
                        dwork[2*mn + k] = DONE;
                    }
                }
            }
        }

        if (ibscl == 1) {
            SLC_DLASCL("G", &int0, &int0, &smlnum, &bnrm, &n_sl, &nrhs_sl, b, &ldb_sl, &info_lapack);
        } else if (ibscl == 2) {
            SLC_DLASCL("G", &int0, &int0, &bignum, &bnrm, &n_sl, &nrhs_sl, b, &ldb_sl, &info_lapack);
        }
    }

    if (iascl == 1) {
        if (nrhs > 0) {
            sl_int nrhs_sl = nrhs;
            SLC_DLASCL("G", &int0, &int0, &anrm, &smlnum, &n_sl, &nrhs_sl, b, &ldb_sl, &info_lapack);
        }
        sl_int rank_sl = *rank;
        SLC_DLASCL("U", &int0, &int0, &smlnum, &anrm, &rank_sl, &rank_sl, a, &lda_sl, &info_lapack);
    } else if (iascl == 2) {
        if (nrhs > 0) {
            sl_int nrhs_sl = nrhs;
            SLC_DLASCL("G", &int0, &int0, &anrm, &bignum, &n_sl, &nrhs_sl, b, &ldb_sl, &info_lapack);
        }
        sl_int rank_sl = *rank;
        SLC_DLASCL("U", &int0, &int0, &bignum, &anrm, &rank_sl, &rank_sl, a, &lda_sl, &info_lapack);
    }

    for (i = mn + *rank - 1; i >= 0; i--) {
        dwork[i + 1] = dwork[i];
    }

    dwork[0] = (f64)maxwrk;
}
