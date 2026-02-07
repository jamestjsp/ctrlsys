/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>

void tg01jd(
    const char* job, const char* systyp, const char* equil,
    const i32 n, const i32 m, const i32 p,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    i32* nr, i32* infred,
    const f64 tol,
    i32* iwork, f64* dwork, const i32 ldwork,
    i32* info
)
{
    const f64 one = 1.0;

    bool done1, done2, done3, fincon, finobs, infcon, infobs, lequil;
    bool ljobc, ljobir, ljobo, lspace, lsysp, lsysr, lsyss;
    i32 i, ib, kwa, kwb, kwc, kwe, lba, lbe, ldm, ldp, ldq, ldz;
    i32 m1, maxmp, n1, nblck, nc, p1;
    f64 nrm, thrsh;
    f64 dum[1];
    i32 int1 = 1;

    *info = 0;
    maxmp = (m > p) ? m : p;
    n1 = (n > 1) ? n : 1;

    ljobir = (job[0] == 'I' || job[0] == 'i');
    ljobc = ljobir || (job[0] == 'C' || job[0] == 'c');
    ljobo = ljobir || (job[0] == 'O' || job[0] == 'o');

    lsysr = (systyp[0] == 'R' || systyp[0] == 'r');
    lsyss = lsysr || (systyp[0] == 'S' || systyp[0] == 's');
    lsysp = lsysr || (systyp[0] == 'P' || systyp[0] == 'p');

    lequil = (equil[0] == 'S' || equil[0] == 's');

    if (!ljobc && !ljobo) {
        *info = -1;
    } else if (!lsyss && !lsysp) {
        *info = -2;
    } else if (!lequil && !(equil[0] == 'N' || equil[0] == 'n')) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (m < 0) {
        *info = -5;
    } else if (p < 0) {
        *info = -6;
    } else if (lda < n1) {
        *info = -8;
    } else if (lde < n1) {
        *info = -10;
    } else if (ldb < n1) {
        *info = -12;
    } else if (ldc < 1 || (n > 0 && ldc < maxmp)) {
        *info = -14;
    } else if (tol >= one) {
        *info = -17;
    } else {
        i32 min_ws = lequil ? (8 * n > 2 * maxmp ? 8 * n : 2 * maxmp)
                            : (n > 2 * maxmp ? n : 2 * maxmp);
        if (ldwork < min_ws) {
            *info = -20;
        }
    }

    if (*info != 0) {
        return;
    }

    infred[0] = -1;
    infred[1] = -1;
    infred[2] = -1;
    infred[3] = -1;
    infred[4] = 0;
    infred[5] = 0;
    infred[6] = 0;

    i32 max_n_maxmp = (n > maxmp) ? n : maxmp;
    if (max_n_maxmp == 0) {
        *nr = 0;
        return;
    }

    m1 = (m > 1) ? m : 1;
    p1 = (p > 1) ? p : 1;
    ldm = (ldc > m) ? ldc : m;
    ldp = (ldc > p) ? ldc : p;

    fincon = ljobc && lsyss;
    infcon = ljobc && lsysp;
    finobs = ljobo && lsyss;
    infobs = ljobo && lsysp;

    lspace = ldwork >= n * (2 * n + m + p) + (n > 2 * maxmp ? n : 2 * maxmp);
    kwa = (n > 2 * maxmp ? n : 2 * maxmp);
    kwe = kwa + n * n;
    kwb = kwe + n * n;
    kwc = kwb + n * m;

    if (lequil) {
        nrm = SLC_DLANGE("1", &n, &n, a, &lda, dwork);
        f64 tmp = SLC_DLANGE("1", &n, &n, e, &lde, dwork);
        if (tmp > nrm) nrm = tmp;
        tmp = SLC_DLANGE("1", &n, &m, b, &ldb, dwork);
        if (tmp > nrm) nrm = tmp;
        tmp = SLC_DLANGE("1", &p, &n, c, &ldc, dwork);
        if (tmp > nrm) nrm = tmp;

        thrsh = nrm * SLC_DLAMCH("P");
        tg01ad("A", n, n, m, p, thrsh, a, lda, e, lde, b, ldb, c, ldp,
               dwork, &dwork[n], &dwork[2 * n], info);
    }

    ldq = 1;
    ldz = 1;
    lba = (n > 1) ? n - 1 : 0;
    lbe = lba;
    nc = n;
    *nr = n;
    ib = 0;

    done1 = false;
    done2 = false;
    done3 = false;

    if (fincon) {
        if (lspace) {
            SLC_DLACPY("F", &nc, &nc, a, &lda, &dwork[kwa], &n1);
            SLC_DLACPY("F", &nc, &nc, e, &lde, &dwork[kwe], &n1);
            SLC_DLACPY("F", &nc, &m, b, &ldb, &dwork[kwb], &n1);
            SLC_DLACPY("F", &p, &nc, c, &ldc, &dwork[kwc], &p1);
        }

        tg01hx("N", "N", nc, nc, m, p, nc, lbe, a, lda, e, lde, b, ldb, c, ldp,
               dum, ldq, dum, ldz, nr, &nblck, iwork, tol, &iwork[n], dwork, info);

        done1 = (*nr < nc) || !lspace;
        if (done1) {
            if (nblck > 1) {
                lba = iwork[0] + iwork[1] - 1;
            } else if (nblck == 1) {
                lba = iwork[0] - 1;
            } else {
                lba = 0;
            }
            lbe = 0;
            infred[0] = nc - *nr;
            infred[6] = nblck;
            nc = *nr;
            ib = n;
        } else {
            SLC_DLACPY("F", &nc, &nc, &dwork[kwa], &n1, a, &lda);
            SLC_DLACPY("F", &nc, &nc, &dwork[kwe], &n1, e, &lde);
            SLC_DLACPY("F", &nc, &m, &dwork[kwb], &n1, b, &ldb);
            SLC_DLACPY("F", &p, &nc, &dwork[kwc], &p1, c, &ldc);
        }
    }

    if (infcon) {
        if (lspace && (!fincon || done1)) {
            SLC_DLACPY("F", &nc, &nc, a, &lda, &dwork[kwa], &n1);
            SLC_DLACPY("F", &nc, &nc, e, &lde, &dwork[kwe], &n1);
            SLC_DLACPY("F", &nc, &m, b, &ldb, &dwork[kwb], &n1);
            SLC_DLACPY("F", &p, &nc, c, &ldc, &dwork[kwc], &p1);
        }

        tg01hx("N", "N", nc, nc, m, p, nc, lba, e, lde, a, lda, b, ldb, c, ldp,
               dum, ldq, dum, ldz, nr, &nblck, &iwork[ib], tol, &iwork[ib + n], dwork, info);

        done2 = (*nr < nc) || !lspace;
        if (done2) {
            if (nblck > 1) {
                lbe = iwork[ib] + iwork[ib + 1] - 1;
            } else if (nblck == 1) {
                lbe = iwork[ib] - 1;
            } else {
                lbe = 0;
            }
            lba = 0;
            infred[1] = nc - *nr;
            infred[6] = nblck;
            nc = *nr;
            if (done1) {
                for (i = 0; i < nblck; i++) {
                    iwork[i] = iwork[ib + i];
                }
            } else {
                ib = n;
            }
        } else {
            SLC_DLACPY("F", &nc, &nc, &dwork[kwa], &n1, a, &lda);
            SLC_DLACPY("F", &nc, &nc, &dwork[kwe], &n1, e, &lde);
            SLC_DLACPY("F", &nc, &m, &dwork[kwb], &n1, b, &ldb);
            SLC_DLACPY("F", &p, &nc, &dwork[kwc], &p1, c, &ldc);
        }
    }

    if (finobs || infobs) {
        i32 nc_m1 = (nc > 1) ? nc - 1 : 0;
        tb01xd("Z", nc, m, p, lba, nc_m1, a, lda, b, ldb, c, ldc, dum, int1, info);
        ma02cd(nc, lbe, nc_m1, e, lde);
    }

    if (finobs) {
        if (lspace) {
            SLC_DLACPY("F", &nc, &nc, a, &lda, &dwork[kwa], &n1);
            SLC_DLACPY("F", &nc, &nc, e, &lde, &dwork[kwe], &n1);
            SLC_DLACPY("F", &nc, &p, b, &ldb, &dwork[kwc], &n1);
            SLC_DLACPY("F", &m, &nc, c, &ldc, &dwork[kwb], &m1);
        }

        tg01hx("N", "N", nc, nc, p, m, nc, lbe, a, lda, e, lde, b, ldb, c, ldm,
               dum, ldz, dum, ldq, nr, &nblck, &iwork[ib], tol, &iwork[ib + n], dwork, info);

        done3 = (*nr < nc) || !lspace;
        if (done3) {
            if (nblck > 1) {
                lba = iwork[ib] + iwork[ib + 1] - 1;
            } else if (nblck == 1) {
                lba = iwork[ib] - 1;
            } else {
                lba = 0;
            }
            lbe = 0;
            infred[2] = nc - *nr;
            infred[6] = nblck;
            nc = *nr;
            if (done1 || done2) {
                for (i = 0; i < nblck; i++) {
                    iwork[i] = iwork[ib + i];
                }
            } else {
                ib = n;
            }
        } else {
            SLC_DLACPY("F", &nc, &nc, &dwork[kwa], &n1, a, &lda);
            SLC_DLACPY("F", &nc, &nc, &dwork[kwe], &n1, e, &lde);
            SLC_DLACPY("F", &nc, &p, &dwork[kwc], &n1, b, &ldb);
            SLC_DLACPY("F", &m, &nc, &dwork[kwb], &m1, c, &ldc);
        }
    }

    if (infobs) {
        if (lspace && (!finobs || done3)) {
            SLC_DLACPY("F", &nc, &nc, a, &lda, &dwork[kwa], &n1);
            SLC_DLACPY("F", &nc, &nc, e, &lde, &dwork[kwe], &n1);
            SLC_DLACPY("F", &nc, &p, b, &ldb, &dwork[kwc], &n1);
            SLC_DLACPY("F", &m, &nc, c, &ldc, &dwork[kwb], &m1);
        }

        tg01hx("N", "N", nc, nc, p, m, nc, lba, e, lde, a, lda, b, ldb, c, ldm,
               dum, ldz, dum, ldq, nr, &nblck, &iwork[ib], tol, &iwork[ib + n], dwork, info);

        if ((*nr < nc) || !lspace) {
            if (nblck > 1) {
                lbe = iwork[ib] + iwork[ib + 1] - 1;
            } else if (nblck == 1) {
                lbe = iwork[ib] - 1;
            } else {
                lbe = 0;
            }
            lba = 0;
            infred[3] = nc - *nr;
            infred[6] = nblck;
            nc = *nr;
            if (done1 || done2 || done3) {
                for (i = 0; i < nblck; i++) {
                    iwork[i] = iwork[ib + i];
                }
            }
        } else {
            SLC_DLACPY("F", &nc, &nc, &dwork[kwa], &n1, a, &lda);
            SLC_DLACPY("F", &nc, &nc, &dwork[kwe], &n1, e, &lde);
            SLC_DLACPY("F", &nc, &p, &dwork[kwc], &n1, b, &ldb);
            SLC_DLACPY("F", &m, &nc, &dwork[kwb], &m1, c, &ldc);
        }
    }

    if (finobs || infobs) {
        i32 nc_m1 = (nc > 1) ? nc - 1 : 0;
        tb01xd("Z", nc, p, m, lba, nc_m1, a, lda, b, ldb, c, ldc, dum, int1, info);
        ma02cd(nc, lbe, nc_m1, e, lde);
    }

    infred[4] = lba;
    infred[5] = lbe;
}
