/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 *
 * AB13ID - Check properness of transfer function of descriptor system
 *
 * Purpose:
 *   To check whether the transfer function
 *     G(lambda) := C*(lambda*E - A)^(-1)*B
 *   of a given linear time-invariant descriptor system with
 *   generalized state space realization (lambda*E-A,B,C) is proper.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>

static inline i32 max_i32(i32 a, i32 b) { return a > b ? a : b; }

bool ab13id(
    const char* jobsys, const char* jobeig, const char* equil,
    const char* cksing, const char* restor, const char* update,
    const i32 n, const i32 m, const i32 p,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    i32* nr, i32* ranke,
    const f64* tol,
    i32* iwork, f64* dwork, const i32 ldwork,
    i32* iwarn, i32* info
)
{
    const f64 zero = 0.0, one = 1.0, ten = 10.0;

    bool lredc, lrema, lequil, lsing, maxacc, lupd, lrupd, lquery;
    i32 i, j, k, isv, itau, iwrk, iws, maxmp, maxwrk, minwrk, n1, na, ranka;
    f64 prec, svlmax, thresh, toldef;
    f64 dum[2];
    f64 tolv[3];
    i32 int1 = 1, intmn;
    char systyp;

    *info = 0;
    *iwarn = 0;
    maxmp = max_i32(m, p);
    n1 = max_i32(1, n);

    lredc = (*jobsys == 'R' || *jobsys == 'r');
    lrema = (*jobeig == 'A' || *jobeig == 'a');
    lequil = (*equil == 'S' || *equil == 's');
    lsing = (*cksing == 'C' || *cksing == 'c');
    maxacc = (*restor == 'R' || *restor == 'r');
    lupd = (*update == 'U' || *update == 'u');
    lrupd = lrema || lupd;

    if (!lredc && !(*jobsys == 'N' || *jobsys == 'n')) {
        *info = -1;
    } else if (!lrema && !(*jobeig == 'I' || *jobeig == 'i')) {
        *info = -2;
    } else if (!lequil && !(*equil == 'N' || *equil == 'n')) {
        *info = -3;
    } else if (!lsing && !(*cksing == 'N' || *cksing == 'n')) {
        *info = -4;
    } else if (!maxacc && !(*restor == 'N' || *restor == 'n')) {
        *info = -5;
    } else if (!lupd && !(*update == 'N' || *update == 'n')) {
        *info = -6;
    } else if (n < 0) {
        *info = -7;
    } else if (m < 0) {
        *info = -8;
    } else if (p < 0) {
        *info = -9;
    } else if (lda < n1) {
        *info = -11;
    } else if (lde < n1) {
        *info = -13;
    } else if (ldb < n1) {
        *info = -15;
    } else if (ldc < 1 || (n > 0 && ldc < maxmp)) {
        *info = -17;
    } else if (tol[0] >= one) {
        *info = -20;
    } else if (tol[1] >= one) {
        *info = -20;
    } else if (lequil) {
        thresh = tol[2];
        if (thresh >= one) {
            *info = -20;
        }
    }

    if (*info == 0) {
        if (lredc) {
            k = n * (2 * n + m + p);
            if (maxacc) {
                minwrk = max_i32(1, 2 * (k + maxmp + n - 1));
            } else {
                minwrk = max_i32(1, 2 * (maxmp + n - 1));
            }
            minwrk = max_i32(minwrk, n * n + 4 * n);
            if (lsing) {
                minwrk = max_i32(minwrk, 2 * n * n + 10 * n + max_i32(n, 23));
            }
            if (lrema) {
                systyp = 'R';
            } else {
                systyp = 'P';
            }
        } else {
            minwrk = 0;
        }
        if (lrupd) {
            minwrk = max_i32(minwrk, max_i32(n * n + 4 * n + 4, n + maxmp));
        } else {
            minwrk = max_i32(minwrk, 4 * n + 4);
        }
        if (lequil) {
            minwrk = max_i32(minwrk, 8 * n);
        }

        maxwrk = minwrk;
        lquery = (ldwork == -1);

        if (lquery) {
            if (lredc) {
                i32 nr_tmp, infred_tmp[7];
                tg01jy("I", &systyp, "N", cksing, restor, n, m, p,
                       a, lda, e, lde, b, ldb, c, ldc,
                       &nr_tmp, infred_tmp, tol, iwork, dwork, -1, info);
                maxwrk = max_i32(maxwrk, (i32)dwork[0]);
            }
            intmn = -1;
            mb03od("Q", n, n, e, lde, iwork, zero, zero, dwork, ranke,
                   &dwork[n], &dwork[n + 3], -1, info);
            maxwrk = max_i32(maxwrk, (i32)dum[0] + n + 3);

            SLC_DORMQR("L", "T", &n, &n, &n, e, &lde, dwork, a, &lda, dum, &intmn, info);
            maxwrk = max_i32(maxwrk, (i32)dum[0] + n);

            if (lrupd) {
                SLC_DORMQR("L", "T", &n, &m, &n, e, &lde, dwork, b, &ldb, &dum[1], &intmn, info);
                maxwrk = max_i32(maxwrk, (i32)dum[1] + n);
            }

            SLC_DTZRZF(&n, &n, e, &lde, dwork, dum, &intmn, info);
            SLC_DORMRZ("R", "T", &n, &n, &n, &n, e, &lde, dwork, a, &lda, &dum[1], &intmn, info);
            maxwrk = max_i32(maxwrk, max_i32((i32)dum[0], (i32)dum[1]) + n);

            if (lrupd) {
                SLC_DORMRZ("R", "T", &p, &n, &n, &n, e, &lde, dwork, c, &ldc, dum, &intmn, info);
                maxwrk = max_i32(maxwrk, (i32)dum[0] + n);
            }
            dwork[0] = (f64)maxwrk;
            return true;
        } else if (ldwork < minwrk) {
            *info = -23;
        }
    }

    if (*info != 0) {
        return false;
    }

    *nr = n;
    if (n == 0) {
        *ranke = 0;
        dwork[0] = one;
        return true;
    }

    toldef = tol[0];
    if (toldef <= zero || lequil) {
        prec = SLC_DLAMCH("P");
        if (lequil) {
            if (tol[2] < zero) {
                thresh = fmax(
                    fmax(SLC_DLANGE("1", &n, &n, a, &lda, dwork),
                         SLC_DLANGE("1", &n, &n, e, &lde, dwork)),
                    fmax(SLC_DLANGE("1", &n, &m, b, &ldb, dwork),
                         SLC_DLANGE("1", &p, &n, c, &ldc, dwork))) * prec;
            }
        }
        if (toldef <= zero) {
            toldef = (f64)(n * n) * prec;
        }
    }
    tolv[0] = toldef;
    tolv[1] = tol[1];
    tolv[2] = tol[2];

    if (lequil) {
        tg01ad("A", n, n, m, p, thresh, a, lda, e, lde, b, ldb, c, ldc,
               dwork, &dwork[n], &dwork[2*n], info);
        maxwrk = max_i32(maxwrk, 8 * n);
    }

    if (lredc) {
        iws = 7;
        i32 nr_tmp;
        tg01jy("I", &systyp, "N", cksing, restor, n, m, p,
               a, lda, e, lde, b, ldb, c, ldc,
               &nr_tmp, iwork, tolv, &iwork[iws], dwork, ldwork, info);
        *nr = nr_tmp;
        maxwrk = max_i32(maxwrk, (i32)dwork[0]);
        if (*info == 1) {
            return false;
        }
    } else {
        iws = 0;
    }

    for (i = iws; i < iws + *nr; i++) {
        iwork[i] = 0;
    }

    svlmax = SLC_DLANGE("F", &n, &n, e, &lde, dwork);
    i32 nr_val = *nr;
    mb03od("Q", nr_val, nr_val, e, lde, &iwork[iws], toldef, svlmax,
           dwork, ranke, &dwork[nr_val], &dwork[nr_val + 3],
           ldwork - (nr_val + 3), info);
    maxwrk = max_i32(maxwrk, (i32)dwork[nr_val + 3] + nr_val + 3);

    if (fabs(dwork[nr_val + 2] / dwork[nr_val] - toldef) < toldef / ten) {
        *iwarn = 1;
    }

    if (*ranke < *nr || lrupd) {
        i32 ldwork_remain = ldwork - nr_val;
        SLC_DORMQR("L", "T", &nr_val, &nr_val, &nr_val, e, &lde, dwork, a, &lda,
                   &dwork[nr_val], &ldwork_remain, info);
        maxwrk = max_i32(maxwrk, (i32)dwork[nr_val] + nr_val);

        for (i = iws; i < iws + *nr; i++) {
            iwork[i] = -iwork[i];
        }

        for (i = iws; i < iws + *nr; i++) {
            if (iwork[i] < 0) {
                j = i;
                iwork[j] = -iwork[j];
                for (;;) {
                    k = iwork[j] + iws - 1;
                    if (k < 0 || k >= iws + *nr) break;
                    if (iwork[k] < 0) {
                        SLC_DSWAP(&nr_val, &a[(j - iws) * lda], &int1, &a[(k - iws) * lda], &int1);
                        iwork[k] = -iwork[k];
                        j = k;
                    } else {
                        break;
                    }
                }
            }
        }

        if (lrupd) {
            ldwork_remain = ldwork - nr_val;
            SLC_DORMQR("L", "T", &nr_val, &m, &nr_val, e, &lde, dwork,
                       b, &ldb, &dwork[nr_val], &ldwork_remain, info);
            maxwrk = max_i32(maxwrk, (i32)dwork[nr_val] + nr_val);

            for (i = iws; i < iws + *nr; i++) {
                iwork[i] = -iwork[i];
            }

            for (i = iws; i < iws + *nr; i++) {
                if (iwork[i] < 0) {
                    j = i;
                    iwork[j] = -iwork[j];
                    for (;;) {
                        k = iwork[j] + iws - 1;
                        if (k < 0 || k >= iws + *nr) break;
                        if (iwork[k] < 0) {
                            SLC_DSWAP(&p, &c[(j - iws) * ldc], &int1, &c[(k - iws) * ldc], &int1);
                            iwork[k] = -iwork[k];
                            j = k;
                        } else {
                            break;
                        }
                    }
                }
            }
        }
    }

    if (*ranke < *nr) {
        i32 ranke_val = *ranke;
        i32 ldwork_remain = ldwork - ranke_val;
        SLC_DTZRZF(&ranke_val, &nr_val, e, &lde, dwork, &dwork[ranke_val], &ldwork_remain, info);
        maxwrk = max_i32(maxwrk, (i32)dwork[ranke_val] + ranke_val);

        na = *nr - *ranke;
        SLC_DORMRZ("R", "T", &nr_val, &nr_val, &ranke_val, &na, e, &lde,
                   dwork, a, &lda, &dwork[ranke_val], &ldwork_remain, info);
        maxwrk = max_i32(maxwrk, (i32)dwork[ranke_val] + ranke_val);

        if (lrupd) {
            SLC_DORMRZ("R", "T", &p, &nr_val, &ranke_val, &na, e, &lde,
                       dwork, c, &ldc, &dwork[ranke_val], &ldwork_remain, info);
            maxwrk = max_i32(maxwrk, (i32)dwork[ranke_val] + ranke_val);
        }

        for (i = iws; i < iws + na; i++) {
            iwork[i] = 0;
        }

        svlmax = SLC_DLANGE("F", &nr_val, &nr_val, a, &lda, dwork);

        if (lrupd) {
            itau = na * na;
            isv = itau + na;
            iwrk = isv + 3;
            SLC_DLACPY("F", &na, &na, &a[*ranke + (*ranke) * lda], &lda, dwork, &na);
            mb03od("Q", na, na, dwork, na, &iwork[iws], toldef, svlmax,
                   &dwork[itau], &ranka, &dwork[isv], &dwork[iwrk],
                   ldwork - iwrk, info);
        } else {
            isv = na;
            iwrk = isv + 3;
            mb03od("Q", na, na, &a[*ranke + (*ranke) * lda], lda, &iwork[iws],
                   toldef, svlmax, dwork, &ranka, &dwork[isv], &dwork[iwrk],
                   ldwork - iwrk, info);
        }

        maxwrk = max_i32(maxwrk, (i32)dwork[iwrk] + iwrk);

        if (fabs(dwork[isv + 2] / dwork[isv] - toldef) < toldef / ten) {
            *iwarn = 1;
        }

        if (*nr > 1) {
            i32 nr_m1 = *nr - 1;
            SLC_DLASET("L", &nr_m1, &ranke_val, &zero, &zero, &e[1], &lde);
            SLC_DLASET("F", &nr_val, &na, &zero, &zero, &e[(*ranke) * lde], &lde);
        }
    } else {
        ranka = 0;
        na = 0;
        if (*ranke > 1) {
            i32 ranke_m1 = *ranke - 1;
            SLC_DLASET("L", &ranke_m1, &ranke_m1, &zero, &zero, &e[1], &lde);
        }
    }

    dwork[0] = (f64)maxwrk;

    if (na == ranka) {
        return true;
    } else {
        return false;
    }
}
