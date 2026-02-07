// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

#define NSMAX 15

void mb03xp(const char *job, const char *compq, const char *compz,
            i32 n, i32 ilo, i32 ihi, f64 *a, i32 lda, f64 *b, i32 ldb,
            f64 *q, i32 ldq, f64 *z, i32 ldz,
            f64 *alphar, f64 *alphai, f64 *beta,
            f64 *dwork, i32 ldwork, i32 *info) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 NEGONE = -1.0;
    const i32 INT1 = 1;
    const i32 INT3 = 3;

    bool wantt = (job[0] == 'S' || job[0] == 's');
    bool initq = (compq[0] == 'I' || compq[0] == 'i');
    bool wantq = initq || (compq[0] == 'V' || compq[0] == 'v');
    bool initz = (compz[0] == 'I' || compz[0] == 'i');
    bool wantz = initz || (compz[0] == 'V' || compz[0] == 'v');

    *info = 0;

    if (!(job[0] == 'E' || job[0] == 'e') && !wantt) {
        *info = -1;
    } else if (!(compq[0] == 'N' || compq[0] == 'n') && !wantq) {
        *info = -2;
    } else if (!(compz[0] == 'N' || compz[0] == 'n') && !wantz) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (ilo < 1 || ilo > (n > 0 ? n + 1 : 1)) {
        *info = -5;
    } else if (ihi < (ilo < n ? ilo : n) || ihi > n) {
        *info = -6;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -8;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -10;
    } else if (ldq < 1 || (wantq && ldq < n)) {
        *info = -12;
    } else if (ldz < 1 || (wantz && ldz < n)) {
        *info = -14;
    } else if (ldwork < (n > 1 ? n : 1)) {
        dwork[0] = (f64)(n > 1 ? n : 1);
        *info = -19;
    }

    if (*info != 0) {
        return;
    }

    if (initq) {
        SLC_DLASET("All", &n, &n, &ZERO, &ONE, q, &ldq);
    }
    if (initz) {
        SLC_DLASET("All", &n, &n, &ZERO, &ONE, z, &ldz);
    }

    // Store isolated eigenvalues and standardize B.
    for (i32 idx = 0; idx < ilo - 1; idx++) {
        if (b[idx + idx * ldb] < ZERO) {
            if (!wantt) {
                b[idx + idx * ldb] = -b[idx + idx * ldb];
                a[idx + idx * lda] = -a[idx + idx * lda];
            }
        }
        alphar[idx] = a[idx + idx * lda];
        alphai[idx] = ZERO;
        beta[idx] = b[idx + idx * ldb];
    }
    for (i32 idx = ihi; idx < n; idx++) {
        if (b[idx + idx * ldb] < ZERO) {
            if (!wantt) {
                b[idx + idx * ldb] = -b[idx + idx * ldb];
                a[idx + idx * lda] = -a[idx + idx * lda];
            }
        }
        alphar[idx] = a[idx + idx * lda];
        alphai[idx] = ZERO;
        beta[idx] = b[idx + idx * ldb];
    }

    if (n == 0 || ilo == ihi + 1) {
        dwork[0] = ONE;
        return;
    }

    for (i32 j = ilo - 1; j < ihi - 2; j++) {
        for (i32 i = j + 2; i < n; i++) {
            a[i + j * lda] = ZERO;
        }
    }
    for (i32 j = ilo - 1; j < ihi - 1; j++) {
        for (i32 i = j + 1; i < n; i++) {
            b[i + j * ldb] = ZERO;
        }
    }

    i32 nh = ihi - ilo + 1;

    char opts[3];
    opts[0] = job[0];
    opts[1] = wantq ? compq[0] : compz[0];
    opts[2] = '\0';

    i32 ns   = ue01md(4, "MB03XP", opts, n, ilo, ihi);
    i32 maxb = ue01md(8, "MB03XP", opts, n, ilo, ihi);

    if (ns <= 2 || ns > nh || maxb >= nh) {
        i32 ierr;
        mb03yd(wantt, wantq, wantz, n, ilo, ihi, ilo, ihi,
               a, lda, b, ldb, q, ldq, z, ldz,
               alphar, alphai, beta, dwork, ldwork, &ierr);
        *info = ierr;
        dwork[0] = (f64)(n > 1 ? n : 1);
        return;
    }

    maxb = maxb > 3 ? maxb : 3;
    ns = ns < maxb ? ns : maxb;
    ns = ns < NSMAX ? ns : NSMAX;

    f64 unfl = SLC_DLAMCH("Safe minimum");
    f64 ovfl = ONE / unfl;
    SLC_DLABAD(&unfl, &ovfl);
    f64 ulp = SLC_DLAMCH("Precision");
    f64 smlnum = unfl * ((f64)nh / ulp);

    i32 i1, i2;
    if (wantt) {
        i1 = 1;
        i2 = n;
    } else {
        i1 = 0;
        i2 = 0;
    }

    i32 iseed[4] = {1, 0, 0, 1};
    i32 itn = 30 * nh;

    f64 as[NSMAX * NSMAX];
    f64 bs[NSMAX * NSMAX];
    f64 v[3 * NSMAX + 6];
    f64 tauv, tauw;
    i32 its;

    // Main loop. Eigenvalues i+1:ihi have converged (i is 1-based).
    i32 i = ihi;
    i32 l;

    while (i >= ilo) {
        l = ilo;

        for (its = 0; its <= itn; its++) {
            // Look for deflations in A.
            i32 k;
            for (k = i; k >= l + 1; k--) {
                f64 tst = fabs(a[(k-2) + (k-2)*lda]) + fabs(a[(k-1) + (k-1)*lda]);
                if (tst == ZERO) {
                    i32 nh_loc = i - l + 1;
                    tst = SLC_DLANHS("1", &nh_loc, &a[(l-1) + (l-1)*lda], &lda, dwork);
                }
                if (fabs(a[(k-1) + (k-2)*lda]) <= fmax(ulp * tst, smlnum))
                    break;
            }

            // Look for deflation in B if problem size > 1.
            i32 kk;
            if (i - k >= 1) {
                for (kk = i; kk >= k; kk--) {
                    f64 tst;
                    if (kk == i) {
                        tst = fabs(b[(kk-2) + (kk-1)*ldb]);
                    } else if (kk == k) {
                        tst = fabs(b[(kk-1) + kk*ldb]);
                    } else {
                        tst = fabs(b[(kk-2) + (kk-1)*ldb]) + fabs(b[(kk-1) + kk*ldb]);
                    }
                    if (tst == ZERO) {
                        i32 nh_loc = i - k + 1;
                        tst = SLC_DLANHS("1", &nh_loc, &b[(k-1) + (k-1)*ldb], &ldb, dwork);
                    }
                    if (fabs(b[(kk-1) + (kk-1)*ldb]) <= fmax(ulp * tst, smlnum))
                        break;
                }
            } else {
                kk = k - 1;
            }

            if (kk >= k) {
                b[(kk-1) + (kk-1)*ldb] = ZERO;
                mb03ya(wantt, wantq, wantz, n, k, i, ilo, ihi, kk,
                       a, lda, b, ldb, q, ldq, z, ldz, info);
                k = kk + 1;
            }
            l = k;
            if (l > ilo) {
                a[(l-1) + (l-2)*lda] = ZERO;
            }

            if (l >= i - maxb + 1)
                goto label200;

            if (!wantt) {
                i1 = l;
                i2 = i;
            }

            i32 ldas = NSMAX;
            i32 ldbs = NSMAX;
            if (its == 10 || its == 20) {
                i32 ns1 = ns + 1;
                SLC_DLARNV(&INT3, iseed, &ns1, v);
            } else {
                SLC_DLACPY("Full", &ns, &ns, &a[(i-ns) + (i-ns)*lda], &lda, as, &ldas);
                SLC_DLACPY("Full", &ns, &ns, &b[(i-ns) + (i-ns)*ldb], &ldb, bs, &ldbs);
                i32 ierr;
                mb03yd(false, false, false, ns, 1, ns, 1, ns,
                       as, ldas, bs, ldbs, q, ldq, z, ldz,
                       &alphar[i-ns], &alphai[i-ns], &beta[i-ns],
                       dwork, ldwork, &ierr);
            }

            // Compute the nonzero elements of the first column of
            // (A*B - w(1))(A*B - w(2)) .. (A*B - w(ns)).
            v[0] = ONE;
            i32 nv = 1;

            while (nv <= ns) {
                i32 itemp;
                f64 temp;

                if (nv == ns || as[nv + (nv-1)*ldas] == ZERO) {
                    // Real shift.
                    v[nv] = ZERO;
                    i32 pv2 = nv + 1;

                    SLC_DCOPY(&nv, v, &INT1, &v[pv2], &INT1);
                    SLC_DTRMV("Upper", "No transpose", "No unit diagonal",
                              &nv, &b[(l-1) + (l-1)*ldb], &ldb, &v[pv2], &INT1);
                    f64 bsnv = bs[(nv-1) + (nv-1)*ldas];
                    SLC_DSCAL(&nv, &bsnv, v, &INT1);
                    i32 len = 2*nv + 1;
                    itemp = SLC_IDAMAX(&len, v, &INT1);
                    temp = ONE / fmax(fabs(v[itemp-1]), smlnum);
                    SLC_DSCAL(&len, &temp, v, &INT1);
                    i32 nv1 = nv + 1;
                    f64 neg_asnv = -as[(nv-1) + (nv-1)*ldas];
                    SLC_DGEMV("No transpose", &nv1, &nv, &ONE,
                              &a[(l-1) + (l-1)*lda], &lda, &v[pv2], &INT1,
                              &neg_asnv, v, &INT1);
                    nv = nv + 1;
                } else {
                    // Double shift using product formulation.
                    v[nv] = ZERO;
                    v[nv+1] = ZERO;
                    i32 pv2 = nv + 2;
                    i32 pv3 = 2*nv + 4;

                    i32 nv2 = nv + 2;
                    i32 nv1 = nv + 1;
                    SLC_DCOPY(&nv2, v, &INT1, &v[pv2], &INT1);
                    SLC_DCOPY(&nv1, v, &INT1, &v[pv3], &INT1);
                    f64 bsnv1 = bs[nv + nv*ldas];
                    SLC_DSCAL(&nv, &bsnv1, &v[pv2], &INT1);
                    SLC_DTRMV("Upper", "No transpose", "No unit diagonal",
                              &nv, &b[(l-1) + (l-1)*ldb], &ldb, &v[pv3], &INT1);
                    i32 len = 2*nv + 3;
                    itemp = SLC_IDAMAX(&len, &v[pv2], &INT1);
                    temp = ONE / fmax(fabs(v[pv2 + itemp - 1]), smlnum);
                    SLC_DSCAL(&len, &temp, &v[pv2], &INT1);

                    SLC_DCOPY(&nv, &v[pv2], &INT1, v, &INT1);
                    f64 asnv1nv1 = as[nv + nv*ldas];
                    SLC_DGEMV("No transpose", &nv1, &nv, &NEGONE,
                              &a[(l-1) + (l-1)*lda], &lda, &v[pv3], &INT1,
                              &asnv1nv1, &v[pv2], &INT1);
                    f64 asnvnv1 = as[(nv-1) + nv*ldas];
                    SLC_DSCAL(&nv, &asnvnv1, v, &INT1);
                    itemp = SLC_IDAMAX(&len, v, &INT1);
                    temp = ONE / fmax(fabs(v[itemp - 1]), smlnum);
                    SLC_DSCAL(&len, &temp, v, &INT1);

                    f64 neg_asnv1nv = -as[nv + (nv-1)*ldas];
                    SLC_DSCAL(&nv, &neg_asnv1nv, v, &INT1);
                    f64 asnvnv = as[(nv-1) + (nv-1)*ldas];
                    SLC_DAXPY(&nv1, &asnvnv, &v[pv2], &INT1, v, &INT1);
                    itemp = SLC_IDAMAX(&len, v, &INT1);
                    temp = ONE / fmax(fabs(v[itemp - 1]), smlnum);
                    SLC_DSCAL(&len, &temp, v, &INT1);

                    f64 bsnvnv = bs[(nv-1) + (nv-1)*ldas];
                    SLC_DSCAL(&nv1, &bsnvnv, v, &INT1);
                    SLC_DTRMV("Upper", "No transpose", "No unit diagonal",
                              &nv1, &b[(l-1) + (l-1)*ldb], &ldb, &v[pv2], &INT1);
                    itemp = SLC_IDAMAX(&len, v, &INT1);
                    temp = ONE / fmax(fabs(v[itemp - 1]), smlnum);
                    SLC_DSCAL(&len, &temp, v, &INT1);

                    SLC_DGEMV("No transpose", &nv2, &nv1, &NEGONE,
                              &a[(l-1) + (l-1)*lda], &lda, &v[pv2], &INT1,
                              &ONE, v, &INT1);
                    nv = nv + 2;
                }

                itemp = SLC_IDAMAX(&nv, v, &INT1);
                temp = fabs(v[itemp - 1]);
                if (temp == ZERO) {
                    v[0] = ONE;
                    for (i32 kk2 = 1; kk2 < nv; kk2++)
                        v[kk2] = ZERO;
                } else {
                    temp = fmax(temp, smlnum);
                    f64 inv_temp = ONE / temp;
                    SLC_DSCAL(&nv, &inv_temp, v, &INT1);
                }
            }

            // Multi-shift product QR step.
            i32 pv2 = ns + 1;

            for (i32 k2 = l; k2 <= i - 1; k2++) {
                i32 nr = ns + 1;
                if (nr > i - k2 + 1) nr = i - k2 + 1;

                if (k2 > l)
                    SLC_DCOPY(&nr, &a[(k2-1) + (k2-2)*lda], &INT1, v, &INT1);

                SLC_DLARFG(&nr, &v[0], &v[1], &INT1, &tauv);

                if (k2 > l) {
                    a[(k2-1) + (k2-2)*lda] = v[0];
                    for (i32 kk2 = k2; kk2 < i; kk2++)
                        a[kk2 + (k2-2)*lda] = ZERO;
                }

                // Apply reflector V from the right to B in rows I1:min(K+NS,I).
                v[0] = ONE;
                i32 kns = k2 + ns;
                if (kns > i) kns = i;
                i32 nrows = kns - i1 + 1;
                SLC_DLARFX("Right", &nrows, &nr, v, &tauv,
                           &b[(i1-1) + (k2-1)*ldb], &ldb, dwork);

                // Annihilate the introduced nonzeros in the K-th column.
                SLC_DCOPY(&nr, &b[(k2-1) + (k2-1)*ldb], &INT1, &v[pv2], &INT1);
                SLC_DLARFG(&nr, &v[pv2], &v[pv2+1], &INT1, &tauw);
                b[(k2-1) + (k2-1)*ldb] = v[pv2];
                for (i32 kk2 = k2; kk2 < i; kk2++)
                    b[kk2 + (k2-1)*ldb] = ZERO;
                v[pv2] = ONE;

                // Apply reflector W from the left to B in columns K+1:I2.
                i32 ncols = i2 - k2;
                SLC_DLARFX("Left", &nr, &ncols, &v[pv2], &tauw,
                           &b[(k2-1) + k2*ldb], &ldb, dwork);

                // Apply reflector V from the left to A in columns K:I2.
                i32 ncols2 = i2 - k2 + 1;
                SLC_DLARFX("Left", &nr, &ncols2, v, &tauv,
                           &a[(k2-1) + (k2-1)*lda], &lda, dwork);

                // Apply reflector W from the right to A in rows I1:min(K+NS+1,I).
                i32 kns1 = k2 + ns + 1;
                if (kns1 > i) kns1 = i;
                i32 nrows2 = kns1 - i1 + 1;
                SLC_DLARFX("Right", &nrows2, &nr, &v[pv2], &tauw,
                           &a[(i1-1) + (k2-1)*lda], &lda, dwork);

                // Accumulate transformations in Q and Z.
                if (wantq) {
                    SLC_DLARFX("Right", &nh, &nr, v, &tauv,
                               &q[(ilo-1) + (k2-1)*ldq], &ldq, dwork);
                }
                if (wantz) {
                    SLC_DLARFX("Right", &nh, &nr, &v[pv2], &tauw,
                               &z[(ilo-1) + (k2-1)*ldz], &ldz, dwork);
                }
            }
        }

        // Failure to converge.
        *info = i;
        return;

label200:
        {
            i32 ierr;
            mb03yd(wantt, wantq, wantz, n, l, i, ilo, ihi,
                   a, lda, b, ldb, q, ldq, z, ldz,
                   alphar, alphai, beta, dwork, ldwork, &ierr);
            if (ierr > 0) {
                *info = ierr;
                return;
            }
        }
        itn = itn - its;
        i = l - 1;
    }

    dwork[0] = (f64)(n > 1 ? n : 1);
}
