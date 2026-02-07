// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void mb03yd(bool wantt, bool wantq, bool wantz, i32 n, i32 ilo, i32 ihi,
            i32 iloq, i32 ihiq, f64 *a, i32 lda, f64 *b, i32 ldb,
            f64 *q, i32 ldq, f64 *z, i32 ldz,
            f64 *alphar, f64 *alphai, f64 *beta,
            f64 *dwork, i32 ldwork, i32 *info) {
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const i32 IONE = 1;

    i32 nh = ihi - ilo + 1;
    i32 nq = ihiq - iloq + 1;
    *info = 0;

    if (n < 0) {
        *info = -4;
    } else if (ilo < 1 || ilo > (n > 1 ? n : 1)) {
        *info = -5;
    } else if (ihi < (ilo < n ? ilo : n) || ihi > n) {
        *info = -6;
    } else if (iloq < 1 || iloq > ilo) {
        *info = -7;
    } else if (ihiq < ihi || ihiq > n) {
        *info = -8;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -10;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -12;
    } else if (ldq < 1 || (wantq && ldq < n)) {
        *info = -14;
    } else if (ldz < 1 || (wantz && ldz < n)) {
        *info = -16;
    } else if (ldwork < (n > 1 ? n : 1)) {
        dwork[0] = (f64)(n > 1 ? n : 1);
        *info = -21;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        return;
    }

    f64 unfl = SLC_DLAMCH("S");
    f64 ovfl = ONE / unfl;
    SLC_DLABAD(&unfl, &ovfl);
    f64 ulp = SLC_DLAMCH("P");
    f64 smlnum = unfl * ((f64)nh / ulp);

    i32 i1, i2;
    if (wantt) {
        i1 = 0;
        i2 = n - 1;
    } else {
        i1 = ilo - 1;
        i2 = ihi - 1;
    }

    i32 iseed[4] = {1, 0, 0, 1};
    i32 itn = 30 * nh;

    i32 i = ihi - 1;
    i32 l, k, kk, its;

    while (i >= ilo - 1) {
        l = ilo - 1;

        for (its = 0; its <= itn; its++) {
            for (k = i; k >= l + 1; k--) {
                f64 tst = fabs(a[k - 1 + (k - 1) * lda]) + fabs(a[k + k * lda]);
                if (tst == ZERO) {
                    i32 len = i - l + 1;
                    tst = SLC_DLANHS("1", &len, &a[l + l * lda], &lda, dwork);
                }
                if (fabs(a[k + (k - 1) * lda]) <= fmax(ulp * tst, smlnum)) {
                    break;
                }
            }

            if (i - k >= 1) {
                for (kk = i; kk >= k; kk--) {
                    f64 tst;
                    if (kk == i) {
                        tst = fabs(b[kk - 1 + kk * ldb]);
                    } else if (kk == k) {
                        tst = fabs(b[kk + (kk + 1) * ldb]);
                    } else {
                        tst = fabs(b[kk - 1 + kk * ldb]) + fabs(b[kk + (kk + 1) * ldb]);
                    }
                    if (tst == ZERO) {
                        i32 len = i - k + 1;
                        tst = SLC_DLANHS("1", &len, &b[k + k * ldb], &ldb, dwork);
                    }
                    if (fabs(b[kk + kk * ldb]) <= fmax(ulp * tst, smlnum)) {
                        break;
                    }
                }
            } else {
                kk = k - 1;
            }

            if (kk >= k) {
                b[kk + kk * ldb] = ZERO;
                i32 info_ya;
                mb03ya(wantt, wantq, wantz, n, k + 1, i + 1, iloq, ihiq, kk + 1,
                       a, lda, b, ldb, q, ldq, z, ldz, &info_ya);
                k = kk + 1;
            }

            l = k;
            if (l > ilo - 1) {
                a[l + (l - 1) * lda] = ZERO;
            }

            if (l >= i - 1) {
                goto converged;
            }

            if (!wantt) {
                i1 = l;
                i2 = i;
            }

            f64 v[3], w[3];
            f64 cs1, sn1, cs2, sn2, cs3, sn3;
            f64 alpha_val, betax, gamma_val, delta, temp, tauv, tauw;

            if (its == 10 || its == 20) {
                i32 three = 3;
                SLC_DLARNV(&three, iseed, &three, v);
            } else {
                SLC_DLARTG(&b[l + l * ldb], &b[i + i * ldb], &cs2, &sn2, &temp);
                SLC_DLARTG(&temp, &b[i - 1 + i * ldb], &cs1, &sn1, &alpha_val);

                alpha_val = a[l + l * lda] * cs2 - a[i + i * lda] * sn2;
                betax = cs1 * (cs2 * a[l + 1 + l * lda]);
                gamma_val = cs1 * (sn2 * a[i - 1 + i * lda]) + sn1 * a[i - 1 + (i - 1) * lda];
                alpha_val = alpha_val * cs1 - a[i + (i - 1) * lda] * sn1;
                SLC_DLARTG(&alpha_val, &betax, &cs1, &sn1, &temp);

                SLC_DLARTG(&temp, &gamma_val, &cs2, &sn2, &alpha_val);
                alpha_val = cs2;
                gamma_val = (a[i - 1 + (i - 1) * lda] * cs1) * cs2 + a[i + (i - 1) * lda] * sn2;
                delta = (a[i - 1 + (i - 1) * lda] * sn1) * cs2;
                SLC_DLARTG(&gamma_val, &delta, &cs3, &sn3, &temp);
                SLC_DLARTG(&alpha_val, &temp, &cs2, &sn2, &alpha_val);

                alpha_val = (b[l + l * ldb] * cs1 + b[l + (l + 1) * ldb] * sn1) * cs2;
                betax = (b[l + 1 + (l + 1) * ldb] * sn1) * cs2;
                gamma_val = b[i - 1 + (i - 1) * ldb] * sn2;
                SLC_DLARTG(&alpha_val, &betax, &cs1, &sn1, &temp);
                SLC_DLARTG(&temp, &gamma_val, &cs2, &sn2, &alpha_val);

                alpha_val = cs1 * a[l + l * lda] + sn1 * a[l + (l + 1) * lda];
                betax = cs1 * a[l + 1 + l * lda] + sn1 * a[l + 1 + (l + 1) * lda];
                gamma_val = sn1 * a[l + 2 + (l + 1) * lda];

                v[0] = cs2 * alpha_val - sn2 * cs3;
                v[1] = cs2 * betax - sn2 * sn3;
                v[2] = gamma_val * cs2;
            }

            for (k = l; k <= i - 1; k++) {
                i32 nr = (3 < i - k + 1) ? 3 : (i - k + 1);

                if (k > l) {
                    for (i32 ii = 0; ii < nr; ii++) {
                        v[ii] = a[k + ii + (k - 1) * lda];
                    }
                }

                SLC_DLARFG(&nr, &v[0], &v[1], &IONE, &tauv);

                if (k > l) {
                    a[k + (k - 1) * lda] = v[0];
                    a[k + 1 + (k - 1) * lda] = ZERO;
                    if (k < i - 1) {
                        a[k + 2 + (k - 1) * lda] = ZERO;
                    }
                }

                v[0] = ONE;
                i32 len_rows = ((k + 2 < i) ? (k + 2) : i) - i1 + 1;
                SLC_DLARFX("R", &len_rows, &nr, v, &tauv, &b[i1 + k * ldb], &ldb, dwork);

                for (i32 ii = 0; ii < nr; ii++) {
                    w[ii] = b[k + ii + k * ldb];
                }
                SLC_DLARFG(&nr, &w[0], &w[1], &IONE, &tauw);
                b[k + k * ldb] = w[0];
                b[k + 1 + k * ldb] = ZERO;
                if (k < i - 1) {
                    b[k + 2 + k * ldb] = ZERO;
                }

                w[0] = ONE;
                i32 len_cols = i2 - k;
                SLC_DLARFX("L", &nr, &len_cols, w, &tauw, &b[k + (k + 1) * ldb], &ldb, dwork);

                i32 len_a_cols = i2 - k + 1;
                SLC_DLARFX("L", &nr, &len_a_cols, v, &tauv, &a[k + k * lda], &lda, dwork);

                i32 len_a_rows = ((k + 3 < i) ? (k + 3) : i) - i1 + 1;
                SLC_DLARFX("R", &len_a_rows, &nr, w, &tauw, &a[i1 + k * lda], &lda, dwork);

                if (wantq) {
                    SLC_DLARFX("R", &nq, &nr, v, &tauv, &q[iloq - 1 + k * ldq], &ldq, dwork);
                }
                if (wantz) {
                    SLC_DLARFX("R", &nq, &nr, w, &tauw, &z[iloq - 1 + k * ldz], &ldz, dwork);
                }
            }
        }

        *info = i + 1;
        return;

    converged:
        if (l == i) {
            if (b[i + i * ldb] < ZERO) {
                if (wantt) {
                    for (k = i1; k <= i; k++) {
                        b[k + i * ldb] = -b[k + i * ldb];
                    }
                    for (k = i; k <= i2; k++) {
                        a[i + k * lda] = -a[i + k * lda];
                    }
                } else {
                    b[i + i * ldb] = -b[i + i * ldb];
                    a[i + i * lda] = -a[i + i * lda];
                }
                if (wantq) {
                    for (k = iloq - 1; k < ihiq; k++) {
                        q[k + i * ldq] = -q[k + i * ldq];
                    }
                }
            }
            alphar[i] = a[i + i * lda];
            alphai[i] = ZERO;
            beta[i] = b[i + i * ldb];
        } else if (l == i - 1) {
            f64 csl, snl, csr, snr;
            mb03yt(&a[i - 1 + (i - 1) * lda], lda, &b[i - 1 + (i - 1) * ldb], ldb,
                   &alphar[i - 1], &alphai[i - 1], &beta[i - 1],
                   &csl, &snl, &csr, &snr);

            if (i2 > i) {
                i32 len = i2 - i;
                SLC_DROT(&len, &a[i - 1 + (i + 1) * lda], &lda,
                         &a[i + (i + 1) * lda], &lda, &csl, &snl);
            }
            i32 len1 = i - i1 - 1;
            SLC_DROT(&len1, &a[i1 + (i - 1) * lda], &IONE,
                     &a[i1 + i * lda], &IONE, &csr, &snr);
            if (i2 > i) {
                i32 len = i2 - i;
                SLC_DROT(&len, &b[i - 1 + (i + 1) * ldb], &ldb,
                         &b[i + (i + 1) * ldb], &ldb, &csr, &snr);
            }
            SLC_DROT(&len1, &b[i1 + (i - 1) * ldb], &IONE,
                     &b[i1 + i * ldb], &IONE, &csl, &snl);

            if (wantq) {
                SLC_DROT(&nq, &q[iloq - 1 + (i - 1) * ldq], &IONE,
                         &q[iloq - 1 + i * ldq], &IONE, &csl, &snl);
            }
            if (wantz) {
                SLC_DROT(&nq, &z[iloq - 1 + (i - 1) * ldz], &IONE,
                         &z[iloq - 1 + i * ldz], &IONE, &csr, &snr);
            }
        }

        itn = itn - (its + 1);
        i = l - 1;
    }

    dwork[0] = (f64)(n > 1 ? n : 1);
}
