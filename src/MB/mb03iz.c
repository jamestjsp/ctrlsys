// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

void mb03iz(const char *compq, const char *compu, i32 n, c128 *a, i32 lda,
            c128 *c, i32 ldc, c128 *d, i32 ldd, c128 *b, i32 ldb, c128 *f,
            i32 ldf, c128 *q, i32 ldq, c128 *u1, i32 ldu1, c128 *u2, i32 ldu2,
            i32 *neig, f64 tol, i32 *info) {
    const f64 zero = 0.0;
    const f64 ten = 10.0;
    const c128 czero = 0.0 + 0.0 * I;
    const c128 cone = 1.0 + 0.0 * I;

    i32 m = n / 2;
    *neig = 0;

    bool liniq = (compq[0] == 'I' || compq[0] == 'i');
    bool lupdq = (compq[0] == 'U' || compq[0] == 'u');
    bool lcmpq = liniq || lupdq;

    bool liniu = (compu[0] == 'I' || compu[0] == 'i');
    bool lupdu = (compu[0] == 'U' || compu[0] == 'u');
    bool lcmpu = liniu || lupdu;

    *info = 0;
    bool compq_n = (compq[0] == 'N' || compq[0] == 'n');
    bool compu_n = (compu[0] == 'N' || compu[0] == 'n');

    if (!compq_n && !lcmpq) {
        *info = -1;
    } else if (!compu_n && !lcmpu) {
        *info = -2;
    } else if (n < 0 || (n % 2) != 0) {
        *info = -3;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -5;
    } else if (ldc < (m > 1 ? m : 1)) {
        *info = -7;
    } else if (ldd < (m > 1 ? m : 1)) {
        *info = -9;
    } else if (ldb < (m > 1 ? m : 1)) {
        *info = -11;
    } else if (ldf < (m > 1 ? m : 1)) {
        *info = -13;
    } else if (ldq < 1 || (lcmpq && ldq < n)) {
        *info = -15;
    } else if (ldu1 < 1 || (lcmpu && ldu1 < m)) {
        *info = -17;
    } else if (ldu2 < 1 || (lcmpu && ldu2 < m)) {
        *info = -19;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        return;
    }

    f64 eps = tol;
    if (eps <= zero) {
        f64 n_dbl = (f64)n;
        f64 min_val = (n_dbl < ten) ? n_dbl : ten;
        eps = min_val * SLC_DLAMCH("P");
    }

    i32 iupd, upds;
    if (liniq) {
        iupd = m;
        upds = m;
        SLC_ZLASET("F", &n, &n, &czero, &cone, q, &ldq);
    } else if (lupdq) {
        iupd = 0;
        upds = n;
    }

    if (liniu) {
        SLC_ZLASET("F", &m, &m, &czero, &cone, u1, &ldu1);
        SLC_ZLASET("F", &m, &m, &czero, &czero, u2, &ldu2);
    }

    i32 mm = 0;
    i32 mp = m + 1;

    f64 dum[1];
    f64 nrma = SLC_ZLANTR("O", "U", "N", &m, &m, a, &lda, dum) *
               SLC_ZLANTR("O", "L", "N", &m, &m, c, &ldc, dum);
    f64 nrmb = SLC_ZLANTR("O", "U", "N", &m, &m, b, &ldb, dum);

    c128 hlp[4];
    i32 hlp_ld = 2;
    f64 co1, co2, co3;
    c128 si1, si2, si3;
    c128 cjf, tmp;
    i32 int1 = 1;

    for (i32 k = 0; k < m; k++) {
        c128 bkk = b[k + k * ldb];
        c128 ckk = c[k + k * ldc];
        c128 akk_conj = conj(a[k + k * lda]);

        f64 test_val = creal(bkk * ckk * akk_conj) * nrmb;

        if (test_val <= -eps * nrma) {
            for (i32 j = k - 1; j >= mm; j--) {
                hlp[0] = conj(c[j + j * ldc]);
                hlp[2] = conj(c[(j + 1) + j * ldc]);
                hlp[3] = conj(c[(j + 1) + (j + 1) * ldc]);

                mb03cz(hlp, hlp_ld, &a[j + j * lda], lda, &b[j + j * ldb], ldb,
                       &co1, &si1, &co2, &si2, &co3, &si3);

                i32 len_j = j + 1;
                SLC_ZROT(&len_j, &a[0 + (j + 1) * lda], &int1, &a[0 + j * lda],
                         &int1, &co1, &si1);
                a[j + j * lda] = co2 * a[j + j * lda] +
                                 si2 * a[(j + 1) + (j + 1) * lda] * conj(si1);
                a[(j + 1) + (j + 1) * lda] = co1 * a[(j + 1) + (j + 1) * lda];
                i32 len_mj = m - j - 1;
                c128 neg_si2 = -si2;
                SLC_ZROT(&len_mj, &a[j + (j + 1) * lda], &lda,
                         &a[(j + 1) + (j + 1) * lda], &lda, &co2, &neg_si2);

                SLC_ZROT(&m, &d[0 + (j + 1) * ldd], &int1, &d[0 + j * ldd],
                         &int1, &co3, &si3);
                SLC_ZROT(&m, &d[j + 0 * ldd], &ldd, &d[(j + 1) + 0 * ldd], &ldd,
                         &co2, &neg_si2);

                SLC_ZROT(&len_mj, &c[(j + 1) + (j + 1) * ldc], &int1,
                         &c[(j + 1) + j * ldc], &int1, &co3, &si3);
                c[(j + 1) + (j + 1) * ldc] =
                    co2 * c[(j + 1) + (j + 1) * ldc] +
                    si3 * c[j + j * ldc] * conj(si2);
                c[j + j * ldc] = co3 * c[j + j * ldc];
                SLC_ZROT(&len_j, &c[j + 0 * ldc], &ldc, &c[(j + 1) + 0 * ldc],
                         &ldc, &co2, &neg_si2);

                SLC_ZROT(&len_j, &b[0 + (j + 1) * ldb], &int1, &b[0 + j * ldb],
                         &int1, &co1, &si1);
                b[j + j * ldb] = co3 * b[j + j * ldb] +
                                 si3 * b[(j + 1) + (j + 1) * ldb] * conj(si1);
                b[(j + 1) + (j + 1) * ldb] = co1 * b[(j + 1) + (j + 1) * ldb];
                c128 neg_si3 = -si3;
                SLC_ZROT(&len_mj, &b[j + (j + 1) * ldb], &ldb,
                         &b[(j + 1) + (j + 1) * ldb], &ldb, &co3, &neg_si3);

                cjf = conj(f[j + (j + 1) * ldf]);
                tmp = co3 * cjf - conj(si3) * f[(j + 1) + (j + 1) * ldf];
                SLC_ZROT(&len_j, &f[0 + (j + 1) * ldf], &int1, &f[0 + j * ldf],
                         &int1, &co3, &si3);
                f[j + j * ldf] = co3 * f[j + j * ldf] - si3 * tmp;
                f[(j + 1) + (j + 1) * ldf] =
                    co3 * f[(j + 1) + (j + 1) * ldf] + si3 * cjf;
                SLC_ZROT(&len_mj, &f[j + (j + 1) * ldf], &ldf,
                         &f[(j + 1) + (j + 1) * ldf], &ldf, &co3, &neg_si3);

                if (lcmpq) {
                    SLC_ZROT(&upds, &q[0 + (j + 1) * ldq], &int1,
                             &q[0 + j * ldq], &int1, &co1, &si1);
                    SLC_ZROT(&upds, &q[iupd + (m + j + 1) * ldq], &int1,
                             &q[iupd + (m + j) * ldq], &int1, &co3, &si3);
                }

                if (lcmpu) {
                    SLC_ZROT(&m, &u1[0 + (j + 1) * ldu1], &int1,
                             &u1[0 + j * ldu1], &int1, &co2, &si2);
                    if (lupdu) {
                        SLC_ZROT(&m, &u2[0 + (j + 1) * ldu2], &int1,
                                 &u2[0 + j * ldu2], &int1, &co2, &si2);
                    }
                }
            }
            mm++;
        }
    }

    for (i32 k = m - 1; k >= mm; k--) {
        c128 bkk = b[k + k * ldb];
        c128 ckk = c[k + k * ldc];
        c128 akk_conj = conj(a[k + k * lda]);

        f64 test_val = creal(bkk * ckk * akk_conj) * nrmb;

        if (test_val >= eps * nrma) {
            for (i32 j = k; j < mp - 2; j++) {
                hlp[0] = conj(c[j + j * ldc]);
                hlp[2] = conj(c[(j + 1) + j * ldc]);
                hlp[3] = conj(c[(j + 1) + (j + 1) * ldc]);

                mb03cz(hlp, hlp_ld, &a[j + j * lda], lda, &b[j + j * ldb], ldb,
                       &co1, &si1, &co2, &si2, &co3, &si3);

                i32 len_j = j + 1;
                SLC_ZROT(&len_j, &a[0 + (j + 1) * lda], &int1, &a[0 + j * lda],
                         &int1, &co1, &si1);
                a[j + j * lda] = co2 * a[j + j * lda] +
                                 si2 * a[(j + 1) + (j + 1) * lda] * conj(si1);
                a[(j + 1) + (j + 1) * lda] = co1 * a[(j + 1) + (j + 1) * lda];
                i32 len_mj = m - j - 1;
                c128 neg_si2 = -si2;
                SLC_ZROT(&len_mj, &a[j + (j + 1) * lda], &lda,
                         &a[(j + 1) + (j + 1) * lda], &lda, &co2, &neg_si2);

                SLC_ZROT(&m, &d[0 + (j + 1) * ldd], &int1, &d[0 + j * ldd],
                         &int1, &co3, &si3);
                SLC_ZROT(&m, &d[j + 0 * ldd], &ldd, &d[(j + 1) + 0 * ldd], &ldd,
                         &co2, &neg_si2);

                SLC_ZROT(&len_mj, &c[(j + 1) + (j + 1) * ldc], &int1,
                         &c[(j + 1) + j * ldc], &int1, &co3, &si3);
                c[(j + 1) + (j + 1) * ldc] =
                    co2 * c[(j + 1) + (j + 1) * ldc] +
                    si3 * c[j + j * ldc] * conj(si2);
                c[j + j * ldc] = co3 * c[j + j * ldc];
                SLC_ZROT(&len_j, &c[j + 0 * ldc], &ldc, &c[(j + 1) + 0 * ldc],
                         &ldc, &co2, &neg_si2);

                SLC_ZROT(&len_j, &b[0 + (j + 1) * ldb], &int1, &b[0 + j * ldb],
                         &int1, &co1, &si1);
                b[j + j * ldb] = co3 * b[j + j * ldb] +
                                 si3 * b[(j + 1) + (j + 1) * ldb] * conj(si1);
                b[(j + 1) + (j + 1) * ldb] = co1 * b[(j + 1) + (j + 1) * ldb];
                c128 neg_si3 = -si3;
                SLC_ZROT(&len_mj, &b[j + (j + 1) * ldb], &ldb,
                         &b[(j + 1) + (j + 1) * ldb], &ldb, &co3, &neg_si3);

                cjf = conj(f[j + (j + 1) * ldf]);
                tmp = co3 * cjf - conj(si3) * f[(j + 1) + (j + 1) * ldf];
                SLC_ZROT(&len_j, &f[0 + (j + 1) * ldf], &int1, &f[0 + j * ldf],
                         &int1, &co3, &si3);
                f[j + j * ldf] = co3 * f[j + j * ldf] - si3 * tmp;
                f[(j + 1) + (j + 1) * ldf] =
                    co3 * f[(j + 1) + (j + 1) * ldf] + si3 * cjf;
                SLC_ZROT(&len_mj, &f[j + (j + 1) * ldf], &ldf,
                         &f[(j + 1) + (j + 1) * ldf], &ldf, &co3, &neg_si3);

                if (lcmpq) {
                    SLC_ZROT(&upds, &q[0 + (j + 1) * ldq], &int1,
                             &q[0 + j * ldq], &int1, &co1, &si1);
                    SLC_ZROT(&upds, &q[iupd + (m + j + 1) * ldq], &int1,
                             &q[iupd + (m + j) * ldq], &int1, &co3, &si3);
                }

                if (lcmpu) {
                    SLC_ZROT(&m, &u1[0 + (j + 1) * ldu1], &int1,
                             &u1[0 + j * ldu1], &int1, &co2, &si2);
                    if (lupdu) {
                        SLC_ZROT(&m, &u2[0 + (j + 1) * ldu2], &int1,
                                 &u2[0 + j * ldu2], &int1, &co2, &si2);
                    }
                }
            }
            mp--;
        }
    }

    for (i32 k = m - 1; k >= mp - 1; k--) {
        i32 mlast = m - 1;
        mb03gz(a[mlast + mlast * lda], d[mlast + mlast * ldd],
               c[mlast + mlast * ldc], b[mlast + mlast * ldb],
               f[mlast + mlast * ldf], &co1, &si1, &co2, &si2);

        SLC_ZROT(&m, &d[0 + mlast * ldd], &int1, &a[0 + mlast * lda], &int1,
                 &co1, &si1);
        tmp = -conj(si1) * c[mlast + mlast * ldc];
        c[mlast + mlast * ldc] = co1 * c[mlast + mlast * ldc];
        c128 neg_conj_si2 = -conj(si2);
        SLC_ZROT(&m, &d[mlast + 0 * ldd], &ldd, &c[mlast + 0 * ldc], &ldc, &co2,
                 &neg_conj_si2);
        a[mlast + mlast * lda] =
            co2 * a[mlast + mlast * lda] - conj(si2) * tmp;

        tmp = -conj(b[mlast + mlast * ldb]);
        SLC_ZROT(&m, &f[0 + mlast * ldf], &int1, &b[0 + mlast * ldb], &int1,
                 &co1, &si1);
        b[mlast + mlast * ldb] = b[mlast + mlast * ldb] * co1 +
                                 tmp * conj(si1) * conj(si1);
        f[mlast + mlast * ldf] =
            f[mlast + mlast * ldf] * co1 - tmp * conj(si1) * co1;

        if (lcmpq) {
            i32 nlast = n - 1;
            SLC_ZROT(&n, &q[0 + nlast * ldq], &int1, &q[0 + mlast * ldq], &int1,
                     &co1, &si1);
        }

        if (lcmpu) {
            SLC_ZROT(&m, &u2[0 + mlast * ldu2], &int1, &u1[0 + mlast * ldu1],
                     &int1, &co2, &si2);
        }

        mm++;
        for (i32 j = m - 2; j >= mm - 1; j--) {
            hlp[0] = conj(c[j + j * ldc]);
            hlp[2] = conj(c[(j + 1) + j * ldc]);
            hlp[3] = conj(c[(j + 1) + (j + 1) * ldc]);

            mb03cz(hlp, hlp_ld, &a[j + j * lda], lda, &b[j + j * ldb], ldb,
                   &co1, &si1, &co2, &si2, &co3, &si3);

            i32 len_j = j + 1;
            SLC_ZROT(&len_j, &a[0 + (j + 1) * lda], &int1, &a[0 + j * lda],
                     &int1, &co1, &si1);
            a[j + j * lda] = co2 * a[j + j * lda] +
                             si2 * a[(j + 1) + (j + 1) * lda] * conj(si1);
            a[(j + 1) + (j + 1) * lda] = co1 * a[(j + 1) + (j + 1) * lda];
            i32 len_mj = m - j - 1;
            c128 neg_si2 = -si2;
            SLC_ZROT(&len_mj, &a[j + (j + 1) * lda], &lda,
                     &a[(j + 1) + (j + 1) * lda], &lda, &co2, &neg_si2);

            SLC_ZROT(&m, &d[0 + (j + 1) * ldd], &int1, &d[0 + j * ldd], &int1,
                     &co3, &si3);
            SLC_ZROT(&m, &d[j + 0 * ldd], &ldd, &d[(j + 1) + 0 * ldd], &ldd,
                     &co2, &neg_si2);

            SLC_ZROT(&len_mj, &c[(j + 1) + (j + 1) * ldc], &int1,
                     &c[(j + 1) + j * ldc], &int1, &co3, &si3);
            c[(j + 1) + (j + 1) * ldc] = co2 * c[(j + 1) + (j + 1) * ldc] +
                                         si3 * c[j + j * ldc] * conj(si2);
            c[j + j * ldc] = co3 * c[j + j * ldc];
            SLC_ZROT(&len_j, &c[j + 0 * ldc], &ldc, &c[(j + 1) + 0 * ldc], &ldc,
                     &co2, &neg_si2);

            SLC_ZROT(&len_j, &b[0 + (j + 1) * ldb], &int1, &b[0 + j * ldb],
                     &int1, &co1, &si1);
            b[j + j * ldb] = co3 * b[j + j * ldb] +
                             si3 * b[(j + 1) + (j + 1) * ldb] * conj(si1);
            b[(j + 1) + (j + 1) * ldb] = co1 * b[(j + 1) + (j + 1) * ldb];
            c128 neg_si3 = -si3;
            SLC_ZROT(&len_mj, &b[j + (j + 1) * ldb], &ldb,
                     &b[(j + 1) + (j + 1) * ldb], &ldb, &co3, &neg_si3);

            cjf = conj(f[j + (j + 1) * ldf]);
            tmp = co3 * cjf - conj(si3) * f[(j + 1) + (j + 1) * ldf];
            SLC_ZROT(&len_j, &f[0 + (j + 1) * ldf], &int1, &f[0 + j * ldf],
                     &int1, &co3, &si3);
            f[j + j * ldf] = co3 * f[j + j * ldf] - si3 * tmp;
            f[(j + 1) + (j + 1) * ldf] =
                co3 * f[(j + 1) + (j + 1) * ldf] + si3 * cjf;
            SLC_ZROT(&len_mj, &f[j + (j + 1) * ldf], &ldf,
                     &f[(j + 1) + (j + 1) * ldf], &ldf, &co3, &neg_si3);

            if (lcmpq) {
                SLC_ZROT(&n, &q[0 + (j + 1) * ldq], &int1, &q[0 + j * ldq],
                         &int1, &co1, &si1);
                SLC_ZROT(&n, &q[0 + (m + j + 1) * ldq], &int1,
                         &q[0 + (m + j) * ldq], &int1, &co3, &si3);
            }

            if (lcmpu) {
                SLC_ZROT(&m, &u1[0 + (j + 1) * ldu1], &int1, &u1[0 + j * ldu1],
                         &int1, &co2, &si2);
                SLC_ZROT(&m, &u2[0 + (j + 1) * ldu2], &int1, &u2[0 + j * ldu2],
                         &int1, &co2, &si2);
            }
        }
    }

    *neig = mm;
}
