#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <complex.h>
#include <ctype.h>

f64 ab13dx(const char *dico, const char *jobe, const char *jobd,
           i32 n, i32 m, i32 p, f64 omega,
           f64 *a, i32 lda, const f64 *e, i32 lde,
           f64 *b, i32 ldb, const f64 *c, i32 ldc,
           f64 *d, i32 ldd,
           i32 *iwork, f64 *dwork, i32 ldwork,
           c128 *zwork, i32 lzwork, i32 *info)
{
    char dico_upper = (char)toupper((unsigned char)dico[0]);
    char jobe_upper = (char)toupper((unsigned char)jobe[0]);
    char jobd_upper = (char)toupper((unsigned char)jobd[0]);

    int discr = (dico_upper == 'D');
    int fulle = (jobe_upper == 'G');
    int withd = (jobd_upper == 'D');

    *info = 0;

    if (!discr && dico_upper != 'C') {
        *info = -1;
        return 0.0;
    }
    if (!fulle && jobe_upper != 'I') {
        *info = -2;
        return 0.0;
    }
    if (!withd && jobd_upper != 'Z') {
        *info = -3;
        return 0.0;
    }
    if (n < 0) {
        *info = -4;
        return 0.0;
    }
    if (m < 0) {
        *info = -5;
        return 0.0;
    }
    if (p < 0) {
        *info = -6;
        return 0.0;
    }
    if (lda < (n > 1 ? n : 1)) {
        *info = -9;
        return 0.0;
    }
    if (lde < 1 || (fulle && lde < n)) {
        *info = -11;
        return 0.0;
    }
    if (ldb < (n > 1 ? n : 1)) {
        *info = -13;
        return 0.0;
    }
    if (ldc < (p > 1 ? p : 1)) {
        *info = -15;
        return 0.0;
    }
    if (ldd < 1 || (withd && ldd < p)) {
        *info = -17;
        return 0.0;
    }

    f64 bnorm = SLC_DLANGE("1", &n, &m, b, &ldb, dwork);
    f64 cnorm = SLC_DLANGE("1", &p, &n, c, &ldc, dwork);

    i32 minpm = (p < m) ? p : m;
    i32 maxpm = (p > m) ? p : m;
    int nodyn = (n == 0) || (bnorm == 0.0) || (cnorm == 0.0);
    int specl = !nodyn && (omega == 0.0) && !discr;

    i32 minwrk, mincwr;

    if (minpm == 0 || (nodyn && !withd)) {
        minwrk = 0;
    } else if (specl || (nodyn && withd)) {
        minwrk = (4 * minpm + maxpm > 6 * minpm) ? (4 * minpm + maxpm) : (6 * minpm);
        if (specl && !withd) {
            minwrk += p * m;
        }
    } else {
        minwrk = 6 * minpm;
    }
    minwrk = (minwrk > 1) ? minwrk : 1;

    if (ldwork < minwrk) {
        *info = -20;
        return 0.0;
    }

    if (nodyn || (omega == 0.0 && !discr) || minpm == 0) {
        mincwr = 1;
    } else {
        mincwr = (n + m) * (n + p) + 2 * minpm + maxpm;
        if (mincwr < 1) mincwr = 1;
    }

    if (lzwork < mincwr) {
        *info = -22;
        return 0.0;
    }

    c128 cone = 1.0 + 0.0 * I;

    if (minpm == 0) {
        dwork[0] = 1.0;
        zwork[0] = cone;
        return 0.0;
    }

    i32 is = 0;
    i32 iwrk = is + minpm;
    i32 ierr;

    if (nodyn) {
        if (withd) {
            i32 pm_size = p;
            i32 mm_size = m;
            SLC_DGESVD("N", "N", &pm_size, &mm_size, d, &ldd,
                       &dwork[is], dwork, &pm_size, dwork, &mm_size,
                       &dwork[iwrk], &ldwork, &ierr);
            if (ierr > 0) {
                *info = n + 1;
                return 0.0;
            }
            f64 result = dwork[is];
            dwork[0] = dwork[iwrk] + iwrk;
            zwork[0] = cone;
            return result;
        } else {
            dwork[0] = 1.0;
            zwork[0] = cone;
            return 0.0;
        }
    }

    if (specl) {
        mb02sd(n, a, lda, iwork, &ierr);
        if (ierr > 0) {
            *info = ierr;
            dwork[0] = 1.0;
            zwork[0] = cone;
            return 0.0;
        }

        mb02rd("N", n, m, a, lda, iwork, b, ldb, &ierr);

        if (withd) {
            f64 alpha = -1.0;
            f64 beta = 1.0;
            SLC_DGEMM("N", "N", &p, &m, &n, &alpha, c, &ldc, b, &ldb, &beta, d, &ldd);
            i32 pm_size = p;
            i32 mm_size = m;
            SLC_DGESVD("N", "N", &pm_size, &mm_size, d, &ldd,
                       &dwork[is], dwork, &pm_size, dwork, &mm_size,
                       &dwork[iwrk], &ldwork, &ierr);
        } else {
            i32 id = iwrk;
            iwrk = id + p * m;
            f64 alpha = -1.0;
            f64 beta = 0.0;
            SLC_DGEMM("N", "N", &p, &m, &n, &alpha, c, &ldc, b, &ldb, &beta, &dwork[id], &p);
            i32 pm_size = p;
            i32 mm_size = m;
            SLC_DGESVD("N", "N", &pm_size, &mm_size, &dwork[id], &pm_size,
                       &dwork[is], dwork, &pm_size, dwork, &mm_size,
                       &dwork[iwrk], &ldwork, &ierr);
        }

        if (ierr > 0) {
            *info = n + 1;
            return 0.0;
        }

        f64 result = dwork[is];
        dwork[0] = dwork[iwrk] + iwrk;
        zwork[0] = cone;
        return result;
    }

    i32 icb = n * n;
    i32 icc = icb + n * m;
    i32 icd = icc + p * n;
    i32 icwk = icd + p * m;

    f64 upd = withd ? 1.0 : 0.0;

    f64 lambdr, lambdi;
    if (discr) {
        lambdr = cos(omega);
        lambdi = sin(omega);

        if (fulle) {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i <= j; i++) {
                    f64 eij = e[i + j * lde];
                    f64 aij = a[i + j * lda];
                    zwork[i + j * n] = (lambdr * eij - aij) + I * (lambdi * eij);
                }
                if (j < n - 1) {
                    zwork[(j + 1) + j * n] = -a[(j + 1) + j * lda] + I * 0.0;
                }
            }
        } else {
            for (i32 j = 0; j < n; j++) {
                i32 imax = (j + 1 < n - 1) ? (j + 1) : (n - 1);
                for (i32 i = 0; i <= imax; i++) {
                    zwork[i + j * n] = -a[i + j * lda] + I * 0.0;
                }
                zwork[j + j * n] = (lambdr - a[j + j * lda]) + I * lambdi;
            }
        }
    } else {
        if (fulle) {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i <= j; i++) {
                    f64 eij = e[i + j * lde];
                    f64 aij = a[i + j * lda];
                    zwork[i + j * n] = -aij + I * (omega * eij);
                }
                if (j < n - 1) {
                    zwork[(j + 1) + j * n] = -a[(j + 1) + j * lda] + I * 0.0;
                }
            }
        } else {
            for (i32 j = 0; j < n; j++) {
                i32 imax = (j + 1 < n - 1) ? (j + 1) : (n - 1);
                for (i32 i = 0; i <= imax; i++) {
                    zwork[i + j * n] = -a[i + j * lda] + I * 0.0;
                }
                zwork[j + j * n] = -a[j + j * lda] + I * omega;
            }
        }
    }

    SLC_ZLACP2("F", &n, &m, b, &ldb, &zwork[icb], &n);
    SLC_ZLACP2("F", &p, &n, c, &ldc, &zwork[icc], &p);
    if (withd) {
        SLC_ZLACP2("F", &p, &m, d, &ldd, &zwork[icd], &p);
    }

    ierr = slicot_mb02sz(n, zwork, n, iwork);
    if (ierr > 0) {
        *info = ierr;
        dwork[0] = 1.0;
        zwork[0] = (c128)(icwk);
        return 0.0;
    }

    ierr = slicot_mb02rz('N', n, m, zwork, n, iwork, &zwork[icb], n);

    c128 z_one = 1.0 + 0.0 * I;
    c128 z_upd = upd + 0.0 * I;
    SLC_ZGEMM("N", "N", &p, &m, &n, &z_one, &zwork[icc], &p,
              &zwork[icb], &n, &z_upd, &zwork[icd], &p);

    SLC_ZGESVD("N", "N", &p, &m, &zwork[icd], &p,
               &dwork[is], zwork, &p, zwork, &m,
               &zwork[icwk], &lzwork, &dwork[iwrk], &ierr);

    if (ierr > 0) {
        *info = n + 1;
        return 0.0;
    }

    f64 result = dwork[is];

    dwork[0] = 6 * minpm;
    zwork[0] = zwork[icwk] + (c128)(icwk);

    return result;
}
