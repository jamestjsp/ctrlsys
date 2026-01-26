/**
 * @file mb04rs.c
 * @brief Solve generalized real Sylvester equation with matrices in generalized Schur form.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

#define LDZ 8

void mb04rs(
    const i32 m,
    const i32 n,
    const f64 pmax,
    const f64* a, const i32 lda,
    const f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    const f64* d, const i32 ldd,
    const f64* e, const i32 lde,
    f64* f, const i32 ldf,
    f64* scale,
    i32* iwork,
    i32* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    i32 int1 = 1;
    f64 dbl0 = 0.0, dbl1 = 1.0, dbl_neg1 = -1.0;

    i32 i, ie, ierr, ii, is, isp1, ix, j, je, jep1, jj, js, jsp1;
    i32 k, l, mb, nb, p, q, zdim;
    f64 alpha, sc, scaloc;

    i32 ipiv[LDZ], jpiv[LDZ];
    f64 rhs[LDZ], z[LDZ * LDZ], zs[LDZ * LDZ];

    *info = 0;

    p = 0;
    i = 0;

    while (i < m) {
        iwork[p] = i;
        p++;
        if (i == m - 1) {
            break;
        }
        if (a[(i + 1) + i * lda] != ZERO) {
            i += 2;
        } else {
            i += 1;
        }
    }

    iwork[p] = m;

    q = p;
    j = 0;

    while (j < n) {
        q++;
        iwork[q] = j;
        if (j == n - 1) {
            break;
        }
        if (b[(j + 1) + j * ldb] != ZERO) {
            j += 2;
        } else {
            j += 1;
        }
    }

    iwork[q + 1] = n;

    *scale = ONE;
    scaloc = ONE;

    for (j = p + 1; j <= q; j++) {
        js = iwork[j];
        jsp1 = js + 1;
        jep1 = iwork[j + 1];
        je = jep1 - 1;
        nb = jep1 - js;

        for (i = p - 1; i >= 0; i--) {
            is = iwork[i];
            isp1 = is + 1;
            ie = iwork[i + 1] - 1;
            mb = ie - is + 1;
            zdim = mb * nb * 2;

            if ((mb == 1) && (nb == 1)) {
                z[0] = a[is + is * lda];
                z[1] = d[is + is * ldd];
                z[LDZ] = -b[js + js * ldb];
                z[LDZ + 1] = -e[js + js * lde];

                SLC_DLACPY("F", &zdim, &zdim, z, &(i32){LDZ}, zs, &(i32){LDZ});

                rhs[0] = c[is + js * ldc];
                rhs[1] = f[is + js * ldf];

                SLC_DGETC2(&zdim, z, &(i32){LDZ}, ipiv, jpiv, &ierr);
                if (ierr > 0) {
                    *info = 2;

                    for (l = 0; l < zdim; l++) {
                        ix = SLC_IDAMAX(&zdim, &zs[l], &(i32){LDZ}) - 1;
                        sc = fabs(zs[l + ix * LDZ]);
                        if (sc == ZERO) {
                            *info = 1;
                            return;
                        } else if (sc != ONE) {
                            f64 inv_sc = ONE / sc;
                            SLC_DSCAL(&zdim, &inv_sc, &zs[l], &(i32){LDZ});
                            rhs[l] = rhs[l] / sc;
                        }
                    }

                    SLC_DGETC2(&zdim, zs, &(i32){LDZ}, ipiv, jpiv, &ierr);
                    if (ierr == 0)
                        *info = 0;
                    if (*info > 0)
                        return;
                    SLC_DLACPY("F", &zdim, &zdim, zs, &(i32){LDZ}, z, &(i32){LDZ});
                }

                SLC_DGESC2(&zdim, z, &(i32){LDZ}, rhs, ipiv, jpiv, &scaloc);

                *scale *= scaloc;
                f64 max_rhs = fabs(rhs[0]);
                if (fabs(rhs[1]) > max_rhs) max_rhs = fabs(rhs[1]);
                if (max_rhs * (*scale) > pmax) {
                    *info = 1;
                    return;
                }

                if (scaloc != ONE) {
                    for (k = 0; k < n; k++) {
                        SLC_DSCAL(&m, &scaloc, &c[k * ldc], &int1);
                        SLC_DSCAL(&m, &scaloc, &f[k * ldf], &int1);
                    }
                }

                c[is + js * ldc] = rhs[0];
                f[is + js * ldf] = rhs[1];

                if (i > 0) {
                    alpha = -rhs[0];
                    i32 len = is;
                    SLC_DAXPY(&len, &alpha, &a[is * lda], &int1, &c[js * ldc], &int1);
                    SLC_DAXPY(&len, &alpha, &d[is * ldd], &int1, &f[js * ldf], &int1);
                }
                if (j < q) {
                    i32 len = n - je - 1;
                    SLC_DAXPY(&len, &rhs[1], &b[js + jep1 * ldb], &ldb, &c[is + jep1 * ldc], &ldc);
                    SLC_DAXPY(&len, &rhs[1], &e[js + jep1 * lde], &lde, &f[is + jep1 * ldf], &ldf);
                }

            } else if ((mb == 1) && (nb == 2)) {
                z[0] = a[is + is * lda];
                z[1] = ZERO;
                z[2] = d[is + is * ldd];
                z[3] = ZERO;

                z[LDZ + 0] = ZERO;
                z[LDZ + 1] = a[is + is * lda];
                z[LDZ + 2] = ZERO;
                z[LDZ + 3] = d[is + is * ldd];

                z[2 * LDZ + 0] = -b[js + js * ldb];
                z[2 * LDZ + 1] = -b[js + jsp1 * ldb];
                z[2 * LDZ + 2] = -e[js + js * lde];
                z[2 * LDZ + 3] = -e[js + jsp1 * lde];

                z[3 * LDZ + 0] = -b[jsp1 + js * ldb];
                z[3 * LDZ + 1] = -b[jsp1 + jsp1 * ldb];
                z[3 * LDZ + 2] = ZERO;
                z[3 * LDZ + 3] = -e[jsp1 + jsp1 * lde];

                SLC_DLACPY("F", &zdim, &zdim, z, &(i32){LDZ}, zs, &(i32){LDZ});

                rhs[0] = c[is + js * ldc];
                rhs[1] = c[is + jsp1 * ldc];
                rhs[2] = f[is + js * ldf];
                rhs[3] = f[is + jsp1 * ldf];

                SLC_DGETC2(&zdim, z, &(i32){LDZ}, ipiv, jpiv, &ierr);
                if (ierr > 0) {
                    *info = 2;

                    for (l = 0; l < zdim; l++) {
                        ix = SLC_IDAMAX(&zdim, &zs[l], &(i32){LDZ}) - 1;
                        sc = fabs(zs[l + ix * LDZ]);
                        if (sc == ZERO) {
                            *info = 1;
                            return;
                        } else if (sc != ONE) {
                            f64 inv_sc = ONE / sc;
                            SLC_DSCAL(&zdim, &inv_sc, &zs[l], &(i32){LDZ});
                            rhs[l] = rhs[l] / sc;
                        }
                    }

                    SLC_DGETC2(&zdim, zs, &(i32){LDZ}, ipiv, jpiv, &ierr);
                    if (ierr == 0)
                        *info = 0;
                    if (*info > 0)
                        return;
                    SLC_DLACPY("F", &zdim, &zdim, zs, &(i32){LDZ}, z, &(i32){LDZ});
                }

                SLC_DGESC2(&zdim, z, &(i32){LDZ}, rhs, ipiv, jpiv, &scaloc);

                *scale *= scaloc;
                f64 max_rhs = fabs(rhs[0]);
                for (l = 1; l < 4; l++) {
                    if (fabs(rhs[l]) > max_rhs) max_rhs = fabs(rhs[l]);
                }
                if (max_rhs * (*scale) > pmax) {
                    *info = 1;
                    return;
                }

                if (scaloc != ONE) {
                    for (k = 0; k < n; k++) {
                        SLC_DSCAL(&m, &scaloc, &c[k * ldc], &int1);
                        SLC_DSCAL(&m, &scaloc, &f[k * ldf], &int1);
                    }
                }

                c[is + js * ldc] = rhs[0];
                c[is + jsp1 * ldc] = rhs[1];
                f[is + js * ldf] = rhs[2];
                f[is + jsp1 * ldf] = rhs[3];

                if (i > 0) {
                    i32 len = is;
                    SLC_DGER(&len, &nb, &dbl_neg1, &a[is * lda], &int1, &rhs[0], &int1, &c[js * ldc], &ldc);
                    SLC_DGER(&len, &nb, &dbl_neg1, &d[is * ldd], &int1, &rhs[0], &int1, &f[js * ldf], &ldf);
                }
                if (j < q) {
                    i32 len = n - je - 1;
                    SLC_DAXPY(&len, &rhs[2], &b[js + jep1 * ldb], &ldb, &c[is + jep1 * ldc], &ldc);
                    SLC_DAXPY(&len, &rhs[2], &e[js + jep1 * lde], &lde, &f[is + jep1 * ldf], &ldf);
                    SLC_DAXPY(&len, &rhs[3], &b[jsp1 + jep1 * ldb], &ldb, &c[is + jep1 * ldc], &ldc);
                    SLC_DAXPY(&len, &rhs[3], &e[jsp1 + jep1 * lde], &lde, &f[is + jep1 * ldf], &ldf);
                }

            } else if ((mb == 2) && (nb == 1)) {
                z[0] = a[is + is * lda];
                z[1] = a[isp1 + is * lda];
                z[2] = d[is + is * ldd];
                z[3] = ZERO;

                z[LDZ + 0] = a[is + isp1 * lda];
                z[LDZ + 1] = a[isp1 + isp1 * lda];
                z[LDZ + 2] = d[is + isp1 * ldd];
                z[LDZ + 3] = d[isp1 + isp1 * ldd];

                z[2 * LDZ + 0] = -b[js + js * ldb];
                z[2 * LDZ + 1] = ZERO;
                z[2 * LDZ + 2] = -e[js + js * lde];
                z[2 * LDZ + 3] = ZERO;

                z[3 * LDZ + 0] = ZERO;
                z[3 * LDZ + 1] = -b[js + js * ldb];
                z[3 * LDZ + 2] = ZERO;
                z[3 * LDZ + 3] = -e[js + js * lde];

                SLC_DLACPY("F", &zdim, &zdim, z, &(i32){LDZ}, zs, &(i32){LDZ});

                rhs[0] = c[is + js * ldc];
                rhs[1] = c[isp1 + js * ldc];
                rhs[2] = f[is + js * ldf];
                rhs[3] = f[isp1 + js * ldf];

                SLC_DGETC2(&zdim, z, &(i32){LDZ}, ipiv, jpiv, &ierr);
                if (ierr > 0) {
                    *info = 2;

                    for (l = 0; l < zdim; l++) {
                        ix = SLC_IDAMAX(&zdim, &zs[l], &(i32){LDZ}) - 1;
                        sc = fabs(zs[l + ix * LDZ]);
                        if (sc == ZERO) {
                            *info = 1;
                            return;
                        } else if (sc != ONE) {
                            f64 inv_sc = ONE / sc;
                            SLC_DSCAL(&zdim, &inv_sc, &zs[l], &(i32){LDZ});
                            rhs[l] = rhs[l] / sc;
                        }
                    }

                    SLC_DGETC2(&zdim, zs, &(i32){LDZ}, ipiv, jpiv, &ierr);
                    if (ierr == 0)
                        *info = 0;
                    if (*info > 0)
                        return;
                    SLC_DLACPY("F", &zdim, &zdim, zs, &(i32){LDZ}, z, &(i32){LDZ});
                }

                SLC_DGESC2(&zdim, z, &(i32){LDZ}, rhs, ipiv, jpiv, &scaloc);

                *scale *= scaloc;
                f64 max_rhs = fabs(rhs[0]);
                for (l = 1; l < 4; l++) {
                    if (fabs(rhs[l]) > max_rhs) max_rhs = fabs(rhs[l]);
                }
                if (max_rhs * (*scale) > pmax) {
                    *info = 1;
                    return;
                }

                if (scaloc != ONE) {
                    for (k = 0; k < n; k++) {
                        SLC_DSCAL(&m, &scaloc, &c[k * ldc], &int1);
                        SLC_DSCAL(&m, &scaloc, &f[k * ldf], &int1);
                    }
                }

                c[is + js * ldc] = rhs[0];
                c[isp1 + js * ldc] = rhs[1];
                f[is + js * ldf] = rhs[2];
                f[isp1 + js * ldf] = rhs[3];

                if (i > 0) {
                    i32 len = is;
                    SLC_DGEMV("N", &len, &mb, &dbl_neg1, &a[is * lda], &lda, &rhs[0], &int1, &dbl1, &c[js * ldc], &int1);
                    SLC_DGEMV("N", &len, &mb, &dbl_neg1, &d[is * ldd], &ldd, &rhs[0], &int1, &dbl1, &f[js * ldf], &int1);
                }
                if (j < q) {
                    i32 len = n - je - 1;
                    SLC_DGER(&mb, &len, &dbl1, &rhs[2], &int1, &b[js + jep1 * ldb], &ldb, &c[is + jep1 * ldc], &ldc);
                    SLC_DGER(&mb, &len, &dbl1, &rhs[2], &int1, &e[js + jep1 * lde], &lde, &f[is + jep1 * ldf], &ldf);
                }

            } else if ((mb == 2) && (nb == 2)) {
                SLC_DLASET("F", &(i32){LDZ}, &(i32){LDZ}, &dbl0, &dbl0, z, &(i32){LDZ});

                z[0] = a[is + is * lda];
                z[1] = a[isp1 + is * lda];
                z[4] = d[is + is * ldd];

                z[LDZ + 0] = a[is + isp1 * lda];
                z[LDZ + 1] = a[isp1 + isp1 * lda];
                z[LDZ + 4] = d[is + isp1 * ldd];
                z[LDZ + 5] = d[isp1 + isp1 * ldd];

                z[2 * LDZ + 2] = a[is + is * lda];
                z[2 * LDZ + 3] = a[isp1 + is * lda];
                z[2 * LDZ + 6] = d[is + is * ldd];

                z[3 * LDZ + 2] = a[is + isp1 * lda];
                z[3 * LDZ + 3] = a[isp1 + isp1 * lda];
                z[3 * LDZ + 6] = d[is + isp1 * ldd];
                z[3 * LDZ + 7] = d[isp1 + isp1 * ldd];

                z[4 * LDZ + 0] = -b[js + js * ldb];
                z[4 * LDZ + 2] = -b[js + jsp1 * ldb];
                z[4 * LDZ + 4] = -e[js + js * lde];
                z[4 * LDZ + 6] = -e[js + jsp1 * lde];

                z[5 * LDZ + 1] = -b[js + js * ldb];
                z[5 * LDZ + 3] = -b[js + jsp1 * ldb];
                z[5 * LDZ + 5] = -e[js + js * lde];
                z[5 * LDZ + 7] = -e[js + jsp1 * lde];

                z[6 * LDZ + 0] = -b[jsp1 + js * ldb];
                z[6 * LDZ + 2] = -b[jsp1 + jsp1 * ldb];
                z[6 * LDZ + 6] = -e[jsp1 + jsp1 * lde];

                z[7 * LDZ + 1] = -b[jsp1 + js * ldb];
                z[7 * LDZ + 3] = -b[jsp1 + jsp1 * ldb];
                z[7 * LDZ + 7] = -e[jsp1 + jsp1 * lde];

                SLC_DLACPY("F", &zdim, &zdim, z, &(i32){LDZ}, zs, &(i32){LDZ});

                k = 0;
                ii = mb * nb;
                for (jj = js; jj < js + nb; jj++) {
                    SLC_DCOPY(&mb, &c[is + jj * ldc], &int1, &rhs[k], &int1);
                    SLC_DCOPY(&mb, &f[is + jj * ldf], &int1, &rhs[ii], &int1);
                    k += mb;
                    ii += mb;
                }

                SLC_DGETC2(&zdim, z, &(i32){LDZ}, ipiv, jpiv, &ierr);
                if (ierr > 0) {
                    *info = 2;

                    for (l = 0; l < zdim; l++) {
                        ix = SLC_IDAMAX(&zdim, &zs[l], &(i32){LDZ}) - 1;
                        sc = fabs(zs[l + ix * LDZ]);
                        if (sc == ZERO) {
                            *info = 1;
                            return;
                        } else if (sc != ONE) {
                            f64 inv_sc = ONE / sc;
                            SLC_DSCAL(&zdim, &inv_sc, &zs[l], &(i32){LDZ});
                            rhs[l] = rhs[l] / sc;
                        }
                    }

                    SLC_DGETC2(&zdim, zs, &(i32){LDZ}, ipiv, jpiv, &ierr);
                    if (ierr == 0)
                        *info = 0;
                    if (*info > 0)
                        return;
                    SLC_DLACPY("F", &zdim, &zdim, zs, &(i32){LDZ}, z, &(i32){LDZ});
                }

                SLC_DGESC2(&zdim, z, &(i32){LDZ}, rhs, ipiv, jpiv, &scaloc);

                *scale *= scaloc;
                f64 max_rhs = fabs(rhs[0]);
                for (l = 1; l < 8; l++) {
                    if (fabs(rhs[l]) > max_rhs) max_rhs = fabs(rhs[l]);
                }
                if (max_rhs * (*scale) > pmax) {
                    *info = 1;
                    return;
                }

                if (scaloc != ONE) {
                    for (k = 0; k < n; k++) {
                        SLC_DSCAL(&m, &scaloc, &c[k * ldc], &int1);
                        SLC_DSCAL(&m, &scaloc, &f[k * ldf], &int1);
                    }
                }

                k = 0;
                ii = mb * nb;
                for (jj = js; jj < js + nb; jj++) {
                    SLC_DCOPY(&mb, &rhs[k], &int1, &c[is + jj * ldc], &int1);
                    SLC_DCOPY(&mb, &rhs[ii], &int1, &f[is + jj * ldf], &int1);
                    k += mb;
                    ii += mb;
                }

                if (i > 0) {
                    i32 len = is;
                    SLC_DGEMM("N", "N", &len, &nb, &mb, &dbl_neg1, &a[is * lda], &lda, &rhs[0], &mb, &dbl1, &c[js * ldc], &ldc);
                    SLC_DGEMM("N", "N", &len, &nb, &mb, &dbl_neg1, &d[is * ldd], &ldd, &rhs[0], &mb, &dbl1, &f[js * ldf], &ldf);
                }
                if (j < q) {
                    i32 len = n - je - 1;
                    k = mb * nb;
                    SLC_DGEMM("N", "N", &mb, &len, &nb, &dbl1, &rhs[k], &mb, &b[js + jep1 * ldb], &ldb, &dbl1, &c[is + jep1 * ldc], &ldc);
                    SLC_DGEMM("N", "N", &mb, &len, &nb, &dbl1, &rhs[k], &mb, &e[js + jep1 * lde], &lde, &dbl1, &f[is + jep1 * ldf], &ldf);
                }
            }
        }
    }
}
