/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 *
 * MB04RW - Blocked complex generalized Sylvester equation solver
 *
 * Solves using Level 3 BLAS:
 *     A * R - L * B = scale * C
 *     D * R - L * E = scale * F
 *
 * where A, B, D, E are complex upper triangular (generalized Schur form).
 * This is the blocked version that calls MB04RV for subproblems.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <complex.h>

void mb04rw(i32 m, i32 n, f64 pmax,
            const c128 *a, i32 lda, const c128 *b, i32 ldb,
            c128 *c, i32 ldc, const c128 *d, i32 ldd,
            const c128 *e, i32 lde, c128 *f, i32 ldf,
            f64 *scale, i32 *iwork, i32 *info) {

    const f64 ONE = 1.0;
    const c128 CONE = 1.0 + 0.0*I;
    const c128 NEG_CONE = -1.0 + 0.0*I;
    const i32 int0 = 0;

    *info = 0;
    *scale = ONE;

    if (m == 0 || n == 0) {
        return;
    }

    i32 mb = SLC_ILAENV(&(i32){2}, "ZTGSYL", "NoTran", &m, &n, &(i32){-1}, &(i32){-1});
    i32 nb = SLC_ILAENV(&(i32){5}, "ZTGSYL", "NoTran", &m, &n, &(i32){-1}, &(i32){-1});

    if ((mb <= 1 && nb <= 1) || (mb >= m && nb >= n)) {
        mb04rv(m, n, pmax, a, lda, b, ldb, c, ldc, d, ldd, e, lde, f, ldf, scale, info);
        return;
    }

    i32 p = 0;
    i32 i = 1;  // Fortran 1-based index

    while (i <= m) {
        p++;
        iwork[p - 1] = i;  // Convert to 0-based storage
        i += mb;
        if (i >= m) break;
    }
    iwork[p] = m + 1;  // Sentinel (still 1-based value)
    if (iwork[p - 1] == iwork[p]) {
        p--;
    }

    i32 q = p + 1;
    i32 j = 1;  // Fortran 1-based index

    while (j <= n) {
        q++;
        iwork[q - 1] = j;  // Convert to 0-based storage
        j += nb;
        if (j >= n) break;
    }
    iwork[q] = n + 1;  // Sentinel (still 1-based value)
    if (iwork[q - 1] == iwork[q]) {
        q--;
    }

    f64 scaloc;
    i32 k;

    // Solve the (I, J)-subsystem
    // For J = P+2, ..., Q (in Fortran 1-based)
    // For I = P, P-1, ..., 1 (in Fortran 1-based)
    for (i32 jj = p + 2; jj <= q; jj++) {  // jj is Fortran 1-based block index
        i32 js = iwork[jj - 1];  // Start column (1-based)
        i32 je = iwork[jj] - 1;  // End column (1-based)
        i32 nb_blk = je - js + 1;

        for (i32 ii = p; ii >= 1; ii--) {  // ii is Fortran 1-based block index
            i32 is = iwork[ii - 1];  // Start row (1-based)
            i32 ie = iwork[ii] - 1;  // End row (1-based)
            i32 mb_blk = ie - is + 1;

            // Convert to 0-based indices for C array access
            i32 is_c = is - 1;
            i32 js_c = js - 1;

            mb04rv(mb_blk, nb_blk, pmax,
                   &a[is_c + is_c*lda], lda, &b[js_c + js_c*ldb], ldb,
                   &c[is_c + js_c*ldc], ldc, &d[is_c + is_c*ldd], ldd,
                   &e[js_c + js_c*lde], lde, &f[is_c + js_c*ldf], ldf,
                   &scaloc, info);

            if (*info > 0) {
                return;
            }

            if (scaloc != ONE) {
                // Scale C(:, 1:JS-1) and F(:, 1:JS-1)
                i32 cols_before = js - 1;
                if (cols_before > 0) {
                    SLC_ZLASCL("G", &int0, &int0, &ONE, &scaloc, &m, &cols_before,
                               c, &ldc, &k);
                    SLC_ZLASCL("G", &int0, &int0, &ONE, &scaloc, &m, &cols_before,
                               f, &ldf, &k);
                }

                // Scale C(1:IS-1, JS:JE) and F(1:IS-1, JS:JE)
                i32 rows_before = is - 1;
                if (rows_before > 0) {
                    SLC_ZLASCL("G", &int0, &int0, &ONE, &scaloc, &rows_before, &nb_blk,
                               &c[0 + js_c*ldc], &ldc, &k);
                    SLC_ZLASCL("G", &int0, &int0, &ONE, &scaloc, &rows_before, &nb_blk,
                               &f[0 + js_c*ldf], &ldf, &k);
                }

                // Scale C(IE+1:M, JS:JE) and F(IE+1:M, JS:JE)
                i32 rows_after = m - ie;
                if (rows_after > 0) {
                    SLC_ZLASCL("G", &int0, &int0, &ONE, &scaloc, &rows_after, &nb_blk,
                               &c[ie + js_c*ldc], &ldc, &k);
                    SLC_ZLASCL("G", &int0, &int0, &ONE, &scaloc, &rows_after, &nb_blk,
                               &f[ie + js_c*ldf], &ldf, &k);
                }

                // Scale C(:, JE+1:N) and F(:, JE+1:N)
                i32 cols_after = n - je;
                if (cols_after > 0) {
                    SLC_ZLASCL("G", &int0, &int0, &ONE, &scaloc, &m, &cols_after,
                               &c[0 + je*ldc], &ldc, &k);
                    SLC_ZLASCL("G", &int0, &int0, &ONE, &scaloc, &m, &cols_after,
                               &f[0 + je*ldf], &ldf, &k);
                }

                *scale = (*scale) * scaloc;
            }

            // Substitute R(I, J) and L(I, J) into the remaining equation
            if (ii > 1) {
                // C(1:IS-1, JS:JE) -= A(1:IS-1, IS:IE) * C(IS:IE, JS:JE)
                // F(1:IS-1, JS:JE) -= D(1:IS-1, IS:IE) * C(IS:IE, JS:JE)
                i32 rows_before = is - 1;
                SLC_ZGEMM("N", "N", &rows_before, &nb_blk, &mb_blk, &NEG_CONE,
                          &a[0 + is_c*lda], &lda, &c[is_c + js_c*ldc], &ldc,
                          &CONE, &c[0 + js_c*ldc], &ldc);
                SLC_ZGEMM("N", "N", &rows_before, &nb_blk, &mb_blk, &NEG_CONE,
                          &d[0 + is_c*ldd], &ldd, &c[is_c + js_c*ldc], &ldc,
                          &CONE, &f[0 + js_c*ldf], &ldf);
            }

            if (jj < q) {
                // C(IS:IE, JE+1:N) += F(IS:IE, JS:JE) * B(JS:JE, JE+1:N)
                // F(IS:IE, JE+1:N) += F(IS:IE, JS:JE) * E(JS:JE, JE+1:N)
                i32 cols_after = n - je;
                SLC_ZGEMM("N", "N", &mb_blk, &cols_after, &nb_blk, &CONE,
                          &f[is_c + js_c*ldf], &ldf, &b[js_c + je*ldb], &ldb,
                          &CONE, &c[is_c + je*ldc], &ldc);
                SLC_ZGEMM("N", "N", &mb_blk, &cols_after, &nb_blk, &CONE,
                          &f[is_c + js_c*ldf], &ldf, &e[js_c + je*lde], &lde,
                          &CONE, &f[is_c + je*ldf], &ldf);
            }
        }
    }
}
