/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * MB02XD - Solve A'*A*X = B using Cholesky factorization
 *
 * Solves a set of systems of linear equations A'*A*X = B, or in implicit
 * form f(A)*X = B, with A'*A or f(A) positive definite, using symmetric
 * Gaussian elimination (Cholesky factorization).
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>

void mb02xd(
    const char* form_str,
    const char* stor_str,
    const char* uplo_str,
    mb02xd_callback f,
    const i32* m_ptr,
    const i32* n_ptr,
    const i32* nrhs_ptr,
    const i32* ipar,
    const i32* lipar_ptr,
    const f64* dpar,
    const i32* ldpar_ptr,
    const f64* a,
    const i32* lda_ptr,
    f64* b,
    const i32* ldb_ptr,
    f64* ata,
    const i32* ldata_ptr,
    f64* dwork,
    const i32* ldwork_ptr,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    char form = toupper((unsigned char)form_str[0]);
    char stor = toupper((unsigned char)stor_str[0]);
    char uplo = toupper((unsigned char)uplo_str[0]);

    i32 m = *m_ptr;
    i32 n = *n_ptr;
    i32 nrhs = *nrhs_ptr;
    i32 lipar = *lipar_ptr;
    i32 ldpar = *ldpar_ptr;
    i32 lda = *lda_ptr;
    i32 ldb = *ldb_ptr;
    i32 ldata = *ldata_ptr;
    bool mat = (form == 'S');
    bool full = (stor == 'F');
    bool upper = (uplo == 'U');

    *info = 0;

    if (!mat && form != 'F') {
        *info = -1;
    } else if (!full && stor != 'P') {
        *info = -2;
    } else if (!upper && uplo != 'L') {
        *info = -3;
    } else if (m < 0) {
        *info = -5;
    } else if (n < 0) {
        *info = -6;
    } else if (nrhs < 0) {
        *info = -7;
    } else if (!mat && lipar < 0) {
        *info = -9;
    } else if (!mat && ldpar < 0) {
        *info = -11;
    } else if (lda < 1 || (mat && lda < m)) {
        *info = -13;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -15;
    } else if (ldata < 1 || (full && ldata < n)) {
        *info = -17;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 || (mat && m == 0)) {
        return;
    }

    i32 ierr = 0;

    if (mat) {
        if (full) {
            SLC_DSYRK(&uplo, "T", &n, &m, &one, a, &lda, &zero, ata, &ldata);
        } else if (upper) {
            i32 j1 = 0;
            for (i32 j = 0; j < n; j++) {
                i32 jp1 = j + 1;
                i32 inc = 1;
                SLC_DGEMV("T", &m, &jp1, &one, a, &lda, &a[j * lda], &inc,
                          &zero, &ata[j1], &inc);
                j1 += jp1;
            }
        } else {
            i32 j1 = 0;
            for (i32 j = 0; j < n; j++) {
                i32 len = n - j;
                i32 inc = 1;
                SLC_DGEMV("T", &m, &len, &one, &a[j * lda], &lda, &a[j * lda],
                          &inc, &zero, &ata[j1], &inc);
                j1 += len;
            }
        }
    } else {
        f(stor_str, uplo_str, n_ptr, ipar, lipar_ptr, dpar, ldpar_ptr,
          a, lda_ptr, ata, ldata_ptr, dwork, ldwork_ptr, &ierr);
        if (ierr != 0) {
            *info = n + ierr;
            return;
        }
    }

    if (full) {
        SLC_DPOTRF(&uplo, &n, ata, &ldata, &ierr);
    } else {
        SLC_DPPTRF(&uplo, &n, ata, &ierr);
    }

    if (ierr != 0) {
        *info = ierr;
        return;
    }

    if (full) {
        SLC_DPOTRS(&uplo, &n, &nrhs, ata, &ldata, b, &ldb, &ierr);
    } else {
        SLC_DPPTRS(&uplo, &n, &nrhs, ata, b, &ldb, &ierr);
    }
}
