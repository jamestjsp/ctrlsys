#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>

void mb01vd(const char *trana, const char *tranb, i32 ma, i32 na, i32 mb, i32 nb,
            f64 alpha, f64 beta, const f64 *a, i32 lda, const f64 *b, i32 ldb,
            f64 *c, i32 ldc, i32 *mc, i32 *nc, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    bool transa = (trana[0] == 'T' || trana[0] == 't' || trana[0] == 'C' || trana[0] == 'c');
    bool transb = (tranb[0] == 'T' || tranb[0] == 't' || tranb[0] == 'C' || tranb[0] == 'c');

    *mc = ma * mb;
    *info = 0;

    if (!(transa || trana[0] == 'N' || trana[0] == 'n')) {
        *info = -1;
    } else if (!(transb || tranb[0] == 'N' || tranb[0] == 'n')) {
        *info = -2;
    } else if (ma < 0) {
        *info = -3;
    } else if (na < 0) {
        *info = -4;
    } else if (mb < 0) {
        *info = -5;
    } else if (nb < 0) {
        *info = -6;
    } else if ((transa && lda < na) || lda < 1 || (!transa && lda < ma)) {
        *info = -10;
    } else if ((transb && ldb < nb) || ldb < 1 || (!transb && ldb < mb)) {
        *info = -12;
    } else if (ldc < 1 || (*mc > 0 && ldc < *mc)) {
        *info = -14;
    }

    if (*info != 0) {
        return;
    }

    *nc = na * nb;
    if (*mc == 0 || *nc == 0) {
        return;
    }

    if (alpha == ZERO) {
        if (beta == ZERO) {
            SLC_DLASET("Full", mc, nc, &ZERO, &ZERO, c, &ldc);
        } else if (beta != ONE) {
            for (i32 j = 0; j < *nc; j++) {
                SLC_DSCAL(mc, &beta, &c[j * ldc], &(i32){1});
            }
        }
        return;
    }

    i32 jc = 0;

    if (!transa && !transb) {
        for (i32 j = 0; j < na; j++) {
            for (i32 k = 0; k < nb; k++) {
                i32 ic = 0;
                for (i32 i = 0; i < ma; i++) {
                    f64 aij = alpha * a[i + j * lda];
                    for (i32 l = 0; l < mb; l++) {
                        if (beta == ZERO) {
                            c[ic + l + jc * ldc] = aij * b[l + k * ldb];
                        } else {
                            c[ic + l + jc * ldc] = beta * c[ic + l + jc * ldc] + aij * b[l + k * ldb];
                        }
                    }
                    ic += mb;
                }
                jc++;
            }
        }
    } else if (transa && !transb) {
        for (i32 j = 0; j < na; j++) {
            for (i32 k = 0; k < nb; k++) {
                i32 ic = 0;
                for (i32 i = 0; i < ma; i++) {
                    f64 aij = alpha * a[j + i * lda];
                    for (i32 l = 0; l < mb; l++) {
                        if (beta == ZERO) {
                            c[ic + l + jc * ldc] = aij * b[l + k * ldb];
                        } else {
                            c[ic + l + jc * ldc] = beta * c[ic + l + jc * ldc] + aij * b[l + k * ldb];
                        }
                    }
                    ic += mb;
                }
                jc++;
            }
        }
    } else if (!transa && transb) {
        for (i32 j = 0; j < na; j++) {
            for (i32 k = 0; k < nb; k++) {
                i32 ic = 0;
                for (i32 i = 0; i < ma; i++) {
                    f64 aij = alpha * a[i + j * lda];
                    for (i32 l = 0; l < mb; l++) {
                        if (beta == ZERO) {
                            c[ic + l + jc * ldc] = aij * b[k + l * ldb];
                        } else {
                            c[ic + l + jc * ldc] = beta * c[ic + l + jc * ldc] + aij * b[k + l * ldb];
                        }
                    }
                    ic += mb;
                }
                jc++;
            }
        }
    } else {
        for (i32 j = 0; j < na; j++) {
            for (i32 k = 0; k < nb; k++) {
                i32 ic = 0;
                for (i32 i = 0; i < ma; i++) {
                    f64 aij = alpha * a[j + i * lda];
                    for (i32 l = 0; l < mb; l++) {
                        if (beta == ZERO) {
                            c[ic + l + jc * ldc] = aij * b[k + l * ldb];
                        } else {
                            c[ic + l + jc * ldc] = beta * c[ic + l + jc * ldc] + aij * b[k + l * ldb];
                        }
                    }
                    ic += mb;
                }
                jc++;
            }
        }
    }
}
