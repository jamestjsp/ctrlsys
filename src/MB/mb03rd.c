/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void mb03rd(
    const char* jobx,
    const char* sort,
    const i32 n,
    const f64 pmax,
    f64* a,
    const i32 lda,
    f64* x,
    const i32 ldx,
    i32* nblcks,
    i32* blsize,
    f64* wr,
    f64* wi,
    const f64 tol,
    f64* dwork,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    bool ljobx = (jobx[0] == 'U' || jobx[0] == 'u');
    bool lsorn = (sort[0] == 'N' || sort[0] == 'n');
    bool lsors = (sort[0] == 'S' || sort[0] == 's');
    bool lsort = (sort[0] == 'B' || sort[0] == 'b') || lsors;

    *info = 0;

    if (!ljobx && !(jobx[0] == 'N' || jobx[0] == 'n')) {
        *info = -1;
    } else if (!lsorn && !lsort && !(sort[0] == 'C' || sort[0] == 'c')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (pmax < ONE) {
        *info = -4;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -6;
    } else if ((ldx < 1) || (ljobx && ldx < n)) {
        *info = -8;
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("MB03RD", &neginfo);
        return;
    }

    *nblcks = 0;
    if (n == 0) {
        return;
    }

    f64 safemn = SLC_DLAMCH("Safe minimum");
    f64 sc = ONE / safemn;
    SLC_DLABAD(&safemn, &sc);
    safemn = safemn / SLC_DLAMCH("Precision");

    const char* jobv = jobx;
    char jobv_buf[2] = "V";
    if (ljobx) {
        jobv = jobv_buf;
    }

    i32 dummy_info;
    mb03qx(n, a, lda, wr, wi, &dummy_info);

    f64 thresh = 0.0;
    if (lsort) {
        thresh = fabs(tol);
        if (thresh == ZERO) {
            thresh = sqrt(sqrt(SLC_DLAMCH("Epsilon")));
        }

        if (tol <= ZERO) {
            f64 emax = ZERO;
            i32 l = 0;
            while (l < n) {
                if (wi[l] == ZERO) {
                    emax = fmax(emax, fabs(wr[l]));
                    l = l + 1;
                } else {
                    emax = fmax(emax, hypot(wr[l], wi[l]));
                    l = l + 2;
                }
            }
            thresh = thresh * emax;
        }
    }

    i32 l11 = 0;
    while (l11 < n) {
        *nblcks = *nblcks + 1;

        i32 da11;
        if (wi[l11] == ZERO) {
            da11 = 1;
        } else {
            da11 = 2;
        }

        if (lsort) {
            i32 l22 = l11 + da11;
            i32 k = l22;
            while (k < n) {
                f64 edif = hypot(wr[l11] - wr[k], wi[l11] - wi[k]);
                if (edif <= thresh) {
                    i32 ku = k + 1;
                    mb03rx(jobv, n, l22 + 1, &ku, a, lda, x, ldx, wr, wi, dwork);

                    if (wi[l22] == ZERO) {
                        da11 = da11 + 1;
                    } else {
                        da11 = da11 + 2;
                    }
                    l22 = l11 + da11;
                }
                if (wi[k] == ZERO) {
                    k = k + 1;
                } else {
                    k = k + 2;
                }
            }
        }

        i32 l22 = l11 + da11;
        i32 l22m1 = l22 - 1;

        while (l22 < n) {
            i32 da22 = n - l22;

            ma02ad("Full", da11, da22, &a[l11 + l22 * lda], lda,
                   &a[l22 + l11 * lda], lda);

            i32 ierr;
            mb03ry(da11, da22, pmax, &a[l11 + l11 * lda], lda,
                   &a[l22 + l22 * lda], lda, &a[l11 + l22 * lda], lda, &ierr);

            if (ierr == 0) {
                break;
            }

            ma02ad("Full", da22, da11, &a[l22 + l11 * lda], lda,
                   &a[l11 + l22 * lda], lda);

            for (i32 j = 0; j < da11; j++) {
                for (i32 i = 0; i < da22; i++) {
                    a[(l22 + i) + (l11 + j) * lda] = ZERO;
                }
            }

            i32 kpos;
            f64 d;

            if (lsorn || lsors) {
                f64 rav = ZERO;
                f64 cav = ZERO;

                for (i32 i = l11; i <= l22m1; i++) {
                    rav = rav + wr[i];
                    cav = cav + fabs(wi[i]);
                }

                rav = rav / da11;
                cav = cav / da11;

                d = hypot(rav - wr[l22], cav - wi[l22]);
                kpos = l22;

                i32 lpos;
                if (wi[l22] == ZERO) {
                    lpos = l22 + 1;
                } else {
                    lpos = l22 + 2;
                }

                while (lpos < n) {
                    f64 c = hypot(rav - wr[lpos], cav - wi[lpos]);
                    if (c < d) {
                        d = c;
                        kpos = lpos;
                    }
                    if (wi[lpos] == ZERO) {
                        lpos = lpos + 1;
                    } else {
                        lpos = lpos + 2;
                    }
                }
            } else {
                d = sc;
                i32 lpos = l22;
                kpos = l22;

                while (lpos < n) {
                    i32 ipos = l11;
                    while (ipos <= l22m1) {
                        f64 c = hypot(wr[ipos] - wr[lpos], wi[ipos] - wi[lpos]);
                        if (c < d) {
                            d = c;
                            kpos = lpos;
                        }
                        if (wi[ipos] == ZERO) {
                            ipos = ipos + 1;
                        } else {
                            ipos = ipos + 2;
                        }
                    }
                    if (wi[lpos] == ZERO) {
                        lpos = lpos + 1;
                    } else {
                        lpos = lpos + 2;
                    }
                }
            }

            i32 ku = kpos + 1;
            mb03rx(jobv, n, l22 + 1, &ku, a, lda, x, ldx, wr, wi, dwork);

            if (wi[l22] == ZERO) {
                da11 = da11 + 1;
            } else {
                da11 = da11 + 2;
            }
            l22 = l11 + da11;
            l22m1 = l22 - 1;
        }

        if (ljobx) {
            i32 da22 = n - l22;
            if (l22 < n) {
                f64 alpha = ONE;
                f64 beta = ONE;
                SLC_DGEMM("N", "N", &n, &da22, &da11, &alpha,
                          &x[l11 * ldx], &ldx, &a[l11 + l22 * lda], &lda,
                          &beta, &x[l22 * ldx], &ldx);
            }

            for (i32 j = l11; j <= l22m1; j++) {
                i32 len = n;
                i32 inc = 1;
                sc = SLC_DNRM2(&len, &x[j * ldx], &inc);
                if (sc > safemn) {
                    for (i32 ii = 0; ii < da11; ii++) {
                        a[j + (l11 + ii) * lda] *= sc;
                    }
                    f64 invsc = ONE / sc;
                    SLC_DSCAL(&len, &invsc, &x[j * ldx], &inc);
                    for (i32 ii = 0; ii < da11; ii++) {
                        a[(l11 + ii) + j * lda] *= invsc;
                    }
                }
            }
        }

        if (l22 < n) {
            i32 da22 = n - l22;
            for (i32 j = 0; j < da22; j++) {
                for (i32 i = 0; i < da11; i++) {
                    a[(l11 + i) + (l22 + j) * lda] = ZERO;
                }
            }
            for (i32 j = 0; j < da11; j++) {
                for (i32 i = 0; i < da22; i++) {
                    a[(l22 + i) + (l11 + j) * lda] = ZERO;
                }
            }
        }

        blsize[*nblcks - 1] = da11;
        l11 = l22;
    }
}
