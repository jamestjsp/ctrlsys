// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

static int lfdum(const f64* x, const f64* y) {
    (void)x;
    (void)y;
    return 0;
}

void mb03za(const char* compc, const char* compu, const char* compv,
            const char* compw, const char* which, const i32* select,
            const i32 n, f64* a, const i32 lda, f64* b, const i32 ldb,
            f64* c, const i32 ldc, f64* u1, const i32 ldu1, f64* u2,
            const i32 ldu2, f64* v1, const i32 ldv1, f64* v2, const i32 ldv2,
            f64* w, const i32 ldw, f64* wr, f64* wi, i32* m, f64* dwork,
            const i32 ldwork, i32* info) {
    const i32 ldqz = 4;
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    bool wantc = (compc[0] == 'U' || compc[0] == 'u');
    bool wantu = (compu[0] == 'U' || compu[0] == 'u');
    bool wantv = (compv[0] == 'U' || compv[0] == 'u');
    bool initw = (compw[0] == 'I' || compw[0] == 'i');
    bool wantw = initw || (compw[0] == 'V' || compw[0] == 'v');
    bool cmpall = (which[0] == 'A' || which[0] == 'a');

    i32 wrkmin = (4*n > 1) ? 4*n : 1;

    *info = 0;

    if (!wantc && !(compc[0] == 'N' || compc[0] == 'n')) {
        *info = -1;
    } else if (!wantu && !(compu[0] == 'N' || compu[0] == 'n')) {
        *info = -2;
    } else if (!wantv && !(compv[0] == 'N' || compv[0] == 'n')) {
        *info = -3;
    } else if (!wantw && !(compw[0] == 'N' || compw[0] == 'n')) {
        *info = -4;
    } else if (!cmpall && !(which[0] == 'S' || which[0] == 's')) {
        *info = -5;
    } else {
        if (cmpall) {
            *m = n;
        } else {
            *m = 0;
            bool pair = false;
            for (i32 k = 0; k < n; k++) {
                if (pair) {
                    pair = false;
                } else {
                    if (k < n-1) {
                        if (a[(k+1) + k*lda] == ZERO) {
                            if (select[k])
                                (*m)++;
                        } else {
                            pair = true;
                            if (select[k] || select[k+1])
                                *m += 2;
                        }
                    } else {
                        if (select[n-1])
                            (*m)++;
                    }
                }
            }
        }

        i32 wrkmin_new = (8*(*m) > wrkmin) ? 8*(*m) : wrkmin;
        wrkmin = wrkmin_new;

        if (n < 0) {
            *info = -7;
        } else if (lda < (n > 1 ? n : 1)) {
            *info = -9;
        } else if (ldb < (n > 1 ? n : 1)) {
            *info = -11;
        } else if (ldc < 1 || (wantc && !cmpall && ldc < n)) {
            *info = -13;
        } else if (ldu1 < 1 || (wantu && !cmpall && ldu1 < n)) {
            *info = -15;
        } else if (ldu2 < 1 || (wantu && !cmpall && ldu2 < n)) {
            *info = -17;
        } else if (ldv1 < 1 || (wantv && !cmpall && ldv1 < n)) {
            *info = -19;
        } else if (ldv2 < 1 || (wantv && !cmpall && ldv2 < n)) {
            *info = -21;
        } else if (ldw < 1 || (wantw && ldw < 2*(*m))) {
            *info = -23;
        } else if (ldwork < wrkmin) {
            *info = -28;
            dwork[0] = (f64)wrkmin;
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        dwork[0] = ONE;
        return;
    }

    f64 q[ldqz * ldqz];
    f64 z[ldqz * ldqz];
    f64 t[ldqz * ldqz];
    f64 dw12[12];
    f64 wrnew[4], winew[4];
    i32 selnew[4];
    i32 idum[1];

    i32 int1 = 1;
    i32 int2 = 2;

    if (!cmpall) {
        i32 ks = 0;
        bool pair = false;

        for (i32 k = 0; k < n; k++) {
            if (pair) {
                pair = false;
            } else {
                bool swap = select[k] != 0;
                if (k < n-1) {
                    if (a[(k+1) + k*lda] != ZERO) {
                        pair = true;
                        swap = swap || (select[k+1] != 0);
                    }
                }

                i32 nbf;
                if (pair) {
                    nbf = 2;
                } else {
                    nbf = 1;
                }

                if (swap) {
                    ks++;
                    i32 ifst = k;
                    i32 ilst = ks - 1;
                    if (ilst > 0) {
                        if (a[ilst + (ilst-1)*lda] != ZERO) {
                            ilst--;
                        }
                    }

                    if (ilst == ifst) {
                        if (pair) ks++;
                        continue;
                    }

                    i32 here = ifst;
                    i32 ierr;

                    while (here > ilst) {
                        if (nbf == 1 || nbf == 2) {
                            i32 nbnext = 1;
                            if (here >= 2) {
                                if (a[(here-1) + (here-2)*lda] != ZERO)
                                    nbnext = 2;
                            }
                            i32 pos = here - nbnext;
                            i32 nb = nbnext + nbf;

                            for (i32 j = 0; j < nb; j++) {
                                for (i32 i = 0; i < nb; i++) {
                                    q[i + j*ldqz] = (i == j) ? ONE : ZERO;
                                    z[i + j*ldqz] = (i == j) ? ONE : ZERO;
                                }
                            }

                            mb03wa(true, true, nbnext, nbf,
                                   &a[pos + pos*lda], lda,
                                   &b[pos + pos*ldb], ldb,
                                   q, ldqz, z, ldqz, &ierr);

                            if (ierr != 0) {
                                dwork[0] = (f64)wrkmin;
                                *info = 1;
                                return;
                            }

                            if (pos > 0) {
                                SLC_DGEMM("N", "N", &pos, &nb, &nb, &ONE,
                                          &a[pos*lda], &lda, z, &ldqz,
                                          &ZERO, dwork, &n);
                                SLC_DLACPY("A", &pos, &nb, dwork, &n,
                                           &a[pos*lda], &lda);
                            }
                            if (pos + nb <= n - 1) {
                                i32 ncols = n - pos - nb;
                                SLC_DGEMM("T", "N", &nb, &ncols, &nb, &ONE,
                                          q, &ldqz, &a[pos + (pos+nb)*lda],
                                          &lda, &ZERO, dwork, &nb);
                                SLC_DLACPY("A", &nb, &ncols, dwork, &nb,
                                           &a[pos + (pos+nb)*lda], &lda);
                            }

                            if (pos > 0) {
                                SLC_DGEMM("N", "N", &pos, &nb, &nb, &ONE,
                                          &b[pos*ldb], &ldb, q, &ldqz,
                                          &ZERO, dwork, &n);
                                SLC_DLACPY("A", &pos, &nb, dwork, &n,
                                           &b[pos*ldb], &ldb);
                            }
                            if (pos + nb <= n - 1) {
                                i32 ncols = n - pos - nb;
                                SLC_DGEMM("T", "N", &nb, &ncols, &nb, &ONE,
                                          z, &ldqz, &b[pos + (pos+nb)*ldb],
                                          &ldb, &ZERO, dwork, &nb);
                                SLC_DLACPY("A", &nb, &ncols, dwork, &nb,
                                           &b[pos + (pos+nb)*ldb], &ldb);
                            }

                            if (wantc) {
                                SLC_DGEMM("N", "N", &n, &nb, &nb, &ONE,
                                          &c[pos*ldc], &ldc, q, &ldqz,
                                          &ZERO, dwork, &n);
                                SLC_DLACPY("A", &n, &nb, dwork, &n,
                                           &c[pos*ldc], &ldc);
                                SLC_DGEMM("T", "N", &nb, &n, &nb, &ONE,
                                          z, &ldqz, &c[pos], &ldc,
                                          &ZERO, dwork, &nb);
                                SLC_DLACPY("A", &nb, &n, dwork, &nb,
                                           &c[pos], &ldc);
                            }

                            if (wantu) {
                                SLC_DGEMM("N", "N", &n, &nb, &nb, &ONE,
                                          &u1[pos*ldu1], &ldu1, z, &ldqz,
                                          &ZERO, dwork, &n);
                                SLC_DLACPY("A", &n, &nb, dwork, &n,
                                           &u1[pos*ldu1], &ldu1);
                                SLC_DGEMM("N", "N", &n, &nb, &nb, &ONE,
                                          &u2[pos*ldu2], &ldu2, z, &ldqz,
                                          &ZERO, dwork, &n);
                                SLC_DLACPY("A", &n, &nb, dwork, &n,
                                           &u2[pos*ldu2], &ldu2);
                            }

                            if (wantv) {
                                SLC_DGEMM("N", "N", &n, &nb, &nb, &ONE,
                                          &v1[pos*ldv1], &ldv1, q, &ldqz,
                                          &ZERO, dwork, &n);
                                SLC_DLACPY("A", &n, &nb, dwork, &n,
                                           &v1[pos*ldv1], &ldv1);
                                SLC_DGEMM("N", "N", &n, &nb, &nb, &ONE,
                                          &v2[pos*ldv2], &ldv2, q, &ldqz,
                                          &ZERO, dwork, &n);
                                SLC_DLACPY("A", &n, &nb, dwork, &n,
                                           &v2[pos*ldv2], &ldv2);
                            }

                            here -= nbnext;

                            if (nbf == 2 && a[(here+1) + here*lda] == ZERO)
                                nbf = 3;

                        } else {
                            i32 nbnext = 1;
                            if (here >= 2) {
                                if (a[(here-1) + (here-2)*lda] != ZERO)
                                    nbnext = 2;
                            }
                            i32 pos = here - nbnext;
                            i32 nb = nbnext + 1;

                            for (i32 j = 0; j < nb; j++) {
                                for (i32 i = 0; i < nb; i++) {
                                    q[i + j*ldqz] = (i == j) ? ONE : ZERO;
                                    z[i + j*ldqz] = (i == j) ? ONE : ZERO;
                                }
                            }

                            mb03wa(true, true, nbnext, 1,
                                   &a[pos + pos*lda], lda,
                                   &b[pos + pos*ldb], ldb,
                                   q, ldqz, z, ldqz, &ierr);

                            if (ierr != 0) {
                                dwork[0] = (f64)wrkmin;
                                *info = 1;
                                return;
                            }

                            if (pos > 0) {
                                SLC_DGEMM("N", "N", &pos, &nb, &nb, &ONE,
                                          &a[pos*lda], &lda, z, &ldqz,
                                          &ZERO, dwork, &n);
                                SLC_DLACPY("A", &pos, &nb, dwork, &n,
                                           &a[pos*lda], &lda);
                            }
                            if (pos + nb <= n - 1) {
                                i32 ncols = n - pos - nb;
                                SLC_DGEMM("T", "N", &nb, &ncols, &nb, &ONE,
                                          q, &ldqz, &a[pos + (pos+nb)*lda],
                                          &lda, &ZERO, dwork, &nb);
                                SLC_DLACPY("A", &nb, &ncols, dwork, &nb,
                                           &a[pos + (pos+nb)*lda], &lda);
                            }

                            if (pos > 0) {
                                SLC_DGEMM("N", "N", &pos, &nb, &nb, &ONE,
                                          &b[pos*ldb], &ldb, q, &ldqz,
                                          &ZERO, dwork, &n);
                                SLC_DLACPY("A", &pos, &nb, dwork, &n,
                                           &b[pos*ldb], &ldb);
                            }
                            if (pos + nb <= n - 1) {
                                i32 ncols = n - pos - nb;
                                SLC_DGEMM("T", "N", &nb, &ncols, &nb, &ONE,
                                          z, &ldqz, &b[pos + (pos+nb)*ldb],
                                          &ldb, &ZERO, dwork, &nb);
                                SLC_DLACPY("A", &nb, &ncols, dwork, &nb,
                                           &b[pos + (pos+nb)*ldb], &ldb);
                            }

                            if (wantc) {
                                SLC_DGEMM("N", "N", &n, &nb, &nb, &ONE,
                                          &c[pos*ldc], &ldc, q, &ldqz,
                                          &ZERO, dwork, &n);
                                SLC_DLACPY("A", &n, &nb, dwork, &n,
                                           &c[pos*ldc], &ldc);
                                SLC_DGEMM("T", "N", &nb, &n, &nb, &ONE,
                                          z, &ldqz, &c[pos], &ldc,
                                          &ZERO, dwork, &nb);
                                SLC_DLACPY("A", &nb, &n, dwork, &nb,
                                           &c[pos], &ldc);
                            }

                            if (wantu) {
                                SLC_DGEMM("N", "N", &n, &nb, &nb, &ONE,
                                          &u1[pos*ldu1], &ldu1, z, &ldqz,
                                          &ZERO, dwork, &n);
                                SLC_DLACPY("A", &n, &nb, dwork, &n,
                                           &u1[pos*ldu1], &ldu1);
                                SLC_DGEMM("N", "N", &n, &nb, &nb, &ONE,
                                          &u2[pos*ldu2], &ldu2, z, &ldqz,
                                          &ZERO, dwork, &n);
                                SLC_DLACPY("A", &n, &nb, dwork, &n,
                                           &u2[pos*ldu2], &ldu2);
                            }

                            if (wantv) {
                                SLC_DGEMM("N", "N", &n, &nb, &nb, &ONE,
                                          &v1[pos*ldv1], &ldv1, q, &ldqz,
                                          &ZERO, dwork, &n);
                                SLC_DLACPY("A", &n, &nb, dwork, &n,
                                           &v1[pos*ldv1], &ldv1);
                                SLC_DGEMM("N", "N", &n, &nb, &nb, &ONE,
                                          &v2[pos*ldv2], &ldv2, q, &ldqz,
                                          &ZERO, dwork, &n);
                                SLC_DLACPY("A", &n, &nb, dwork, &n,
                                           &v2[pos*ldv2], &ldv2);
                            }

                            if (nbnext == 1) {
                                pos = here;
                                nb = nbnext + 1;

                                for (i32 j = 0; j < nb; j++) {
                                    for (i32 i = 0; i < nb; i++) {
                                        q[i + j*ldqz] = (i == j) ? ONE : ZERO;
                                        z[i + j*ldqz] = (i == j) ? ONE : ZERO;
                                    }
                                }

                                mb03wa(true, true, nbnext, 1,
                                       &a[pos + pos*lda], lda,
                                       &b[pos + pos*ldb], ldb,
                                       q, ldqz, z, ldqz, &ierr);

                                if (ierr != 0) {
                                    dwork[0] = (f64)wrkmin;
                                    *info = 1;
                                    return;
                                }

                                if (pos > 0) {
                                    SLC_DGEMM("N", "N", &pos, &nb, &nb, &ONE,
                                              &a[pos*lda], &lda, z, &ldqz,
                                              &ZERO, dwork, &n);
                                    SLC_DLACPY("A", &pos, &nb, dwork, &n,
                                               &a[pos*lda], &lda);
                                }
                                if (pos + nb <= n - 1) {
                                    i32 ncols = n - pos - nb;
                                    SLC_DGEMM("T", "N", &nb, &ncols, &nb, &ONE,
                                              q, &ldqz, &a[pos + (pos+nb)*lda],
                                              &lda, &ZERO, dwork, &nb);
                                    SLC_DLACPY("A", &nb, &ncols, dwork, &nb,
                                               &a[pos + (pos+nb)*lda], &lda);
                                }

                                if (pos > 0) {
                                    SLC_DGEMM("N", "N", &pos, &nb, &nb, &ONE,
                                              &b[pos*ldb], &ldb, q, &ldqz,
                                              &ZERO, dwork, &n);
                                    SLC_DLACPY("A", &pos, &nb, dwork, &n,
                                               &b[pos*ldb], &ldb);
                                }
                                if (pos + nb <= n - 1) {
                                    i32 ncols = n - pos - nb;
                                    SLC_DGEMM("T", "N", &nb, &ncols, &nb, &ONE,
                                              z, &ldqz, &b[pos + (pos+nb)*ldb],
                                              &ldb, &ZERO, dwork, &nb);
                                    SLC_DLACPY("A", &nb, &ncols, dwork, &nb,
                                               &b[pos + (pos+nb)*ldb], &ldb);
                                }

                                if (wantc) {
                                    SLC_DGEMM("N", "N", &n, &nb, &nb, &ONE,
                                              &c[pos*ldc], &ldc, q, &ldqz,
                                              &ZERO, dwork, &n);
                                    SLC_DLACPY("A", &n, &nb, dwork, &n,
                                               &c[pos*ldc], &ldc);
                                    SLC_DGEMM("T", "N", &nb, &n, &nb, &ONE,
                                              z, &ldqz, &c[pos], &ldc,
                                              &ZERO, dwork, &nb);
                                    SLC_DLACPY("A", &nb, &n, dwork, &nb,
                                               &c[pos], &ldc);
                                }

                                if (wantu) {
                                    SLC_DGEMM("N", "N", &n, &nb, &nb, &ONE,
                                              &u1[pos*ldu1], &ldu1, z, &ldqz,
                                              &ZERO, dwork, &n);
                                    SLC_DLACPY("A", &n, &nb, dwork, &n,
                                               &u1[pos*ldu1], &ldu1);
                                    SLC_DGEMM("N", "N", &n, &nb, &nb, &ONE,
                                              &u2[pos*ldu2], &ldu2, z, &ldqz,
                                              &ZERO, dwork, &n);
                                    SLC_DLACPY("A", &n, &nb, dwork, &n,
                                               &u2[pos*ldu2], &ldu2);
                                }

                                if (wantv) {
                                    SLC_DGEMM("N", "N", &n, &nb, &nb, &ONE,
                                              &v1[pos*ldv1], &ldv1, q, &ldqz,
                                              &ZERO, dwork, &n);
                                    SLC_DLACPY("A", &n, &nb, dwork, &n,
                                               &v1[pos*ldv1], &ldv1);
                                    SLC_DGEMM("N", "N", &n, &nb, &nb, &ONE,
                                              &v2[pos*ldv2], &ldv2, q, &ldqz,
                                              &ZERO, dwork, &n);
                                    SLC_DLACPY("A", &n, &nb, dwork, &n,
                                               &v2[pos*ldv2], &ldv2);
                                }

                                here--;
                            } else {
                                if (a[here + (here-1)*lda] == ZERO)
                                    nbnext = 1;

                                if (nbnext == 2) {
                                    pos = here - 1;
                                    nb = 3;

                                    for (i32 j = 0; j < nb; j++) {
                                        for (i32 i = 0; i < nb; i++) {
                                            q[i + j*ldqz] = (i == j) ? ONE : ZERO;
                                            z[i + j*ldqz] = (i == j) ? ONE : ZERO;
                                        }
                                    }

                                    mb03wa(true, true, 2, 1,
                                           &a[pos + pos*lda], lda,
                                           &b[pos + pos*ldb], ldb,
                                           q, ldqz, z, ldqz, &ierr);

                                    if (ierr != 0) {
                                        dwork[0] = (f64)wrkmin;
                                        *info = 1;
                                        return;
                                    }

                                    if (pos > 0) {
                                        SLC_DGEMM("N", "N", &pos, &nb, &nb, &ONE,
                                                  &a[pos*lda], &lda, z, &ldqz,
                                                  &ZERO, dwork, &n);
                                        SLC_DLACPY("A", &pos, &nb, dwork, &n,
                                                   &a[pos*lda], &lda);
                                    }
                                    if (pos + nb <= n - 1) {
                                        i32 ncols = n - pos - nb;
                                        SLC_DGEMM("T", "N", &nb, &ncols, &nb, &ONE,
                                                  q, &ldqz, &a[pos + (pos+nb)*lda],
                                                  &lda, &ZERO, dwork, &nb);
                                        SLC_DLACPY("A", &nb, &ncols, dwork, &nb,
                                                   &a[pos + (pos+nb)*lda], &lda);
                                    }

                                    if (pos > 0) {
                                        SLC_DGEMM("N", "N", &pos, &nb, &nb, &ONE,
                                                  &b[pos*ldb], &ldb, q, &ldqz,
                                                  &ZERO, dwork, &n);
                                        SLC_DLACPY("A", &pos, &nb, dwork, &n,
                                                   &b[pos*ldb], &ldb);
                                    }
                                    if (pos + nb <= n - 1) {
                                        i32 ncols = n - pos - nb;
                                        SLC_DGEMM("T", "N", &nb, &ncols, &nb, &ONE,
                                                  z, &ldqz, &b[pos + (pos+nb)*ldb],
                                                  &ldb, &ZERO, dwork, &nb);
                                        SLC_DLACPY("A", &nb, &ncols, dwork, &nb,
                                                   &b[pos + (pos+nb)*ldb], &ldb);
                                    }

                                    if (wantc) {
                                        SLC_DGEMM("N", "N", &n, &nb, &nb, &ONE,
                                                  &c[pos*ldc], &ldc, q, &ldqz,
                                                  &ZERO, dwork, &n);
                                        SLC_DLACPY("A", &n, &nb, dwork, &n,
                                                   &c[pos*ldc], &ldc);
                                        SLC_DGEMM("T", "N", &nb, &n, &nb, &ONE,
                                                  z, &ldqz, &c[pos], &ldc,
                                                  &ZERO, dwork, &nb);
                                        SLC_DLACPY("A", &nb, &n, dwork, &nb,
                                                   &c[pos], &ldc);
                                    }

                                    if (wantu) {
                                        SLC_DGEMM("N", "N", &n, &nb, &nb, &ONE,
                                                  &u1[pos*ldu1], &ldu1, z, &ldqz,
                                                  &ZERO, dwork, &n);
                                        SLC_DLACPY("A", &n, &nb, dwork, &n,
                                                   &u1[pos*ldu1], &ldu1);
                                        SLC_DGEMM("N", "N", &n, &nb, &nb, &ONE,
                                                  &u2[pos*ldu2], &ldu2, z, &ldqz,
                                                  &ZERO, dwork, &n);
                                        SLC_DLACPY("A", &n, &nb, dwork, &n,
                                                   &u2[pos*ldu2], &ldu2);
                                    }

                                    if (wantv) {
                                        SLC_DGEMM("N", "N", &n, &nb, &nb, &ONE,
                                                  &v1[pos*ldv1], &ldv1, q, &ldqz,
                                                  &ZERO, dwork, &n);
                                        SLC_DLACPY("A", &n, &nb, dwork, &n,
                                                   &v1[pos*ldv1], &ldv1);
                                        SLC_DGEMM("N", "N", &n, &nb, &nb, &ONE,
                                                  &v2[pos*ldv2], &ldv2, q, &ldqz,
                                                  &ZERO, dwork, &n);
                                        SLC_DLACPY("A", &n, &nb, dwork, &n,
                                                   &v2[pos*ldv2], &ldv2);
                                    }

                                    here -= 2;
                                } else {
                                    pos = here;
                                    nb = 2;

                                    for (i32 j = 0; j < nb; j++) {
                                        for (i32 i = 0; i < nb; i++) {
                                            q[i + j*ldqz] = (i == j) ? ONE : ZERO;
                                            z[i + j*ldqz] = (i == j) ? ONE : ZERO;
                                        }
                                    }

                                    mb03wa(true, true, 2, 1,
                                           &a[pos + pos*lda], lda,
                                           &b[pos + pos*ldb], ldb,
                                           q, ldqz, z, ldqz, &ierr);

                                    if (ierr != 0) {
                                        dwork[0] = (f64)wrkmin;
                                        *info = 1;
                                        return;
                                    }

                                    if (pos > 0) {
                                        SLC_DGEMM("N", "N", &pos, &nb, &nb, &ONE,
                                                  &a[pos*lda], &lda, z, &ldqz,
                                                  &ZERO, dwork, &n);
                                        SLC_DLACPY("A", &pos, &nb, dwork, &n,
                                                   &a[pos*lda], &lda);
                                    }
                                    if (pos + nb <= n - 1) {
                                        i32 ncols = n - pos - nb;
                                        SLC_DGEMM("T", "N", &nb, &ncols, &nb, &ONE,
                                                  q, &ldqz, &a[pos + (pos+nb)*lda],
                                                  &lda, &ZERO, dwork, &nb);
                                        SLC_DLACPY("A", &nb, &ncols, dwork, &nb,
                                                   &a[pos + (pos+nb)*lda], &lda);
                                    }

                                    if (pos > 0) {
                                        SLC_DGEMM("N", "N", &pos, &nb, &nb, &ONE,
                                                  &b[pos*ldb], &ldb, q, &ldqz,
                                                  &ZERO, dwork, &n);
                                        SLC_DLACPY("A", &pos, &nb, dwork, &n,
                                                   &b[pos*ldb], &ldb);
                                    }
                                    if (pos + nb <= n - 1) {
                                        i32 ncols = n - pos - nb;
                                        SLC_DGEMM("T", "N", &nb, &ncols, &nb, &ONE,
                                                  z, &ldqz, &b[pos + (pos+nb)*ldb],
                                                  &ldb, &ZERO, dwork, &nb);
                                        SLC_DLACPY("A", &nb, &ncols, dwork, &nb,
                                                   &b[pos + (pos+nb)*ldb], &ldb);
                                    }

                                    if (wantc) {
                                        SLC_DGEMM("N", "N", &n, &nb, &nb, &ONE,
                                                  &c[pos*ldc], &ldc, q, &ldqz,
                                                  &ZERO, dwork, &n);
                                        SLC_DLACPY("A", &n, &nb, dwork, &n,
                                                   &c[pos*ldc], &ldc);
                                        SLC_DGEMM("T", "N", &nb, &n, &nb, &ONE,
                                                  z, &ldqz, &c[pos], &ldc,
                                                  &ZERO, dwork, &nb);
                                        SLC_DLACPY("A", &nb, &n, dwork, &nb,
                                                   &c[pos], &ldc);
                                    }

                                    if (wantu) {
                                        SLC_DGEMM("N", "N", &n, &nb, &nb, &ONE,
                                                  &u1[pos*ldu1], &ldu1, z, &ldqz,
                                                  &ZERO, dwork, &n);
                                        SLC_DLACPY("A", &n, &nb, dwork, &n,
                                                   &u1[pos*ldu1], &ldu1);
                                        SLC_DGEMM("N", "N", &n, &nb, &nb, &ONE,
                                                  &u2[pos*ldu2], &ldu2, z, &ldqz,
                                                  &ZERO, dwork, &n);
                                        SLC_DLACPY("A", &n, &nb, dwork, &n,
                                                   &u2[pos*ldu2], &ldu2);
                                    }

                                    if (wantv) {
                                        SLC_DGEMM("N", "N", &n, &nb, &nb, &ONE,
                                                  &v1[pos*ldv1], &ldv1, q, &ldqz,
                                                  &ZERO, dwork, &n);
                                        SLC_DLACPY("A", &n, &nb, dwork, &n,
                                                   &v1[pos*ldv1], &ldv1);
                                        SLC_DGEMM("N", "N", &n, &nb, &nb, &ONE,
                                                  &v2[pos*ldv2], &ldv2, q, &ldqz,
                                                  &ZERO, dwork, &n);
                                        SLC_DLACPY("A", &n, &nb, dwork, &n,
                                                   &v2[pos*ldv2], &ldv2);
                                    }

                                    pos = here - 1;
                                    nb = 2;

                                    for (i32 j = 0; j < nb; j++) {
                                        for (i32 i = 0; i < nb; i++) {
                                            q[i + j*ldqz] = (i == j) ? ONE : ZERO;
                                            z[i + j*ldqz] = (i == j) ? ONE : ZERO;
                                        }
                                    }

                                    mb03wa(true, true, 2, 1,
                                           &a[pos + pos*lda], lda,
                                           &b[pos + pos*ldb], ldb,
                                           q, ldqz, z, ldqz, &ierr);

                                    if (ierr != 0) {
                                        dwork[0] = (f64)wrkmin;
                                        *info = 1;
                                        return;
                                    }

                                    if (pos > 0) {
                                        SLC_DGEMM("N", "N", &pos, &nb, &nb, &ONE,
                                                  &a[pos*lda], &lda, z, &ldqz,
                                                  &ZERO, dwork, &n);
                                        SLC_DLACPY("A", &pos, &nb, dwork, &n,
                                                   &a[pos*lda], &lda);
                                    }
                                    if (pos + nb <= n - 1) {
                                        i32 ncols = n - pos - nb;
                                        SLC_DGEMM("T", "N", &nb, &ncols, &nb, &ONE,
                                                  q, &ldqz, &a[pos + (pos+nb)*lda],
                                                  &lda, &ZERO, dwork, &nb);
                                        SLC_DLACPY("A", &nb, &ncols, dwork, &nb,
                                                   &a[pos + (pos+nb)*lda], &lda);
                                    }

                                    if (pos > 0) {
                                        SLC_DGEMM("N", "N", &pos, &nb, &nb, &ONE,
                                                  &b[pos*ldb], &ldb, q, &ldqz,
                                                  &ZERO, dwork, &n);
                                        SLC_DLACPY("A", &pos, &nb, dwork, &n,
                                                   &b[pos*ldb], &ldb);
                                    }
                                    if (pos + nb <= n - 1) {
                                        i32 ncols = n - pos - nb;
                                        SLC_DGEMM("T", "N", &nb, &ncols, &nb, &ONE,
                                                  z, &ldqz, &b[pos + (pos+nb)*ldb],
                                                  &ldb, &ZERO, dwork, &nb);
                                        SLC_DLACPY("A", &nb, &ncols, dwork, &nb,
                                                   &b[pos + (pos+nb)*ldb], &ldb);
                                    }

                                    if (wantc) {
                                        SLC_DGEMM("N", "N", &n, &nb, &nb, &ONE,
                                                  &c[pos*ldc], &ldc, q, &ldqz,
                                                  &ZERO, dwork, &n);
                                        SLC_DLACPY("A", &n, &nb, dwork, &n,
                                                   &c[pos*ldc], &ldc);
                                        SLC_DGEMM("T", "N", &nb, &n, &nb, &ONE,
                                                  z, &ldqz, &c[pos], &ldc,
                                                  &ZERO, dwork, &nb);
                                        SLC_DLACPY("A", &nb, &n, dwork, &nb,
                                                   &c[pos], &ldc);
                                    }

                                    if (wantu) {
                                        SLC_DGEMM("N", "N", &n, &nb, &nb, &ONE,
                                                  &u1[pos*ldu1], &ldu1, z, &ldqz,
                                                  &ZERO, dwork, &n);
                                        SLC_DLACPY("A", &n, &nb, dwork, &n,
                                                   &u1[pos*ldu1], &ldu1);
                                        SLC_DGEMM("N", "N", &n, &nb, &nb, &ONE,
                                                  &u2[pos*ldu2], &ldu2, z, &ldqz,
                                                  &ZERO, dwork, &n);
                                        SLC_DLACPY("A", &n, &nb, dwork, &n,
                                                   &u2[pos*ldu2], &ldu2);
                                    }

                                    if (wantv) {
                                        SLC_DGEMM("N", "N", &n, &nb, &nb, &ONE,
                                                  &v1[pos*ldv1], &ldv1, q, &ldqz,
                                                  &ZERO, dwork, &n);
                                        SLC_DLACPY("A", &n, &nb, dwork, &n,
                                                   &v1[pos*ldv1], &ldv1);
                                        SLC_DGEMM("N", "N", &n, &nb, &nb, &ONE,
                                                  &v2[pos*ldv2], &ldv2, q, &ldqz,
                                                  &ZERO, dwork, &n);
                                        SLC_DLACPY("A", &n, &nb, dwork, &n,
                                                   &v2[pos*ldv2], &ldv2);
                                    }

                                    here -= 2;
                                }
                            }
                        }
                    }

                    if (pair) ks++;
                }
            }
        }
    }

    i32 mm = *m;

    if (initw) {
        i32 ldw2 = 2 * mm;
        SLC_DLASET("A", &ldw2, &ldw2, &ZERO, &ONE, w, &ldw);
    }

    i32 pwc = 0;
    i32 pwd = pwc + 2 * mm;
    i32 pw = pwd + 2 * mm;

    bool pair = false;
    i32 nb = 1;
    i32 nbl;

    for (i32 k = 0; k < mm; k++) {
        if (pair) {
            pair = false;
            nb = 1;
        } else {
            if (k < n - 1) {
                if (a[(k+1) + k*lda] != ZERO) {
                    pair = true;
                    nb = 2;
                }
            }

            i32 pwck = pwc + 2 * k;
            i32 pwdl = pwd + 2 * k;

            i32 len = mm - k;
            for (i32 j = 0; j < len; j++) {
                for (i32 i = 0; i < nb; i++) {
                    dwork[pwck + i + j*2] = ZERO;
                }
            }

            for (i32 j = 0; j < len; j++) {
                for (i32 i = 0; i < nb; i++) {
                    dwork[pwdl + i + j*2] = a[(k + i) + (k + j)*lda];
                }
            }

            for (i32 j = 0; j < len; j++) {
                for (i32 i = 0; i < nb; i++) {
                    a[(k + i) + (k + j)*lda] = ZERO;
                }
            }

            i32 l = k;

            while (l >= 0) {
                if (k == l) {
                    nbl = nb;

                    i32 nbpnbl = nb + nbl;
                    for (i32 j = 0; j < nbpnbl; j++) {
                        for (i32 i = 0; i < nbpnbl; i++) {
                            t[i + j*ldqz] = ZERO;
                        }
                    }

                    for (i32 j = 0; j < nbl; j++) {
                        for (i32 i = 0; i <= j; i++) {
                            t[(nb + i) + j*ldqz] = b[(l + i) + (l + j)*ldb];
                        }
                    }

                    if (nb == 1) {
                        dwork[pwdl] = -dwork[pwdl];
                    } else {
                        for (i32 i = 0; i < 2*nb; i++) {
                            dwork[pwdl + i] = -dwork[pwdl + i];
                        }
                    }

                    for (i32 j = 0; j < nb; j++) {
                        for (i32 i = 0; i < nb; i++) {
                            t[i + (nb + j)*ldqz] = dwork[pwdl + i + j*2];
                        }
                    }
                } else {
                    i32 nbpnbl = nbl + nb;
                    for (i32 j = 0; j < nbpnbl; j++) {
                        for (i32 i = 0; i < nbpnbl; i++) {
                            t[i + j*ldqz] = ZERO;
                        }
                    }

                    for (i32 j = 0; j < nbl; j++) {
                        for (i32 i = 0; i < nbl; i++) {
                            t[i + j*ldqz] = a[(l + i) + (l + j)*lda];
                        }
                    }

                    for (i32 j = 0; j < nb; j++) {
                        for (i32 i = 0; i < nbl; i++) {
                            t[i + (nbl + j)*ldqz] = b[(l + i) + (k + j)*ldb];
                        }
                    }

                    for (i32 j = 0; j < nb; j++) {
                        for (i32 i = 0; i < nb; i++) {
                            t[(nbl + i) + (nbl + j)*ldqz] = dwork[pwck + i + j*2];
                        }
                    }

                    pwdl = pwd + 2 * l;
                }

                i32 nbpnbl = nb + nbl;
                i32 sdim;
                i32 ierr;
                i32 lwork12 = 12;
                i32 bwork[4];

                SLC_DGEES("V", "N", lfdum, &nbpnbl, t, &ldqz, &sdim,
                          wrnew, winew, q, &ldqz, dw12, &lwork12, bwork, &ierr);

                if (ierr != 0) {
                    dwork[0] = (f64)wrkmin;
                    *info = 3;
                    return;
                }

                i32 mm_sel = 0;
                for (i32 i = 0; i < nbpnbl; i++) {
                    if (wrnew[i] > ZERO) {
                        mm_sel++;
                        selnew[i] = 1;
                    } else {
                        selnew[i] = 0;
                    }
                }

                if (mm_sel < nb) {
                    dwork[0] = (f64)wrkmin;
                    *info = 4;
                    return;
                }

                f64 temp;
                i32 lwork4 = 4;
                i32 liwork1 = 1;
                SLC_DTRSEN("N", "V", selnew, &nbpnbl, t, &ldqz, q, &ldqz,
                           wrnew, winew, &mm_sel, &temp, &temp, dw12, &lwork4,
                           idum, &liwork1, &ierr);

                if (ierr != 0) {
                    dwork[0] = (f64)wrkmin;
                    *info = 2;
                    return;
                }

                if (k != l) {
                    for (i32 j = 0; j < nbpnbl; j++) {
                        for (i32 i = 0; i < nbl; i++) {
                            z[(nb + i) + j*ldqz] = q[i + j*ldqz];
                        }
                    }
                    for (i32 j = 0; j < nbpnbl; j++) {
                        for (i32 i = 0; i < nb; i++) {
                            z[i + j*ldqz] = q[(nbl + i) + j*ldqz];
                        }
                    }
                    for (i32 j = 0; j < nbpnbl; j++) {
                        for (i32 i = 0; i < nbpnbl; i++) {
                            q[i + j*ldqz] = z[i + j*ldqz];
                        }
                    }
                }

                for (i32 j = 0; j < nb; j++) {
                    for (i32 i = 0; i < nb; i++) {
                        dwork[pwck + i + j*2] = t[i + j*ldqz];
                    }
                }
                for (i32 j = 0; j < nbl; j++) {
                    for (i32 i = 0; i < nb; i++) {
                        dwork[pwdl + i + j*2] = t[i + (nb + j)*ldqz];
                    }
                }
                if (nb == 1) {
                    for (i32 j = 0; j < nbl; j++) {
                        dwork[pwdl + j*2] = -dwork[pwdl + j*2];
                    }
                } else {
                    for (i32 j = 0; j < nbl; j++) {
                        for (i32 i = 0; i < 2; i++) {
                            dwork[pwdl + i + j*2] = -dwork[pwdl + i + j*2];
                        }
                    }
                }
                for (i32 j = 0; j < nbl; j++) {
                    for (i32 i = 0; i < nbl; i++) {
                        a[(l + i) + (l + j)*lda] = t[(nb + i) + (nb + j)*ldqz];
                    }
                }

                i32 len_up = l;
                if (len_up > 0) {
                    SLC_DGEMM("N", "N", &len_up, &nb, &nb, &ONE,
                              &b[k*ldb], &ldb, q, &ldqz, &ZERO,
                              &dwork[pw], &mm);
                    i32 mm2 = 2 * mm;
                    SLC_DGEMM("N", "N", &len_up, &nbl, &nb, &ONE,
                              &b[k*ldb], &ldb, &q[nb*ldqz], &ldqz, &ZERO,
                              &dwork[pw + mm2], &mm);
                    SLC_DGEMM("N", "N", &len_up, &nb, &nbl, &ONE,
                              &a[l*lda], &lda, &q[nb], &ldqz, &ONE,
                              &dwork[pw], &mm);
                    SLC_DLACPY("A", &len_up, &nb, &dwork[pw], &mm,
                               &b[k*ldb], &ldb);
                    SLC_DGEMM("N", "N", &len_up, &nbl, &nbl, &ONE,
                              &a[l*lda], &lda, &q[nb + nb*ldqz], &ldqz, &ONE,
                              &dwork[pw + mm2], &mm);
                    SLC_DLACPY("A", &len_up, &nbl, &dwork[pw + mm2], &mm,
                               &a[l*lda], &lda);
                }

                len = mm - l - nbl;
                if (len > 0) {
                    i32 pwdl_plus = pwdl + 2*nbl;
                    SLC_DGEMM("T", "N", &nb, &len, &nb, &ONE,
                              q, &ldqz, &dwork[pwdl_plus], &int2, &ZERO,
                              &dwork[pw], &int2);
                    f64 neg1 = -ONE;
                    i32 mm2 = 2 * mm;
                    SLC_DGEMM("T", "N", &nbl, &len, &nb, &neg1,
                              &q[nb*ldqz], &ldqz, &dwork[pwdl_plus], &int2, &ZERO,
                              &dwork[pw + mm2], &int2);
                    SLC_DGEMM("T", "N", &nb, &len, &nbl, &neg1,
                              &q[nb], &ldqz, &a[l + (l+nbl)*lda], &lda, &ONE,
                              &dwork[pw], &int2);
                    SLC_DLACPY("A", &nb, &len, &dwork[pw], &int2,
                               &dwork[pwdl_plus], &int2);
                    SLC_DGEMM("T", "N", &nbl, &len, &nbl, &ONE,
                              &q[nb + nb*ldqz], &ldqz, &a[l + (l+nbl)*lda], &lda, &ONE,
                              &dwork[pw + mm2], &int2);
                    SLC_DLACPY("A", &nbl, &len, &dwork[pw + mm2], &int2,
                               &a[l + (l+nbl)*lda], &lda);
                }

                len = mm - k - nb;
                if (len > 0) {
                    i32 pwck_plus = pwck + 2*nb;
                    SLC_DGEMM("T", "N", &nb, &len, &nb, &ONE,
                              q, &ldqz, &dwork[pwck_plus], &int2, &ZERO,
                              &dwork[pw], &int2);
                    i32 mm2 = 2 * mm;
                    SLC_DGEMM("T", "N", &nbl, &len, &nb, &ONE,
                              &q[nb*ldqz], &ldqz, &dwork[pwck_plus], &int2, &ZERO,
                              &dwork[pw + mm2], &int2);
                    SLC_DGEMM("T", "N", &nb, &len, &nbl, &ONE,
                              &q[nb], &ldqz, &b[l + (k+nb)*ldb], &ldb, &ONE,
                              &dwork[pw], &int2);
                    SLC_DLACPY("A", &nb, &len, &dwork[pw], &int2,
                               &dwork[pwck_plus], &int2);
                    SLC_DGEMM("T", "N", &nbl, &len, &nbl, &ONE,
                              &q[nb + nb*ldqz], &ldqz, &b[l + (k+nb)*ldb], &ldb, &ONE,
                              &dwork[pw + mm2], &int2);
                    SLC_DLACPY("A", &nbl, &len, &dwork[pw + mm2], &int2,
                               &b[l + (k+nb)*ldb], &ldb);
                }

                if (wantw) {
                    i32 posw, lenw;
                    if (initw) {
                        posw = l;
                        lenw = k + nb - l;
                    } else {
                        posw = 0;
                        lenw = mm;
                    }

                    SLC_DGEMM("N", "N", &lenw, &nb, &nb, &ONE,
                              &w[posw + k*ldw], &ldw, q, &ldqz, &ZERO,
                              &dwork[pw], &mm);
                    i32 mm2 = 2 * mm;
                    SLC_DGEMM("N", "N", &lenw, &nbl, &nb, &ONE,
                              &w[posw + k*ldw], &ldw, &q[nb*ldqz], &ldqz, &ZERO,
                              &dwork[pw + mm2], &mm);
                    SLC_DGEMM("N", "N", &lenw, &nb, &nbl, &ONE,
                              &w[posw + (mm+l)*ldw], &ldw, &q[nb], &ldqz, &ONE,
                              &dwork[pw], &mm);
                    SLC_DLACPY("A", &lenw, &nb, &dwork[pw], &mm,
                               &w[posw + k*ldw], &ldw);
                    SLC_DGEMM("N", "N", &lenw, &nbl, &nbl, &ONE,
                              &w[posw + (mm+l)*ldw], &ldw, &q[nb + nb*ldqz], &ldqz, &ONE,
                              &dwork[pw + mm2], &mm);
                    SLC_DLACPY("A", &lenw, &nbl, &dwork[pw + mm2], &mm,
                               &w[posw + (mm+l)*ldw], &ldw);

                    SLC_DGEMM("N", "N", &lenw, &nb, &nb, &ONE,
                              &w[(mm + posw) + k*ldw], &ldw, q, &ldqz, &ZERO,
                              &dwork[pw], &mm);
                    SLC_DGEMM("N", "N", &lenw, &nbl, &nb, &ONE,
                              &w[(mm + posw) + k*ldw], &ldw, &q[nb*ldqz], &ldqz, &ZERO,
                              &dwork[pw + mm2], &mm);
                    SLC_DGEMM("N", "N", &lenw, &nb, &nbl, &ONE,
                              &w[(mm + posw) + (mm+l)*ldw], &ldw, &q[nb], &ldqz, &ONE,
                              &dwork[pw], &mm);
                    SLC_DLACPY("A", &lenw, &nb, &dwork[pw], &mm,
                               &w[(mm + posw) + k*ldw], &ldw);
                    SLC_DGEMM("N", "N", &lenw, &nbl, &nbl, &ONE,
                              &w[(mm + posw) + (mm+l)*ldw], &ldw, &q[nb + nb*ldqz], &ldqz, &ONE,
                              &dwork[pw + mm2], &mm);
                    SLC_DLACPY("A", &lenw, &nbl, &dwork[pw + mm2], &mm,
                               &w[(mm + posw) + (mm+l)*ldw], &ldw);
                }

                l--;
                nbl = 1;
                if (l > 0) {
                    if (a[l + (l-1)*lda] != ZERO) {
                        nbl = 2;
                        l--;
                    }
                }
            }

            SLC_DCOPY(&nb, wrnew, &int1, &wr[k], &int1);
            SLC_DCOPY(&nb, winew, &int1, &wi[k], &int1);
        }
    }

    dwork[0] = (f64)wrkmin;
}
