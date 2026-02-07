/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB02ND - Optimal state feedback matrix for optimal control problem
 *
 * Computes:
 *   F = (R + B'XB)^(-1) (B'XA + L')  [discrete-time]
 *   F = R^(-1) (B'X + L')            [continuous-time]
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <ctype.h>
#include <math.h>

void sb02nd(
    const char* dico_str,
    const char* fact_str,
    const char* uplo_str,
    const char* jobl_str,
    const i32 n,
    const i32 m,
    const i32 p,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* r,
    const i32 ldr,
    i32* ipiv,
    f64* l,
    const i32 ldl,
    f64* x,
    const i32 ldx,
    const f64 rnorm,
    f64* f,
    const i32 ldf,
    i32* oufact,
    f64* dwork,
    const i32 ldwork,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 mone = -1.0;
    const f64 two = 2.0;
    const i32 int1 = 1;

    char dico = toupper((unsigned char)dico_str[0]);
    char fact = toupper((unsigned char)fact_str[0]);
    char uplo = toupper((unsigned char)uplo_str[0]);
    char jobl = toupper((unsigned char)jobl_str[0]);

    bool discr = (dico == 'D');
    bool lfactc = (fact == 'C');
    bool lfactd = (fact == 'D');
    bool lfactu = (fact == 'U');
    bool luplou = (uplo == 'U');
    bool withl = (jobl == 'N');
    bool lfacta = lfactc || lfactd || lfactu;
    bool lnfact = !lfacta;

    *info = 0;

    if (!discr && dico != 'C') {
        *info = -1;
    } else if ((lnfact && fact != 'N') || (discr && lfactu)) {
        *info = -2;
    } else if (!luplou && uplo != 'L') {
        *info = -3;
    } else if (!withl && jobl != 'Z') {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (m < 0) {
        *info = -6;
    } else if (lfactd && (p < 0 || (!discr && p < m))) {
        *info = -7;
    } else if (lda < 1 || (discr && lda < n)) {
        *info = -9;
    } else if (ldb < 1 || (n > 0 && ldb < n)) {
        *info = -11;
    } else if (ldr < 1 || (m > 0 && ldr < m) || (lfactd && ldr < p)) {
        *info = -13;
    } else if (ldl < 1 || (withl && ldl < n)) {
        *info = -16;
    } else if (ldx < 1 || (n > 0 && ldx < n)) {
        *info = -18;
    } else if (lfactu && rnorm < zero) {
        *info = -19;
    } else if (ldf < 1 || (m > 0 && ldf < m)) {
        *info = -21;
    }

    i32 wrkmin = 2;
    if (*info == 0) {
        if (discr) {
            if (lnfact) {
                i32 tmp = 3 * m > n ? 3 * m : n;
                wrkmin = tmp > 2 ? tmp : 2;
            } else {
                i32 tmp1 = n + 3 * m + 2;
                i32 tmp2 = 4 * n + 1;
                wrkmin = tmp1 > tmp2 ? tmp1 : tmp2;
            }
        } else {
            if (lfactu) {
                wrkmin = 2 * m > 2 ? 2 * m : 2;
            } else {
                wrkmin = 3 * m > 2 ? 3 * m : 2;
            }
        }
        if (ldwork < wrkmin) {
            *info = -25;
            dwork[0] = (f64)wrkmin;
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 || m == 0) {
        dwork[0] = two;
        if (n == 0) {
            dwork[1] = one;
        } else {
            dwork[1] = zero;
        }
        return;
    }

    const char* nt = "N";
    const char* tr = "T";
    f64 eps = SLC_DLAMCH("P");

    i32 wrkopt = n * m > wrkmin ? n * m : wrkmin;
    bool sufwrk = ldwork >= n * m;

    if (sufwrk) {
        if (discr || !withl) {
            SLC_DSYMM("L", &uplo, &n, &m, &one, x, &ldx, b, &ldb, &zero, dwork, &n);
            if (withl) {
                ma02ad("A", n, m, l, ldl, f, ldf);
                SLC_DGEMM(tr, nt, &m, &n, &n, &one, dwork, &n, a, &lda, &one, f, &ldf);
            } else if (discr) {
                SLC_DGEMM(tr, nt, &m, &n, &n, &one, dwork, &n, a, &lda, &zero, f, &ldf);
            } else {
                ma02ad("A", n, m, dwork, n, f, ldf);
            }
        } else {
            SLC_DLACPY("A", &n, &m, l, &ldl, dwork, &n);
            SLC_DSYMM("L", &uplo, &n, &m, &one, x, &ldx, b, &ldb, &one, dwork, &n);
            ma02ad("A", n, m, dwork, n, f, ldf);
        }
    } else {
        SLC_DGEMM(tr, nt, &m, &n, &n, &one, b, &ldb, x, &ldx, &zero, f, &ldf);
    }

    f64 rnormp = 0.0;

    if (lnfact) {
        if (discr) {
            if (sufwrk) {
                mb01rb("L", &uplo, tr, m, n, one, one, r, ldr, dwork, n, b, ldb, info);
            } else {
                mb01rb("L", &uplo, nt, m, n, one, one, r, ldr, f, ldf, b, ldb, info);
            }
        }

        i32 iwork_m[256];
        i32* iwork = (m <= 256) ? iwork_m : (i32*)malloc(m * sizeof(i32));
        if (!iwork) {
            *info = -24;
            return;
        }

        rnormp = SLC_DLANSY("1", &uplo, &m, r, &ldr, dwork);

        if (iwork != iwork_m) free(iwork);
    }

    if (discr && !sufwrk) {
        i32 ms = ldwork / n > 1 ? ldwork / n : 1;
        for (i32 i = 0; i < m; i += ms) {
            i32 nr = ms < (m - i) ? ms : (m - i);
            SLC_DLACPY("A", &nr, &n, &f[i], &ldf, dwork, &nr);
            SLC_DGEMM(nt, nt, &nr, &n, &n, &one, dwork, &nr, a, &lda, &zero, &f[i], &ldf);
        }
    }

    if (withl && !sufwrk) {
        for (i32 i = 0; i < m; i++) {
            SLC_DAXPY(&n, &one, &l[i * ldl], &int1, &f[i], &ldf);
        }
    }

    f64 rcond = 0.0;

    if (lfacta) {
        if (lfactd) {
            i32 jw = (p < m ? p : m) + 1;
            i32 ldw_jw = ldwork - jw + 1;
            SLC_DGEQRF(&p, &m, r, &ldr, dwork, &dwork[jw - 1], &ldw_jw, info);
            i32 opt = (i32)dwork[jw - 1] + jw - 1;
            wrkopt = opt > wrkopt ? opt : wrkopt;

            if (p < m) {
                SLC_DLASET("F", &(i32){m - p}, &m, &zero, &zero, &r[p], &ldr);
            }

            for (i32 i = 0; i < m; i++) {
                if (r[i + i * ldr] < zero) {
                    i32 cnt = m - i;
                    SLC_DSCAL(&cnt, &mone, &r[i + i * ldr], &ldr);
                }
                if (!luplou) {
                    for (i32 j = 0; j < i; j++) {
                        r[i + j * ldr] = r[j + i * ldr];
                    }
                }
            }
        }

        if (discr) {
            i32 jz = 0;
            char nuplo = luplou ? 'L' : 'U';

            SLC_DCOPY(&n, x, &(i32){ldx + 1}, dwork, &int1);
            SLC_DPOTRF(&uplo, &n, x, &ldx, info);

            if (*info == 0) {
                oufact[1] = 1;
                const char* trl = luplou ? nt : tr;
                SLC_DTRMM("L", &uplo, trl, "N", &n, &m, &one, x, &ldx, b, &ldb);
            } else {
                i32 jw = n + 3;
                oufact[1] = 2;
                SLC_DCOPY(&n, dwork, &int1, x, &(i32){ldx + 1});

                i32 ldw_jw = ldwork - jw + 1;
                SLC_DSYEV("V", &nuplo, &n, x, &ldx, &dwork[2], &dwork[jw - 1], &ldw_jw, info);
                if (*info > 0) {
                    *info = m + 2;
                    return;
                }
                i32 opt = (i32)dwork[jw - 1] + jw - 1;
                wrkopt = opt > wrkopt ? opt : wrkopt;

                f64 temp = fabs(dwork[n + 1]) * eps * (f64)n;

                while (jz < n && fabs(dwork[jz + 2]) <= temp) {
                    jz++;
                }

                if (lfactd && n - jz + p < m) {
                    oufact[0] = 1;
                    dwork[1] = zero;
                    *info = m + 1;
                    return;
                }

                if (dwork[jz + 2] < zero) {
                    *info = m + 3;
                    return;
                }

                i32 ms = (ldwork - jw + 1) / n > 1 ? (ldwork - jw + 1) / n : 1;
                for (i32 i = 0; i < m; i += ms) {
                    i32 nr = ms < (m - i) ? ms : (m - i);
                    SLC_DLACPY("A", &n, &nr, &b[i * ldb], &ldb, &dwork[jw - 1], &n);
                    i32 nz_rows = n - jz;
                    SLC_DGEMM(tr, nt, &nz_rows, &nr, &n, &one, &x[jz * ldx], &ldx,
                              &dwork[jw - 1], &n, &zero, &b[jz + i * ldb], &ldb);
                }

                for (i32 i = jz; i < n; i++) {
                    f64 sqrtv = sqrt(dwork[i + 2]);
                    SLC_DSCAL(&m, &sqrtv, &b[i], &ldb);
                }

                jw = n + 3;
            }

            if (!luplou) {
                ma02ed(uplo, m, r, ldr);
            }

            i32 jw = (oufact[1] == 1) ? 1 : n + 3;
            f64 dummy;
            i32 nz_rows = n - jz;
            mb04kd('F', m, 0, nz_rows, r, ldr, &b[jz], ldb, &dummy, n, &dummy, m,
                   &dwork[jw - 1], &dwork[jw - 1 + m]);

            for (i32 i = 0; i < m; i++) {
                if (r[i + i * ldr] < zero) {
                    i32 cnt = m - i;
                    SLC_DSCAL(&cnt, &mone, &r[i + i * ldr], &ldr);
                }
            }

            if (!luplou) {
                ma02ed(nuplo, m, r, ldr);
            }
        } else {
        }

        if (!lfactu) {
            i32 jw = (discr && oufact[1] == 2) ? n + 3 : 1;
            i32 iwork_m[256];
            i32* iwork = (m <= 256) ? iwork_m : (i32*)malloc(m * sizeof(i32));
            if (!iwork) {
                *info = -24;
                return;
            }

            SLC_DTRCON("1", &uplo, "N", &m, r, &ldr, &rcond, &dwork[jw - 1], iwork, info);
            oufact[0] = 1;

            if (iwork != iwork_m) free(iwork);
        } else {
            i32 iwork_m[256];
            i32* iwork = (m <= 256) ? iwork_m : (i32*)malloc(m * sizeof(i32));
            if (!iwork) {
                *info = -24;
                return;
            }

            SLC_DSYCON(&uplo, &m, r, &ldr, ipiv, &rnorm, &rcond, dwork, iwork, info);
            oufact[0] = 2;

            if (iwork != iwork_m) free(iwork);
        }
    } else {
        i32 iwork_m[256];
        i32* iwork = (m <= 256) ? iwork_m : (i32*)malloc(m * sizeof(i32));
        if (!iwork) {
            *info = -24;
            return;
        }

        SLC_DCOPY(&m, r, &(i32){ldr + 1}, dwork, &int1);
        ma02ed(uplo, m, r, ldr);
        SLC_DPOTRF(&uplo, &m, r, &ldr, info);

        if (*info == 0) {
            oufact[0] = 1;
            SLC_DPOCON(&uplo, &m, r, &ldr, &rnormp, &rcond, dwork, iwork, info);
        } else {
            oufact[0] = 2;

            SLC_DCOPY(&m, dwork, &int1, r, &(i32){ldr + 1});

            char nuplo = luplou ? 'L' : 'U';
            ma02ed(nuplo, m, r, ldr);

            SLC_DSYTRF(&uplo, &m, r, &ldr, ipiv, dwork, &ldwork, info);
            if (*info > 0) {
                dwork[1] = zero;
                if (iwork != iwork_m) free(iwork);
                return;
            }
            i32 opt = (i32)dwork[0];
            wrkopt = opt > wrkopt ? opt : wrkopt;

            SLC_DSYCON(&uplo, &m, r, &ldr, ipiv, &rnormp, &rcond, dwork, iwork, info);
        }

        if (iwork != iwork_m) free(iwork);
    }

    dwork[1] = rcond;
    if (rcond < eps) {
        *info = m + 1;
        return;
    }

    if (oufact[0] == 1) {
        SLC_DPOTRS(&uplo, &m, &n, r, &ldr, f, &ldf, info);
    } else {
        SLC_DSYTRS(&uplo, &m, &n, r, &ldr, ipiv, f, &ldf, info);
    }

    dwork[0] = (f64)wrkopt;
}
