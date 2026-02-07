/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB02RU - Construct Hamiltonian or symplectic matrix for Riccati equation
 *
 * For continuous-time (DICO='C'):
 *         ( op(A)   -G    )
 *     S = (               )
 *         (  -Q   -op(A)' )
 *
 * For discrete-time (DICO='D'):
 *                 -1              -1
 *         (  op(A)           op(A)  *G       )
 *     S = (        -1                   -1   )  if HINV='D'
 *         ( Q*op(A)     op(A)' + Q*op(A)  *G )
 *
 *                              -T             -T
 *         ( op(A) + G*op(A)  *Q   -G*op(A)   )
 *     S = (           -T                 -T  )  if HINV='I'
 *         (     -op(A)  *Q          op(A)    )
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <ctype.h>

void sb02ru(
    const char* dico_str,
    const char* hinv_str,
    const char* trana_str,
    const char* uplo_str,
    const i32 n,
    f64* a,
    const i32 lda,
    f64* g,
    const i32 ldg,
    f64* q,
    const i32 ldq,
    f64* s,
    const i32 lds,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 mone = -1.0;
    const i32 int1 = 1;

    char dico = toupper((unsigned char)dico_str[0]);
    char hinv = toupper((unsigned char)hinv_str[0]);
    char trana = toupper((unsigned char)trana_str[0]);
    char uplo = toupper((unsigned char)uplo_str[0]);

    bool discr = (dico == 'D');
    bool lhinv = (hinv == 'D');
    bool notrna = (trana == 'N');
    bool luplo = (uplo == 'U');

    i32 n2 = n + n;

    *info = 0;

    if (!discr && dico != 'C') {
        *info = -1;
    } else if (discr && !lhinv && hinv != 'I') {
        *info = -2;
    } else if (!notrna && trana != 'T' && trana != 'C') {
        *info = -3;
    } else if (!luplo && uplo != 'L') {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldg < (n > 1 ? n : 1)) {
        *info = -9;
    } else if (ldq < (n > 1 ? n : 1)) {
        *info = -11;
    } else if (lds < (n2 > 1 ? n2 : 1)) {
        *info = -13;
    } else if (discr && ldwork < (6 * n > 2 ? 6 * n : 2)) {
        *info = -16;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        if (discr) {
            dwork[0] = one;
            dwork[1] = one;
        }
        return;
    }

    i32 np1 = n + 1;

    if (!discr) {
        for (i32 j = 0; j < n; j++) {
            if (notrna) {
                SLC_DCOPY(&n, &a[j * lda], &int1, &s[j * lds], &int1);
            } else {
                for (i32 i = 0; i < n; i++) {
                    s[i + j * lds] = a[j + i * lda];
                }
            }

            if (luplo) {
                for (i32 i = 0; i <= j; i++) {
                    s[n + i + j * lds] = -q[i + j * ldq];
                }
                for (i32 i = j + 1; i < n; i++) {
                    s[n + i + j * lds] = -q[j + i * ldq];
                }
            } else {
                for (i32 i = 0; i < j; i++) {
                    s[n + i + j * lds] = -q[j + i * ldq];
                }
                for (i32 i = j; i < n; i++) {
                    s[n + i + j * lds] = -q[i + j * ldq];
                }
            }
        }

        for (i32 j = 0; j < n; j++) {
            i32 nj = n + j;
            if (luplo) {
                for (i32 i = 0; i <= j; i++) {
                    s[i + nj * lds] = -g[i + j * ldg];
                }
                for (i32 i = j + 1; i < n; i++) {
                    s[i + nj * lds] = -g[j + i * ldg];
                }
            } else {
                for (i32 i = 0; i < j; i++) {
                    s[i + nj * lds] = -g[j + i * ldg];
                }
                for (i32 i = j; i < n; i++) {
                    s[i + nj * lds] = -g[i + j * ldg];
                }
            }

            if (notrna) {
                for (i32 i = 0; i < n; i++) {
                    s[n + i + nj * lds] = -a[j + i * lda];
                }
            } else {
                for (i32 i = 0; i < n; i++) {
                    s[n + i + nj * lds] = -a[i + j * lda];
                }
            }
        }
    } else {
        ma02ed(uplo, n, q, ldq);
        ma02ed(uplo, n, g, ldg);

        const char* tranat = notrna ? "T" : "N";

        i32 info_loc;
        f64 rcond;
        SLC_DLACPY("F", &n, &n, a, &lda, s, &lds);

        SLC_DGETRF(&n, &n, s, &lds, iwork, &info_loc);
        if (info_loc > 0) {
            *info = info_loc;
            dwork[0] = zero;
            dwork[1] = zero;
            return;
        }

        f64 anorm = SLC_DLANGE("1", &n, &n, a, &lda, dwork);
        SLC_DGECON("1", &n, s, &lds, &anorm, &rcond, dwork, &iwork[n], &info_loc);

        if (rcond == zero) {
            *info = n + 1;
            dwork[0] = rcond;
            dwork[1] = one;
            return;
        }

        SLC_DLACPY("F", &n, &n, q, &ldq, &s[np1 - 1], &lds);
        SLC_DGETRS(tranat, &n, &n, s, &lds, iwork, &s[np1 - 1], &lds, &info_loc);

        if (lhinv) {
            for (i32 j = 0; j < n - 1; j++) {
                SLC_DSWAP(&(i32){n - j - 1}, &s[n + j + 1 + j * lds], &int1,
                          &s[n + j + (j + 1) * lds], &lds);
            }

            SLC_DLASET("F", &n, &n, &zero, &one, &s[n + n * lds], &lds);
            SLC_DGETRS(&trana, &n, &n, s, &lds, iwork, &s[n + n * lds], &lds, &info_loc);

            SLC_DLACPY("F", &n, &n, g, &ldg, &s[n * lds], &lds);
            SLC_DGETRS(&trana, &n, &n, s, &lds, iwork, &s[n * lds], &lds, &info_loc);

            SLC_DLACPY("F", &n, &n, &s[n + n * lds], &lds, s, &lds);

            if (notrna) {
                ma02ad("F", n, n, a, lda, &s[n + n * lds], lds);
            } else {
                SLC_DLACPY("F", &n, &n, a, &lda, &s[n + n * lds], &lds);
            }
            SLC_DGEMM("N", "N", &n, &n, &n, &one, q, &ldq,
                      &s[n * lds], &lds, &one, &s[n + n * lds], &lds);
        } else {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = n; i < n2; i++) {
                    s[i + j * lds] = -s[i + j * lds];
                }
            }

            SLC_DLASET("F", &n, &n, &zero, &one, &s[n + n * lds], &lds);
            SLC_DGETRS(tranat, &n, &n, s, &lds, iwork, &s[n + n * lds], &lds, &info_loc);

            SLC_DLACPY("F", &n, &n, g, &ldg, &s[n * lds], &lds);
            SLC_DGETRS(&trana, &n, &n, s, &lds, iwork, &s[n * lds], &lds, &info_loc);

            for (i32 j = n; j < n2; j++) {
                for (i32 i = 0; i <= j - n; i++) {
                    f64 temp = -s[i + j * lds];
                    s[i + j * lds] = -s[j - n + (i + n) * lds];
                    s[j - n + (i + n) * lds] = temp;
                }
            }

            if (notrna) {
                SLC_DLACPY("F", &n, &n, a, &lda, s, &lds);
            } else {
                ma02ad("F", n, n, a, lda, s, lds);
            }
            SLC_DGEMM("N", "N", &n, &n, &n, &mone, g, &ldg,
                      &s[n], &lds, &one, s, &lds);
        }

        dwork[0] = rcond;
        dwork[1] = one;
    }
}
