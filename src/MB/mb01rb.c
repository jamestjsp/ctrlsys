#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>
#include <math.h>

void mb01rb(const char* side_str, const char* uplo_str, const char* trans_str,
            const i32 m, const i32 n, const f64 alpha, const f64 beta,
            f64* r, const i32 ldr, const f64* a, const i32 lda,
            const f64* b, const i32 ldb, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const i32 n1l = 128;
    const i32 n1p = 512;
    const i32 n2l = 40;
    const i32 n2p = 128;
    const i32 nbs = 48;

    i32 i, ib, j, jb, mn, mx, n1, n2, nb, nbmin = 2, nx;
    bool lside, ltrans, luplo;
    f64 d[1];
    char side = side_str[0];
    char uplo = uplo_str[0];
    char trans = trans_str[0];

    *info = 0;
    lside = (side == 'L' || side == 'l');
    luplo = (uplo == 'U' || uplo == 'u');
    ltrans = (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c');

    if (!lside && side != 'R' && side != 'r') {
        *info = -1;
    } else if (!luplo && uplo != 'L' && uplo != 'l') {
        *info = -2;
    } else if (!ltrans && trans != 'N' && trans != 'n') {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (ldr < (m > 1 ? m : 1)) {
        *info = -9;
    } else if (lda < 1 ||
               (((lside && !ltrans) || (!lside && ltrans)) && lda < m) ||
               (((lside && ltrans) || (!lside && !ltrans)) && lda < n)) {
        *info = -11;
    } else if (ldb < 1 || (lside && ldb < n) || (!lside && ldb < m)) {
        *info = -13;
    }

    if (*info != 0) {
        return;
    }

    if (m == 0) {
        return;
    }

    if (beta == zero || n == 0) {
        if (alpha == zero) {
            SLC_DLASET(&uplo, &m, &m, &zero, &zero, r, &ldr);
        } else {
            if (alpha != one) {
                i32 type = 0;
                SLC_DLASCL(&uplo, &type, &type, &one, &alpha, &m, &m, r, &ldr, info);
            }
        }
        return;
    }

    mx = m > n ? m : n;
    mn = m < n ? m : n;
    i32 lwork = -1;
    SLC_DGEQRF(&mx, &mn, (f64*)a, &mx, d, d, &lwork, info);
    nb = ((i32)d[0]) / mn / 8 * 8;

    if (nb > 1 && nb < m) {
        nx = SLC_ILAENV(&(i32){3}, "DGEQRF", " ", &mx, &mn, &(i32){-1}, &(i32){-1});
        if (nx < 0) nx = 0;
        if (nx < m) {
            nbmin = SLC_ILAENV(&(i32){2}, "DGEQRF", " ", &mx, &mn, &(i32){-1}, &(i32){-1});
            if (nbmin < 2) nbmin = 2;
        }
    } else {
        nx = 0;
        nbmin = 2;
    }

    if (nb >= nbmin && nb < m && nx < m) {
        if (ltrans && lside) {
            if (nb <= nbs) {
                n1 = n1l > nb ? n1l : nb;
                if (n1 > n) n1 = n;
            } else {
                n1 = n1p < n ? n1p : n;
            }
        } else {
            if (nb <= nbs) {
                n2 = n2l > nb ? n2l : nb;
                if (n2 > n) n2 = n;
            } else {
                n2 = n2p < n ? n2p : n;
            }
        }

        for (i = 0; i < m - nx; i += nb) {
            ib = i + nb;
            jb = (m - i) < nb ? (m - i) : nb;

            if (ltrans) {
                if (lside) {
                    slicot_mb01rx(side, uplo, trans, jb, n1, alpha, beta,
                                  &r[i + i*ldr], ldr, &a[i*lda], lda, &b[i*ldb], ldb);
                    for (j = n1; j < n; j += n1) {
                        i32 jb2 = (n - j) < n1 ? (n - j) : n1;
                        slicot_mb01rx(side, uplo, trans, jb, jb2, one, beta,
                                     &r[i + i*ldr], ldr, &a[j + i*lda], lda, &b[j + i*ldb], ldb);
                    }
                    if (ib < m) {
                        i32 rem = m - ib;
                        if (luplo) {
                            SLC_DGEMM(&trans, "N", &jb, &rem, &n, &beta, a + i*lda, &lda,
                                     b + ib*ldb, &ldb, &alpha, &r[i + ib*ldr], &ldr);
                        } else {
                            SLC_DGEMM(&trans, "N", &rem, &jb, &n, &beta, a + ib*lda, &lda,
                                     b + i*ldb, &ldb, &alpha, &r[ib + i*ldr], &ldr);
                        }
                    }
                } else {
                    slicot_mb01rx(side, uplo, trans, jb, n2, alpha, beta,
                                  &r[i + i*ldr], ldr, &a[i], lda, &b[i], ldb);
                    for (j = n2; j < n; j += n2) {
                        i32 jb2 = (n - j) < n2 ? (n - j) : n2;
                        slicot_mb01rx(side, uplo, trans, jb, jb2, one, beta,
                                     &r[i + i*ldr], ldr, &a[i + j*lda], lda, &b[i + j*ldb], ldb);
                    }
                    if (ib < m) {
                        i32 rem = m - ib;
                        if (luplo) {
                            SLC_DGEMM("N", &trans, &jb, &rem, &n, &beta, b + i, &ldb,
                                     a + ib, &lda, &alpha, &r[i + ib*ldr], &ldr);
                        } else {
                            SLC_DGEMM("N", &trans, &rem, &jb, &n, &beta, b + ib, &ldb,
                                     a + i, &lda, &alpha, &r[ib + i*ldr], &ldr);
                        }
                    }
                }
            } else {
                if (lside) {
                    slicot_mb01rx(side, uplo, trans, jb, n2, alpha, beta,
                                  &r[i + i*ldr], ldr, &a[i], lda, &b[i*ldb], ldb);
                    for (j = n2; j < n; j += n2) {
                        i32 jb2 = (n - j) < n2 ? (n - j) : n2;
                        slicot_mb01rx(side, uplo, trans, jb, jb2, one, beta,
                                     &r[i + i*ldr], ldr, &a[i + j*lda], lda, &b[j + i*ldb], ldb);
                    }
                    if (ib < m) {
                        i32 rem = m - ib;
                        if (luplo) {
                            SLC_DGEMM(&trans, "N", &jb, &rem, &n, &beta, a + i, &lda,
                                     b + ib*ldb, &ldb, &alpha, &r[i + ib*ldr], &ldr);
                        } else {
                            SLC_DGEMM(&trans, "N", &rem, &jb, &n, &beta, a + ib, &lda,
                                     b + i*ldb, &ldb, &alpha, &r[ib + i*ldr], &ldr);
                        }
                    }
                } else {
                    slicot_mb01rx(side, uplo, trans, jb, n2, alpha, beta,
                                  &r[i + i*ldr], ldr, &a[i*lda], lda, &b[i], ldb);
                    for (j = n2; j < n; j += n2) {
                        i32 jb2 = (n - j) < n2 ? (n - j) : n2;
                        slicot_mb01rx(side, uplo, trans, jb, jb2, one, beta,
                                     &r[i + i*ldr], ldr, &a[j + i*lda], lda, &b[i + j*ldb], ldb);
                    }
                    if (ib < m) {
                        i32 rem = m - ib;
                        if (luplo) {
                            SLC_DGEMM("N", &trans, &jb, &rem, &n, &beta, b + i, &ldb,
                                     a + ib*lda, &lda, &alpha, &r[i + ib*ldr], &ldr);
                        } else {
                            SLC_DGEMM("N", &trans, &rem, &jb, &n, &beta, b + ib, &ldb,
                                     a + i*lda, &lda, &alpha, &r[ib + i*ldr], &ldr);
                        }
                    }
                }
            }
        }
    } else {
        i = 0;
        n1 = n;
        n2 = n;
    }

    if (i < m) {
        i32 rem = m - i;
        if (ltrans) {
            if (lside) {
                slicot_mb01rx(side, uplo, trans, rem, n1, alpha, beta,
                              &r[i + i*ldr], ldr, &a[i*lda], lda, &b[i*ldb], ldb);
                for (j = n1; j < n; j += n1) {
                    i32 jb = (n - j) < n1 ? (n - j) : n1;
                    slicot_mb01rx(side, uplo, trans, rem, jb, one, beta,
                                 &r[i + i*ldr], ldr, &a[j + i*lda], lda, &b[j + i*ldb], ldb);
                }
            } else {
                slicot_mb01rx(side, uplo, trans, rem, n2, alpha, beta,
                              &r[i + i*ldr], ldr, &a[i], lda, &b[i], ldb);
                for (j = n2; j < n; j += n2) {
                    i32 jb = (n - j) < n2 ? (n - j) : n2;
                    slicot_mb01rx(side, uplo, trans, rem, jb, one, beta,
                                 &r[i + i*ldr], ldr, &a[i + j*lda], lda, &b[i + j*ldb], ldb);
                }
            }
        } else {
            if (lside) {
                slicot_mb01rx(side, uplo, trans, rem, n2, alpha, beta,
                              &r[i + i*ldr], ldr, &a[i], lda, &b[i*ldb], ldb);
                for (j = n2; j < n; j += n2) {
                    i32 jb = (n - j) < n2 ? (n - j) : n2;
                    slicot_mb01rx(side, uplo, trans, rem, jb, one, beta,
                                 &r[i + i*ldr], ldr, &a[i + j*lda], lda, &b[j + i*ldb], ldb);
                }
            } else {
                slicot_mb01rx(side, uplo, trans, rem, n2, alpha, beta,
                              &r[i + i*ldr], ldr, &a[i*lda], lda, &b[i], ldb);
                for (j = n2; j < n; j += n2) {
                    i32 jb = (n - j) < n2 ? (n - j) : n2;
                    slicot_mb01rx(side, uplo, trans, rem, jb, one, beta,
                                 &r[i + i*ldr], ldr, &a[j + i*lda], lda, &b[i + j*ldb], ldb);
                }
            }
        }
    }
}
