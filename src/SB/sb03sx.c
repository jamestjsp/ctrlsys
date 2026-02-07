// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <stdbool.h>

void sb03sx(const char* trana, const char* uplo, const char* lyapun, i32 n,
            f64 xanorm, const f64* t, i32 ldt, const f64* u, i32 ldu,
            f64* r, i32 ldr, f64* ferr, i32* iwork, f64* dwork, i32 ldwork, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 half = 0.5;
    i32 nn = n * n;

    char trana_c = (char)toupper((unsigned char)trana[0]);
    char uplo_c = (char)toupper((unsigned char)uplo[0]);
    char lyapun_c = (char)toupper((unsigned char)lyapun[0]);

    bool notrna = (trana_c == 'N');
    bool update = (lyapun_c == 'O');

    *info = 0;
    *ferr = zero;

    if (!notrna && trana_c != 'T' && trana_c != 'C') {
        *info = -1;
    } else if (uplo_c != 'L' && uplo_c != 'U') {
        *info = -2;
    } else if (!update && lyapun_c != 'R') {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (xanorm < zero) {
        *info = -5;
    } else if (ldt < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldu < 1 || (update && ldu < n)) {
        *info = -9;
    } else if (ldr < (n > 1 ? n : 1)) {
        *info = -11;
    } else if (ldwork < 0 || (n > 0 && ldwork < (2 * nn > 3 ? 2 * nn : 3))) {
        *info = -15;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 || xanorm == zero) {
        return;
    }

    i32 itmp = nn;
    char tranat = notrna ? 'T' : 'N';

    ma02ed(uplo_c, n, r, ldr);

    i32 kase = 0;
    i32 isave[3] = {0, 0, 0};
    f64 est = zero;
    f64 scale = one;

    while (1) {
        SLC_DLACN2(&nn, &dwork[itmp], dwork, iwork, &est, &kase, isave);
        if (kase == 0) break;

        f64 upper_norm = SLC_DLANSY("1", "Upper", &n, dwork, &n, &dwork[itmp]);
        f64 lower_norm = SLC_DLANSY("1", "Lower", &n, dwork, &n, &dwork[itmp]);

        char uplow;
        bool lower;
        if (upper_norm >= lower_norm) {
            uplow = 'U';
            lower = false;
        } else {
            uplow = 'L';
            lower = true;
        }

        if (kase == 2) {
            i32 ij = 0;
            if (lower) {
                for (i32 j = 0; j < n; j++) {
                    for (i32 i = j; i < n; i++) {
                        dwork[ij] = dwork[ij] * r[i + j * ldr];
                        ij++;
                    }
                    ij += j + 1;
                }
            } else {
                for (i32 j = 0; j < n; j++) {
                    for (i32 i = 0; i <= j; i++) {
                        dwork[ij] = dwork[ij] * r[i + j * ldr];
                        ij++;
                    }
                    ij += n - j - 1;
                }
            }
        }

        if (update) {
            i32 info2;
            mb01ru(&uplow, "T", n, n, zero, one, dwork, n, u, ldu,
                   dwork, n, &dwork[itmp], nn, &info2);
            i32 n1 = n + 1;
            SLC_DSCAL(&n, &half, dwork, &n1);
        }
        ma02ed(uplow, n, dwork, n);

        i32 info2 = 0;
        if (kase == 2) {
            sb03mx(&trana_c, n, t, ldt, dwork, n, &scale, &dwork[itmp], &info2);
        } else {
            sb03mx(&tranat, n, t, ldt, dwork, n, &scale, &dwork[itmp], &info2);
        }

        if (info2 > 0) {
            *info = n + 1;
        }

        if (update) {
            i32 info3;
            mb01ru(&uplow, "N", n, n, zero, one, dwork, n, u, ldu,
                   dwork, n, &dwork[itmp], nn, &info3);
            i32 n1 = n + 1;
            SLC_DSCAL(&n, &half, dwork, &n1);
        }

        if (kase == 1) {
            i32 ij = 0;
            if (lower) {
                for (i32 j = 0; j < n; j++) {
                    for (i32 i = j; i < n; i++) {
                        dwork[ij] = dwork[ij] * r[i + j * ldr];
                        ij++;
                    }
                    ij += j + 1;
                }
            } else {
                for (i32 j = 0; j < n; j++) {
                    for (i32 i = 0; i <= j; i++) {
                        dwork[ij] = dwork[ij] * r[i + j * ldr];
                        ij++;
                    }
                    ij += n - j - 1;
                }
            }
        }

        ma02ed(uplow, n, dwork, n);
    }

    f64 temp = xanorm * scale;
    if (temp > est) {
        *ferr = est / temp;
    } else {
        *ferr = one;
    }
}
