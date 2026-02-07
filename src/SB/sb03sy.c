// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <stdbool.h>

void sb03sy(const char* job, const char* trana, const char* lyapun, i32 n,
            const f64* t, i32 ldt, const f64* u, i32 ldu, const f64* xa, i32 ldxa,
            f64* sepd, f64* thnorm, i32* iwork, f64* dwork, i32 ldwork, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 half = 0.5;
    i32 nn = n * n;

    char job_c = (char)toupper((unsigned char)job[0]);
    char trana_c = (char)toupper((unsigned char)trana[0]);
    char lyapun_c = (char)toupper((unsigned char)lyapun[0]);

    bool wants = (job_c == 'S');
    bool wantt = (job_c == 'T');
    bool notrna = (trana_c == 'N');
    bool update = (lyapun_c == 'O');

    *info = 0;

    if (!wants && !wantt && job_c != 'B') {
        *info = -1;
    } else if (!notrna && trana_c != 'T' && trana_c != 'C') {
        *info = -2;
    } else if (!update && lyapun_c != 'R') {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (ldt < (n > 1 ? n : 1)) {
        *info = -6;
    } else if (ldu < 1 || (update && ldu < n)) {
        *info = -8;
    } else if (ldxa < 1 || (!wants && ldxa < n)) {
        *info = -10;
    } else if (ldwork < 0 || (n > 0 && ldwork < (2 * nn > 3 ? 2 * nn : 3))) {
        *info = -15;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        return;
    }

    i32 itmp = nn;
    char tranat = notrna ? 'T' : 'N';

    f64 est = zero;
    f64 scale = one;
    f64 bignum = one / SLC_DLAMCH("S");

    if (!wantt) {
        i32 kase = 0;
        i32 isave[3] = {0, 0, 0};
        est = zero;

        while (1) {
            SLC_DLACN2(&nn, &dwork[itmp], dwork, iwork, &est, &kase, isave);
            if (kase == 0) break;

            f64 upper_norm = SLC_DLANSY("1", "Upper", &n, dwork, &n, &dwork[itmp]);
            f64 lower_norm = SLC_DLANSY("1", "Lower", &n, dwork, &n, &dwork[itmp]);

            char uplo;
            if (upper_norm >= lower_norm) {
                uplo = 'U';
            } else {
                uplo = 'L';
            }

            if (update) {
                i32 info2;
                mb01ru(&uplo, "T", n, n, zero, one, dwork, n, u, ldu,
                       dwork, n, &dwork[itmp], nn, &info2);
                i32 n1 = n + 1;
                SLC_DSCAL(&n, &half, dwork, &n1);
            }
            ma02ed(uplo, n, dwork, n);

            i32 info2 = 0;
            if (kase == 1) {
                sb03mx(&trana_c, n, t, ldt, dwork, n, &scale, &dwork[itmp], &info2);
            } else {
                sb03mx(&tranat, n, t, ldt, dwork, n, &scale, &dwork[itmp], &info2);
            }

            if (info2 > 0) {
                *info = n + 1;
            }

            if (update) {
                i32 info3;
                mb01ru(&uplo, "N", n, n, zero, one, dwork, n, u, ldu,
                       dwork, n, &dwork[itmp], nn, &info3);
                i32 n1 = n + 1;
                SLC_DSCAL(&n, &half, dwork, &n1);

                ma02ed(uplo, n, dwork, n);
            }
        }

        if (est > scale) {
            *sepd = scale / est;
        } else {
            if (scale < est * bignum) {
                *sepd = scale / est;
            } else {
                *sepd = bignum;
            }
        }

        if (*sepd == zero) {
            return;
        }
    }

    if (!wants) {
        i32 kase = 0;
        i32 isave[3] = {0, 0, 0};
        est = zero;

        while (1) {
            SLC_DLACN2(&nn, &dwork[itmp], dwork, iwork, &est, &kase, isave);
            if (kase == 0) break;

            f64 upper_norm = SLC_DLANSY("1", "Upper", &n, dwork, &n, &dwork[itmp]);
            f64 lower_norm = SLC_DLANSY("1", "Lower", &n, dwork, &n, &dwork[itmp]);

            char uplo;
            if (upper_norm >= lower_norm) {
                uplo = 'U';
            } else {
                uplo = 'L';
            }

            ma02ed(uplo, n, dwork, n);

            SLC_DSYR2K(&uplo, &tranat, &n, &n, &one, dwork, &n, xa, &ldxa,
                       &zero, &dwork[itmp], &n);
            SLC_DLACPY(&uplo, &n, &n, &dwork[itmp], &n, dwork, &n);

            if (update) {
                i32 info2;
                mb01ru(&uplo, "T", n, n, zero, one, dwork, n, u, ldu,
                       dwork, n, &dwork[itmp], nn, &info2);
                i32 n1 = n + 1;
                SLC_DSCAL(&n, &half, dwork, &n1);
            }
            ma02ed(uplo, n, dwork, n);

            i32 info2 = 0;
            if (kase == 1) {
                sb03mx(&trana_c, n, t, ldt, dwork, n, &scale, &dwork[itmp], &info2);
            } else {
                sb03mx(&tranat, n, t, ldt, dwork, n, &scale, &dwork[itmp], &info2);
            }

            if (info2 > 0) {
                *info = n + 1;
            }

            if (update) {
                i32 info3;
                mb01ru(&uplo, "N", n, n, zero, one, dwork, n, u, ldu,
                       dwork, n, &dwork[itmp], nn, &info3);
                i32 n1 = n + 1;
                SLC_DSCAL(&n, &half, dwork, &n1);

                ma02ed(uplo, n, dwork, n);
            }
        }

        if (est < scale) {
            *thnorm = est / scale;
        } else {
            if (est < scale * bignum) {
                *thnorm = est / scale;
            } else {
                *thnorm = bignum;
            }
        }
    }
}
