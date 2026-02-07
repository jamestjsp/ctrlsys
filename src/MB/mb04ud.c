#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>

void mb04ud(const char *jobq_str, const char *jobz_str, const i32 m, const i32 n,
            f64 *a, const i32 lda, f64 *e, const i32 lde,
            f64 *q, const i32 ldq, f64 *z, const i32 ldz,
            i32 *ranke, i32 *istair, const f64 tol, f64 *dwork, i32 *info)
{
    char jobq = jobq_str[0];
    char jobz = jobz_str[0];

    bool ljobqi = (jobq == 'I' || jobq == 'i');
    bool ljobzi = (jobz == 'I' || jobz == 'i');
    bool updatq = ljobqi || (jobq == 'U' || jobq == 'u');
    bool updatz = ljobzi || (jobz == 'U' || jobz == 'u');

    *info = 0;

    if (!updatq && jobq != 'N' && jobq != 'n') {
        *info = -1;
    } else if (!updatz && jobz != 'N' && jobz != 'n') {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -6;
    } else if (lde < (m > 1 ? m : 1)) {
        *info = -8;
    } else if ((!updatq && ldq < 1) || (updatq && ldq < (m > 1 ? m : 1))) {
        *info = -10;
    } else if ((!updatz && ldz < 1) || (updatz && ldz < (n > 1 ? n : 1))) {
        *info = -12;
    }

    if (*info != 0) {
        return;
    }

    f64 zero = 0.0;
    f64 one = 1.0;
    i32 int1 = 1;

    if (ljobqi) {
        SLC_DLASET("Full", &m, &m, &zero, &one, q, &ldq);
    }
    if (ljobzi) {
        SLC_DLASET("Full", &n, &n, &zero, &one, z, &ldz);
    }

    *ranke = (m < n) ? m : n;

    if (*ranke == 0) {
        return;
    }

    f64 toler = tol;
    if (toler <= 0.0) {
        f64 eps = SLC_DLAMCH("Epsilon");
        f64 enorm = SLC_DLANGE("M", &m, &n, e, &lde, dwork);
        toler = eps * enorm;
    }

    i32 k = n;  // 1-based column index
    bool lzero = false;

    while (k > 0 && !lzero) {
        i32 mnk = m - n + k;  // 1-based row index (diagonal element row)

        f64 emxnrm = 0.0;
        i32 lk = mnk;  // 1-based row with max norm

        for (i32 l = mnk; l >= 1; l--) {
            i32 l_idx = l - 1;  // 0-based
            i32 idmax = SLC_IDAMAX(&k, &e[l_idx], &lde);  // returns 1-based
            if (idmax >= 1 && idmax <= k) {
                i32 j_idx = idmax - 1;  // 0-based column
                f64 emx = fabs(e[l_idx + j_idx * lde]);
                if (emx > emxnrm) {
                    emxnrm = emx;
                    lk = l;  // 1-based
                }
            }
        }

        if (emxnrm <= toler) {
            SLC_DLASET("Full", &mnk, &k, &zero, &zero, e, &lde);
            lzero = true;
            *ranke = n - k;
        } else {
            i32 lk_idx = lk - 1;      // 0-based
            i32 mnk_idx = mnk - 1;    // 0-based

            if (lk != mnk) {
                SLC_DSWAP(&n, &e[lk_idx], &lde, &e[mnk_idx], &lde);
                SLC_DSWAP(&n, &a[lk_idx], &lda, &a[mnk_idx], &lda);
                if (updatq) {
                    SLC_DSWAP(&m, &q[lk_idx * ldq], &int1, &q[mnk_idx * ldq], &int1);
                }
            }

            i32 k_idx = k - 1;  // 0-based column index
            f64 tau;

            SLC_DLARFG(&k, &e[mnk_idx + k_idx * lde], &e[mnk_idx], &lde, &tau);
            f64 emx = e[mnk_idx + k_idx * lde];
            e[mnk_idx + k_idx * lde] = 1.0;

            if (mnk > 1) {
                i32 mnk_m1 = mnk - 1;
                SLC_DLARF("Right", &mnk_m1, &k, &e[mnk_idx], &lde, &tau, e, &lde, dwork);
            }
            SLC_DLARF("Right", &m, &k, &e[mnk_idx], &lde, &tau, a, &lda, dwork);
            if (updatz) {
                SLC_DLARF("Right", &n, &k, &e[mnk_idx], &lde, &tau, z, &ldz, dwork);
            }

            e[mnk_idx + k_idx * lde] = emx;

            if (k > 1) {
                i32 km1 = k - 1;
                i32 int_one = 1;
                SLC_DLASET("Full", &int_one, &km1, &zero, &zero, &e[mnk_idx], &lde);
            }

            k--;
        }
    }

    for (i32 i = 0; i < *ranke; i++) {
        istair[m - 1 - i] = n - i;  // Positive = corner point (1-based column index)
    }

    i32 nr1 = -(n - *ranke + 1);
    for (i32 i = 0; i < m - *ranke; i++) {
        istair[i] = nr1;  // Negative = boundary but not corner
    }
}
