#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>

void mb02rd(const char *trans, i32 n, i32 nrhs, const f64 *h, i32 ldh,
            const i32 *ipiv, f64 *b, i32 ldb, i32 *info)
{
    i32 j, jp;
    f64 one = 1.0;
    char trans_upper = (char)toupper((unsigned char)trans[0]);
    int notran = (trans_upper == 'N');

    *info = 0;

    if (!notran && trans_upper != 'T' && trans_upper != 'C') {
        *info = -1;
        return;
    }
    if (n < 0) {
        *info = -2;
        return;
    }
    if (nrhs < 0) {
        *info = -3;
        return;
    }
    if (ldh < (n > 1 ? n : 1)) {
        *info = -5;
        return;
    }
    if (ldb < (n > 1 ? n : 1)) {
        *info = -8;
        return;
    }

    if (n == 0 || nrhs == 0) {
        return;
    }

    if (notran) {
        for (j = 0; j < n - 1; j++) {
            jp = ipiv[j] - 1;
            if (jp != j) {
                SLC_DSWAP(&nrhs, &b[jp], &ldb, &b[j], &ldb);
            }
            f64 alpha = -h[(j + 1) + j * ldh];
            SLC_DAXPY(&nrhs, &alpha, &b[j], &ldb, &b[j + 1], &ldb);
        }

        SLC_DTRSM("L", "U", "N", "N", &n, &nrhs, &one, h, &ldh, b, &ldb);
    } else {
        SLC_DTRSM("L", "U", "T", "N", &n, &nrhs, &one, h, &ldh, b, &ldb);

        for (j = n - 2; j >= 0; j--) {
            f64 alpha = -h[(j + 1) + j * ldh];
            SLC_DAXPY(&nrhs, &alpha, &b[j + 1], &ldb, &b[j], &ldb);
            jp = ipiv[j] - 1;
            if (jp != j) {
                SLC_DSWAP(&nrhs, &b[jp], &ldb, &b[j], &ldb);
            }
        }
    }
}
