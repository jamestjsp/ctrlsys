#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void mb02sd(i32 n, f64 *h, i32 ldh, i32 *ipiv, i32 *info)
{
    i32 j, jp;

    *info = 0;

    if (n < 0) {
        *info = -1;
        return;
    }
    if (ldh < (n > 1 ? n : 1)) {
        *info = -3;
        return;
    }

    if (n == 0) {
        return;
    }

    for (j = 0; j < n; j++) {
        jp = j;
        if (j < n - 1) {
            if (fabs(h[(j + 1) + j * ldh]) > fabs(h[j + j * ldh])) {
                jp = j + 1;
            }
        }
        ipiv[j] = jp + 1;

        if (h[jp + j * ldh] != 0.0) {
            if (jp != j) {
                i32 len = n - j;
                SLC_DSWAP(&len, &h[j + j * ldh], &ldh, &h[jp + j * ldh], &ldh);
            }

            if (j < n - 1) {
                h[(j + 1) + j * ldh] = h[(j + 1) + j * ldh] / h[j + j * ldh];
            }
        } else if (*info == 0) {
            *info = j + 1;
        }

        if (j < n - 1) {
            i32 len = n - j - 1;
            f64 alpha = -h[(j + 1) + j * ldh];
            SLC_DAXPY(&len, &alpha, &h[j + (j + 1) * ldh], &ldh,
                      &h[(j + 1) + (j + 1) * ldh], &ldh);
        }
    }
}
