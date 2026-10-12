// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void ma01ad(f64 xr, f64 xi, f64 *yr, f64 *yi) {
    const f64 HALF = 0.5;

    f64 s = sqrt(HALF * (SLC_DLAPY2(&xr, &xi) + fabs(xr)));

    if (xr >= 0.0) {
        *yr = s;
    }
    if (xi < 0.0) {
        s = -s;
    }
    if (xr <= 0.0) {
        *yi = s;
        if (xr < 0.0) {
            *yr = HALF * (xi / s);
        }
    } else {
        *yi = HALF * (xi / (*yr));
    }
}
