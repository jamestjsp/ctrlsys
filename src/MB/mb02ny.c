/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot/mb02.h"
#include "slicot_blas.h"

void mb02ny(const bool updatu, const bool updatv, const i32 m, const i32 n,
            const i32 i_idx, const i32 k, f64* q, f64* e, f64* u, const i32 ldu,
            f64* v, const i32 ldv, f64* dwork)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    if (m <= 0 || n <= 0) {
        return;
    }

    i32 p = (m < n) ? m : n;

    if (i_idx <= p) {
        q[i_idx - 1] = ZERO;  // 1-based to 0-based conversion
    }

    // Annihilate E(i) if i < k
    if (i_idx < k) {
        f64 c = ZERO;
        f64 s = ONE;
        i32 irot = 0;
        i32 nrot = k - i_idx;

        for (i32 l = i_idx; l <= k - 1; l++) {
            // l is 1-based Fortran index
            f64 g = e[l - 1];          // E(L) in Fortran
            e[l - 1] = c * g;
            f64 r;
            f64 sg = s * g;
            SLC_DLARTG(&q[l], &sg, &c, &s, &r);  // Q(L+1) in Fortran = q[l] in C
            q[l] = r;
            if (updatu) {
                irot++;
                dwork[irot - 1] = c;           // DWORK(IROT)
                dwork[irot - 1 + nrot] = s;    // DWORK(IROT+NROT)
            }
        }

        if (updatu) {
            // DLASR: Apply stored rotations from the right
            // CALL DLASR('Right', 'Top', 'Forward', M, NROT+1, DWORK(1), DWORK(NROT+1), U(1,I), LDU)
            i32 nrot_plus_1 = nrot + 1;
            SLC_DLASR("R", "T", "F", &m, &nrot_plus_1, dwork, &dwork[nrot],
                      &u[(i_idx - 1) * ldu], &ldu);
        }
    }

    // Annihilate E(i-1) if i > 1
    if (i_idx > 1) {
        i32 i1 = i_idx - 1;
        f64 f = e[i1 - 1];   // E(I1) = E(i-1) in Fortran
        e[i1 - 1] = ZERO;

        for (i32 l1 = 1; l1 <= i1 - 1; l1++) {
            i32 l = i_idx - l1;   // L = I - L1 (1-based Fortran index)
            f64 c, s, r;
            SLC_DLARTG(&q[l - 1], &f, &c, &s, &r);  // Q(L) in Fortran
            q[l - 1] = r;
            if (updatv) {
                dwork[l - 1] = c;          // DWORK(L)
                dwork[l - 1 + i1] = s;     // DWORK(L + I1)
            }
            f64 g = e[l - 2];     // E(L-1) in Fortran
            f = -s * g;
            e[l - 2] = c * g;
        }

        f64 c, s, r;
        SLC_DLARTG(&q[0], &f, &c, &s, &r);   // Q(1) in Fortran
        q[0] = r;
        if (updatv) {
            dwork[0] = c;       // DWORK(1)
            dwork[i1] = s;      // DWORK(I)
            // CALL DLASR('Right', 'Bottom', 'Backward', N, I, DWORK(1), DWORK(I), V(1,1), LDV)
            SLC_DLASR("R", "B", "B", &n, &i_idx, dwork, &dwork[i1], v, &ldv);
        }
    }
}
