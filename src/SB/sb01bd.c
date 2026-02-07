/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

static int select_dummy(const f64* reig, const f64* ieig)
{
    (void)reig;
    (void)ieig;
    return 0;
}

void sb01bd(
    const char* dico,
    const i32 n,
    const i32 m,
    const i32 np,
    const f64 alpha,
    f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    f64* wr,
    f64* wi,
    i32* nfp,
    i32* nap,
    i32* nup,
    f64* f,
    const i32 ldf,
    f64* z,
    const i32 ldz,
    const f64 tol,
    f64* dwork,
    const i32 ldwork,
    i32* iwarn,
    i32* info
)
{
    const f64 HUNDR = 100.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 ZERO = 0.0;

    bool discr = (dico[0] == 'D' || dico[0] == 'd');
    *iwarn = 0;
    *info = 0;

    // Parameter validation
    if (!(dico[0] == 'C' || dico[0] == 'c' || dico[0] == 'D' || dico[0] == 'd')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (np < 0) {
        *info = -4;
    } else if (discr && alpha < ZERO) {
        *info = -5;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -9;
    } else if (ldf < (m > 1 ? m : 1)) {
        *info = -16;
    } else if (ldz < (n > 1 ? n : 1)) {
        *info = -18;
    } else {
        i32 minwk = 1;
        if (5*m > minwk) minwk = 5*m;
        if (5*n > minwk) minwk = 5*n;
        if (2*n + 4*m > minwk) minwk = 2*n + 4*m;
        if (ldwork < minwk) {
            *info = -21;
        }
    }

    if (*info != 0) {
        return;
    }

    // Quick return
    if (n == 0) {
        *nfp = 0;
        *nap = 0;
        *nup = 0;
        dwork[0] = ONE;
        return;
    }

    // Compute norms and set tolerances
    i32 n_val = n, m_val = m;
    f64 anorm = SLC_DLANGE("1", &n_val, &n_val, a, &lda, dwork);
    f64 bnorm = SLC_DLANGE("1", &n_val, &m_val, b, &ldb, dwork);

    f64 toler, tolerb;
    if (tol <= ZERO) {
        f64 eps = SLC_DLAMCH("Epsilon");
        f64 maxab = anorm > bnorm ? anorm : bnorm;
        toler = (f64)n * maxab * eps;
        tolerb = (f64)n * bnorm * eps;
    } else {
        toler = tol;
        tolerb = tol;
    }

    // Workspace allocation
    i32 kwr = 0;
    i32 kwi = kwr + n;
    i32 kw = kwi + n;

    // Reduce A to real Schur form
    i32 ncur, sdim;
    i32 bwork[1];
    i32 ldwork_dgees = ldwork - kw;
    SLC_DGEES("V", "N", select_dummy, &n_val, a, &lda, &sdim,
              &dwork[kwr], &dwork[kwi], z, &ldz, &dwork[kw],
              &ldwork_dgees, bwork, info);

    i32 wrkopt = kw + (i32)dwork[kw];
    if (*info != 0) {
        *info = 1;
        return;
    }

    // Order Schur form to separate "good" eigenvalues
    mb03qd(dico, "Stable", "Update", n, 1, n, alpha, a, lda, z, ldz, nfp, dwork, info);
    if (*info != 0) {
        return;
    }

    // Set F = 0
    SLC_DLASET("Full", &m_val, &n_val, &ZERO, &ZERO, f, &ldf);

    // Return if B is negligible
    if (bnorm <= tolerb) {
        *nap = 0;
        *nup = n;
        dwork[0] = (f64)wrkopt;
        return;
    }

    // Compute bound for numerical stability
    f64 rmax = HUNDR * anorm / bnorm;

    // Perform eigenvalue assignment
    *nap = 0;
    *nup = 0;

    i32 npr = 0;
    i32 npc = 0;
    i32 ipc = 0;

    if (*nfp < n) {
        i32 kg = 0;
        i32 kfi = kg + 2*m;
        i32 kw2 = kfi + 2*m;

        i32 nlow = *nfp;
        i32 nsup = n - 1;

        // Separate real and complex eigenvalues
        for (i32 i = 0; i < np; i++) {
            if (wi[i] == ZERO) {
                npr++;
                i32 k = i - npr + 1;
                if (k > 0) {
                    f64 s_temp = wr[i];
                    for (i32 j = npr + k - 2; j >= npr - 1; j--) {
                        wr[j + 1] = wr[j];
                        wi[j + 1] = wi[j];
                    }
                    wr[npr - 1] = s_temp;
                    wi[npr - 1] = ZERO;
                }
            }
        }
        npc = np - npr;
        ipc = npr;

        // Main loop
        while (nlow <= nsup && *info == 0) {
            // Determine dimension of last block
            i32 ib = 1;
            if (nlow < nsup) {
                if (a[nsup + (nsup - 1) * lda] != ZERO) ib = 2;
            }

            // Compute G = last ib rows of Z'*B
            i32 nl = nsup - ib + 1;
            f64 alpha_one = ONE, beta_zero = ZERO;
            SLC_DGEMM("T", "N", &ib, &m_val, &n_val, &alpha_one,
                      &z[nl * ldz], &ldz, b, &ldb, &beta_zero, &dwork[kg], &ib);

            // Check controllability
            f64 gnorm = SLC_DLANGE("1", &ib, &m_val, &dwork[kg], &ib, &dwork[kw2]);
            if (gnorm <= tolerb) {
                nsup -= ib;
                *nup += ib;
                continue;
            }

            // Test for termination
            if (*nap == np) {
                *info = 3;
                continue;
            }

            if (ib == 1 && npr == 0 && nlow == nsup) {
                *info = 4;
                continue;
            }

            bool simplb = true;

            // Form 2x2 block if necessary
            if ((ib == 1 && npr == 0) ||
                (ib == 1 && npr == 1 && nsup > nlow && npr + npc > nsup - nlow + 1)) {
                if (nsup > 1) {
                    if (a[nsup - 1 + (nsup - 2) * lda] != ZERO) {
                        // Interchange with adjacent 2x2 block
                        i32 j1 = nsup - 1;
                        i32 n1 = 2;
                        i32 n2 = 1;
                        i32 wantq = 1;
                        SLC_DLAEXC(&wantq, &n_val, a, &lda, z, &ldz,
                                   &j1, &n1, &n2, &dwork[kw2], info);
                        if (*info != 0) {
                            *info = 2;
                            return;
                        }
                    } else {
                        simplb = false;
                    }
                } else {
                    simplb = false;
                }
                ib = 2;
                nl = nsup - ib + 1;

                // Recompute G
                SLC_DGEMM("T", "N", &ib, &m_val, &n_val, &alpha_one,
                          &z[nl * ldz], &ldz, b, &ldb, &beta_zero, &dwork[kg], &ib);

                gnorm = SLC_DLANGE("1", &ib, &m_val, &dwork[kg], &ib, &dwork[kw2]);
                if (gnorm <= tolerb) {
                    nsup -= ib;
                    *nup += ib;
                    continue;
                }
            }

            if (*nap + ib > np) {
                *info = 3;
                continue;
            }

            f64 s_val, p_val, x_val, y_val;
            bool ceig;

            if (ib == 1) {
                // 1x1 block
                x_val = a[nsup + nsup * lda];
                sb01bx(true, npr, x_val, x_val, wr, &x_val, &s_val, &p_val);
                npr--;
                ceig = false;
            } else {
                if (simplb) {
                    // Simple 2x2 block
                    // MB03QY expects 1-based l parameter, nl is 0-based column index
                    mb03qy(n, nl + 1, a, lda, z, ldz, &x_val, &y_val, info);
                    if (npc > 1) {
                        sb01bx(false, npc, x_val, y_val, &wr[ipc], &wi[ipc], &s_val, &p_val);
                        npc -= 2;
                        ceig = true;
                    } else {
                        // Choose two nearest real eigenvalues
                        sb01bx(true, npr, x_val, x_val, wr, &x_val, &s_val, &p_val);
                        sb01bx(true, npr - 1, x_val, x_val, wr, &x_val, &y_val, &p_val);
                        p_val = s_val * y_val;
                        s_val = s_val + y_val;
                        npr -= 2;
                        ceig = false;
                    }
                } else {
                    // Non-simple 2x2 block
                    x_val = (a[nl + nl * lda] + a[nsup + nsup * lda]) / TWO;
                    sb01bx(false, npc, x_val, ZERO, &wr[ipc], &wi[ipc], &s_val, &p_val);
                    npc -= 2;
                }
            }

            // Form 2x2 matrix A2
            f64 a2[4];
            a2[0] = a[nl + nl * lda];
            if (ib > 1) {
                a2[1] = a[nsup + nl * lda];
                a2[2] = a[nl + nsup * lda];
                a2[3] = a[nsup + nsup * lda];
            }

            // Compute feedback Fi
            i32 ierr;
            sb01by(ib, m, s_val, p_val, a2, &dwork[kg], &dwork[kfi], toler, &dwork[kw2], &ierr);

            if (ierr != 0) {
                if (ib == 1 || simplb) {
                    nsup -= ib;
                    if (ceig) {
                        npc += ib;
                    } else {
                        npr += ib;
                    }
                    *nup += ib;
                } else {
                    // Non-simple 2x2 block uncontrollable
                    f64 c_rot = dwork[kfi];
                    f64 s_rot = dwork[kfi + ib];

                    // Apply rotation
                    i32 len = n - nl;
                    SLC_DROT(&len, &a[nl + nl * lda], &lda, &a[nsup + nl * lda], &lda, &c_rot, &s_rot);
                    SLC_DROT(&n_val, &a[nl * lda], &(i32){1}, &a[nsup * lda], &(i32){1}, &c_rot, &s_rot);
                    SLC_DROT(&n_val, &z[nl * ldz], &(i32){1}, &z[nsup * ldz], &(i32){1}, &c_rot, &s_rot);

                    a[nsup + nl * lda] = ZERO;
                    nsup = nl;
                    *nup += 1;
                    npc += 2;
                }
            } else {
                // Successful assignment
                // Update F <- F + [0 Fi] * Z'
                SLC_DGEMM("N", "T", &m_val, &n_val, &ib, &alpha_one,
                          &dwork[kfi], &m_val, &z[nl * ldz], &ldz,
                          &alpha_one, f, &ldf);

                // Check numerical stability
                f64 fnorm = SLC_DLANGE("1", &m_val, &ib, &dwork[kfi], &m_val, &dwork[kw2]);
                if (fnorm > rmax) (*iwarn)++;

                // Update A <- A + Z'*B*[0 Fi]
                SLC_DGEMM("N", "N", &n_val, &ib, &m_val, &alpha_one,
                          b, &ldb, &dwork[kfi], &m_val, &beta_zero,
                          &dwork[kw2], &n_val);
                i32 nsup1 = nsup + 1;
                SLC_DGEMM("T", "N", &nsup1, &ib, &n_val, &alpha_one,
                          z, &ldz, &dwork[kw2], &n_val,
                          &alpha_one, &a[nl * lda], &lda);

                // Try to split 2x2 block
                if (ib == 2) {
                    // MB03QY expects 1-based l parameter
                    mb03qy(n, nl + 1, a, lda, z, ldz, &x_val, &y_val, info);
                }

                *nap += ib;

                if (nlow + ib <= nsup) {
                    // Move blocks to leading position
                    i32 ncur1 = nsup - ib;
                    i32 nmoves = 1;
                    if (ib == 2 && a[nsup + (nsup - 1) * lda] == ZERO) {
                        ib = 1;
                        nmoves = 2;
                    }

                    while (nmoves > 0) {
                        ncur = ncur1;
                        while (ncur >= nlow) {
                            i32 ib1 = 1;
                            if (ncur > nlow) {
                                if (a[ncur + (ncur - 1) * lda] != ZERO) ib1 = 2;
                            }
                            i32 j1_idx = ncur - ib1 + 2;  // +2: ncur is 0-based, J1 is 1-based
                            i32 wantq = 1;
                            SLC_DLAEXC(&wantq, &n_val, a, &lda, z, &ldz,
                                       &j1_idx, &ib1, &ib, &dwork[kw2], info);
                            if (*info != 0) {
                                *info = 2;
                                return;
                            }
                            ncur -= ib1;
                        }
                        nmoves--;
                        ncur1++;
                        nlow += ib;
                    }
                } else {
                    nlow += ib;
                }
            }
        }

        i32 max_wk = 5*m;
        if (2*n + 4*m > max_wk) max_wk = 2*n + 4*m;
        if (wrkopt < max_wk) wrkopt = max_wk;
    }

    // Annihilate elements below first subdiagonal
    if (n > 2) {
        i32 nm2 = n - 2;
        SLC_DLASET("L", &nm2, &nm2, &ZERO, &ZERO, &a[2], &lda);
    }

    // Reorder assigned eigenvalues
    if (*nap > 0) {
        // k = IPC - NPR - 1 in Fortran, but IPC is 1-based = ipc+1 in 0-based
        // So k = (ipc + 1) - npr - 1 = ipc - npr
        i32 k = ipc - npr;
        if (k > 0) {
            if (k <= npr) {
                i32 one = 1;
                SLC_DSWAP(&k, &wr[npr], &one, wr, &one);
            } else {
                SLC_DCOPY(&k, &wr[npr], &(i32){1}, dwork, &(i32){1});
                SLC_DCOPY(&npr, wr, &(i32){1}, &dwork[k], &(i32){1});
                i32 kpr = k + npr;
                SLC_DCOPY(&kpr, dwork, &(i32){1}, wr, &(i32){1});
            }
        }
        i32 j = *nap - k;
        if (j > 0) {
            i32 one = 1;
            SLC_DSWAP(&j, &wr[ipc + npc], &one, &wr[k], &one);
            SLC_DSWAP(&j, &wi[ipc + npc], &one, &wi[k], &one);
        }
    }

    dwork[0] = (f64)wrkopt;
}
