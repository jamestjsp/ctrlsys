// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void ab09iy(
    const char* dico,
    const char* jobc,
    const char* jobo,
    const char* weight,
    const i32 n,
    const i32 m,
    const i32 p,
    const i32 nv,
    const i32 pv,
    const i32 nw,
    const i32 mw,
    const f64 alphac,
    const f64 alphao,
    const f64* a, const i32 lda,
    const f64* b, const i32 ldb,
    const f64* c, const i32 ldc,
    const f64* av, const i32 ldav,
    const f64* bv, const i32 ldbv,
    const f64* cv, const i32 ldcv,
    const f64* dv, const i32 lddv,
    const f64* aw, const i32 ldaw,
    const f64* bw, const i32 ldbw,
    const f64* cw, const i32 ldcw,
    const f64* dw, const i32 lddw,
    f64* scalec,
    f64* scaleo,
    f64* s, const i32 lds,
    f64* r, const i32 ldr,
    f64* dwork,
    const i32 ldwork,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 NEGONE = -1.0;

    bool discr = (dico[0] == 'D' || dico[0] == 'd');
    bool leftw = (weight[0] == 'L' || weight[0] == 'l' ||
                  weight[0] == 'B' || weight[0] == 'b');
    bool rightw = (weight[0] == 'R' || weight[0] == 'r' ||
                   weight[0] == 'B' || weight[0] == 'b');
    bool frwght = leftw || rightw;

    *info = 0;
    i32 lw = 1;
    i32 nnv = n + nv;
    i32 nnw = n + nw;

    if (leftw && pv > 0) {
        i32 max_nnv_pv = nnv > pv ? nnv : pv;
        i32 lwl = nnv * (nnv + max_nnv_pv + 5);
        lw = lw > lwl ? lw : lwl;
    } else {
        i32 lwl = n * (p + 5);
        lw = lw > lwl ? lw : lwl;
    }
    if (rightw && mw > 0) {
        i32 max_nnw_mw = nnw > mw ? nnw : mw;
        i32 lwr = nnw * (nnw + max_nnw_mw + 5);
        lw = lw > lwr ? lw : lwr;
    } else {
        i32 lwr = n * (m + 5);
        lw = lw > lwr ? lw : lwr;
    }

    if (!(dico[0] == 'C' || dico[0] == 'c' || discr)) {
        *info = -1;
    } else if (!(jobc[0] == 'S' || jobc[0] == 's' ||
                 jobc[0] == 'E' || jobc[0] == 'e')) {
        *info = -2;
    } else if (!(jobo[0] == 'S' || jobo[0] == 's' ||
                 jobo[0] == 'E' || jobo[0] == 'e')) {
        *info = -3;
    } else if (!(frwght || weight[0] == 'N' || weight[0] == 'n')) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (m < 0) {
        *info = -6;
    } else if (p < 0) {
        *info = -7;
    } else if (nv < 0) {
        *info = -8;
    } else if (pv < 0) {
        *info = -9;
    } else if (nw < 0) {
        *info = -10;
    } else if (mw < 0) {
        *info = -11;
    } else if (fabs(alphac) > ONE) {
        *info = -12;
    } else if (fabs(alphao) > ONE) {
        *info = -13;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -15;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -17;
    } else if (ldc < (p > 1 ? p : 1)) {
        *info = -19;
    } else if (ldav < 1 || (leftw && ldav < nv)) {
        *info = -21;
    } else if (ldbv < 1 || (leftw && ldbv < nv)) {
        *info = -23;
    } else if (ldcv < 1 || (leftw && ldcv < pv)) {
        *info = -25;
    } else if (lddv < 1 || (leftw && lddv < pv)) {
        *info = -27;
    } else if (ldaw < 1 || (rightw && ldaw < nw)) {
        *info = -29;
    } else if (ldbw < 1 || (rightw && ldbw < nw)) {
        *info = -31;
    } else if (ldcw < 1 || (rightw && ldcw < m)) {
        *info = -33;
    } else if (lddw < 1 || (rightw && lddw < m)) {
        *info = -35;
    } else if (lds < (n > 1 ? n : 1)) {
        *info = -39;
    } else if (ldr < (n > 1 ? n : 1)) {
        *info = -41;
    } else if (ldwork < lw) {
        *info = -43;
    }

    if (*info != 0) {
        return;
    }

    *scalec = ONE;
    *scaleo = ONE;

    i32 min_nmp = n < m ? n : m;
    min_nmp = min_nmp < p ? min_nmp : p;
    if (min_nmp == 0) {
        dwork[0] = ONE;
        return;
    }

    f64 work = 1.0;
    i32 ierr;
    f64 t;
    f64 dum[1];
    i32 int1 = 1;

    if (leftw && pv > 0) {
        i32 kaw = 0;
        i32 ku = kaw + nnv * nnv;
        i32 ldu = nnv > pv ? nnv : pv;

        SLC_DLACPY("Full", &nv, &nv, av, &ldav, &dwork[kaw], &nnv);
        SLC_DLASET("Full", &n, &nv, &ZERO, &ZERO, &dwork[kaw + nv], &nnv);
        SLC_DGEMM("N", "N", &nv, &n, &p, &ONE,
                  bv, &ldbv, c, &ldc, &ZERO, &dwork[kaw + nnv * nv], &nnv);
        SLC_DLACPY("Full", &n, &n, a, &lda, &dwork[kaw + nnv * nv + nv], &nnv);

        SLC_DLACPY("Full", &pv, &nv, cv, &ldcv, &dwork[ku], &ldu);
        SLC_DGEMM("N", "N", &pv, &n, &p, &ONE,
                  dv, &lddv, c, &ldc, &ZERO, &dwork[ku + ldu * nv], &ldu);

        i32 ktau = ku + ldu * nnv;
        i32 kw = ktau + nnv;
        i32 ldwork_rem = ldwork - kw;

        sb03ou(discr, false, nnv, pv, &dwork[kaw], nnv,
               &dwork[ku], ldu, &dwork[ktau], &dwork[ku], ldu,
               scaleo, &dwork[kw], ldwork_rem, &ierr);

        if (ierr != 0) {
            *info = 1;
            return;
        }
        work = work > (dwork[kw] + (f64)(kw)) ? work : (dwork[kw] + (f64)(kw));

        kw = ku + ldu * nv + nv;
        SLC_DLACPY("Upper", &n, &n, &dwork[kw], &ldu, r, &ldr);

        if (alphao != ZERO) {
            t = sqrt(ONE - alphao * alphao);
            for (i32 j = ku + ldu * nv; j < ku + ldu * (nnv); j += ldu) {
                SLC_DSCAL(&nv, &t, &dwork[j], &int1);
            }
        }

        if (alphao < ONE && nv > 0) {
            i32 ktau2 = 0;
            i32 kw2 = ktau2 + n;
            mb04od("Full", n, 0, nv, r, ldr, &dwork[ku + ldu * nv],
                   ldu, dum, 1, dum, 1, &dwork[ktau2], &dwork[kw2]);

            for (i32 j = 0; j < n; j++) {
                dwork[j] = r[j + j * ldr];
                for (i32 i = 0; i <= j; i++) {
                    if (dwork[i] < ZERO) r[i + j * ldr] = -r[i + j * ldr];
                }
            }
        }

        if ((jobo[0] == 'E' || jobo[0] == 'e') && alphao < ONE) {
            // Form Y = -A'*(R'*R)-(R'*R)*A if DICO = 'C', or
            //      Y = -A'*(R'*R)*A+(R'*R) if DICO = 'D'.
            SLC_DLACPY("Upper", &n, &n, r, &ldr, &dwork[ku], &n);
            mb01wd(dico, "U", "N", "H", n, NEGONE, ZERO, r, ldr,
                   (f64*)&dwork[kaw + nnv * nv + nv], nnv, &dwork[ku], n, &ierr);

            // Eigendecomposition of Y: Y = Z*Sigma*Z'
            ku = n;
            SLC_DSYEV("V", "U", &n, r, &ldr, dwork, &dwork[ku],
                      &(i32){ldwork - n}, &ierr);
            if (ierr > 0) {
                *info = 3;
                return;
            }
            work = work > (dwork[ku] + (f64)n) ? work : (dwork[ku] + (f64)n);

            // Partition Sigma = (Sigma1, Sigma2) with Sigma1 <= 0, Sigma2 > 0
            f64 abs_d1 = fabs(dwork[0]);
            f64 abs_dn = fabs(dwork[n - 1]);
            f64 tol = (abs_d1 > abs_dn ? abs_d1 : abs_dn) * SLC_DLAMCH("E");

            // Form Cbar = [ sqrt(Sigma2)*Z2' ]
            i32 pcbar = 0;
            for (i32 j = 0; j < n; j++) {
                if (dwork[j] > tol) {
                    f64 sqval = sqrt(dwork[j]);
                    SLC_DSCAL(&n, &sqval, &r[j * ldr], &int1);
                    SLC_DCOPY(&n, &r[j * ldr], &int1, &dwork[ku + pcbar], &n);
                    pcbar++;
                }
            }

            // Solve Lyapunov for Cholesky factor R of Q = R'*R
            ktau = ku + n * n;
            kw = ktau + n;

            sb03ou(discr, false, n, pcbar, a, lda, &dwork[ku], n,
                   &dwork[ktau], r, ldr, &t, &dwork[kw], ldwork - kw, &ierr);
            if (ierr != 0) {
                *info = 1;
                return;
            }
            *scaleo *= t;
            work = work > (dwork[kw] + (f64)(kw)) ? work : (dwork[kw] + (f64)(kw));
        }

    } else {
        i32 ku = 0;
        i32 ktau = ku + p * n;
        i32 kw = ktau + n;
        i32 ldwork_rem = ldwork - kw;

        SLC_DLACPY("Full", &p, &n, c, &ldc, &dwork[ku], &p);
        sb03ou(discr, false, n, p, a, lda, &dwork[ku], p,
               &dwork[ktau], r, ldr, scaleo, &dwork[kw], ldwork_rem, &ierr);
        if (ierr != 0) {
            *info = 1;
            return;
        }
        work = work > (dwork[kw] + (f64)(kw)) ? work : (dwork[kw] + (f64)(kw));
    }

    if (rightw && mw > 0) {
        i32 kaw = 0;
        i32 ku = kaw + nnw * nnw;

        SLC_DLACPY("Full", &n, &n, a, &lda, &dwork[kaw], &nnw);
        SLC_DLASET("Full", &nw, &n, &ZERO, &ZERO, &dwork[kaw + n], &nnw);
        SLC_DGEMM("N", "N", &n, &nw, &m, &ONE,
                  b, &ldb, cw, &ldcw, &ZERO, &dwork[kaw + nnw * n], &nnw);
        SLC_DLACPY("Full", &nw, &nw, aw, &ldaw, &dwork[kaw + nnw * n + n], &nnw);

        SLC_DGEMM("N", "N", &n, &mw, &m, &ONE,
                  b, &ldb, dw, &lddw, &ZERO, &dwork[ku], &nnw);
        SLC_DLACPY("Full", &nw, &mw, bw, &ldbw, &dwork[ku + n], &nnw);

        i32 max_nnw_mw = nnw > mw ? nnw : mw;
        i32 ktau = ku + nnw * max_nnw_mw;
        i32 kw = ktau + nnw;
        i32 ldwork_rem = ldwork - kw;

        sb03ou(discr, true, nnw, mw, &dwork[kaw], nnw,
               &dwork[ku], nnw, &dwork[ktau], &dwork[ku], nnw,
               scalec, &dwork[kw], ldwork_rem, &ierr);

        if (ierr != 0) {
            *info = 2;
            return;
        }
        work = work > (dwork[kw] + (f64)(kw)) ? work : (dwork[kw] + (f64)(kw));

        SLC_DLACPY("Upper", &n, &n, &dwork[ku], &nnw, s, &lds);

        if (alphac != ZERO) {
            t = sqrt(ONE - alphac * alphac);
            for (i32 j = ku + nnw * n; j < ku + nnw * (nnw); j += nnw) {
                SLC_DSCAL(&n, &t, &dwork[j], &int1);
            }
        }

        if (alphac < ONE && nw > 0) {
            i32 ktau2 = n * nnw;
            i32 kw2 = ktau2 + n;
            SLC_MB04ND("Full", n, 0, nw, s, lds, &dwork[ku + nnw * n],
                       nnw, dum, 1, dum, 1, &dwork[ktau2], &dwork[kw2]);

            for (i32 j = 0; j < n; j++) {
                if (s[j + j * lds] < ZERO) {
                    for (i32 i = 0; i <= j; i++) {
                        s[i + j * lds] = -s[i + j * lds];
                    }
                }
            }
        }

        if ((jobc[0] == 'E' || jobc[0] == 'e') && alphac < ONE) {
            // Form X = -A*(S*S')-( S*S')*A' if DICO = 'C', or
            //      X = -A*(S*S')*A'+(S*S') if DICO = 'D'.
            SLC_DLACPY("Upper", &n, &n, s, &lds, &dwork[ku], &n);
            mb01wd(dico, "U", "T", "H", n, NEGONE, ZERO, s, lds,
                   (f64*)&dwork[kaw], nnw, &dwork[ku], n, &ierr);

            // Eigendecomposition of X: X = Z*Sigma*Z'
            ku = n;
            SLC_DSYEV("V", "U", &n, s, &lds, dwork, &dwork[ku],
                      &(i32){ldwork - n}, &ierr);
            if (ierr > 0) {
                *info = 3;
                return;
            }
            work = work > (dwork[ku] + (f64)n) ? work : (dwork[ku] + (f64)n);

            // Partition Sigma = (Sigma1, Sigma2) with Sigma1 <= 0, Sigma2 > 0
            f64 abs_d1 = fabs(dwork[0]);
            f64 abs_dn = fabs(dwork[n - 1]);
            f64 tol = (abs_d1 > abs_dn ? abs_d1 : abs_dn) * SLC_DLAMCH("E");

            // Form Bbar = [ Z2*sqrt(Sigma2) ]
            i32 mbbar = 0;
            i32 idx = ku;
            for (i32 j = 0; j < n; j++) {
                if (dwork[j] > tol) {
                    mbbar++;
                    f64 sqval = sqrt(dwork[j]);
                    SLC_DSCAL(&n, &sqval, &s[j * lds], &int1);
                    SLC_DCOPY(&n, &s[j * lds], &int1, &dwork[idx], &int1);
                    idx += n;
                }
            }

            // Solve Lyapunov for Cholesky factor S of P = S*S'
            ktau = ku + mbbar * n;
            kw = ktau + n;

            sb03ou(discr, true, n, mbbar, a, lda, &dwork[ku], n,
                   &dwork[ktau], s, lds, &t, &dwork[kw], ldwork - kw, &ierr);
            if (ierr != 0) {
                *info = 2;
                return;
            }
            *scalec *= t;
            work = work > (dwork[kw] + (f64)(kw)) ? work : (dwork[kw] + (f64)(kw));
        }

    } else {
        i32 ku = 0;
        i32 ktau = ku + n * m;
        i32 kw = ktau + n;
        i32 ldwork_rem = ldwork - kw;

        SLC_DLACPY("Full", &n, &m, b, &ldb, &dwork[ku], &n);
        sb03ou(discr, true, n, m, a, lda, &dwork[ku], n,
               &dwork[ktau], s, lds, scalec, &dwork[kw], ldwork_rem, &ierr);
        if (ierr != 0) {
            *info = 2;
            return;
        }
        work = work > (dwork[kw] + (f64)(kw)) ? work : (dwork[kw] + (f64)(kw));
    }

    dwork[0] = work;
}
