/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SG02AD - Generalized Algebraic Riccati Equation Solver
 *
 * Solves continuous-time or discrete-time generalized algebraic Riccati
 * equations for descriptor systems using the method of deflating subspaces.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <math.h>
#include <ctype.h>

static int select_stable_cont_gen(const f64* alphar, const f64* alphai, const f64* beta) {
    (void)alphai;
    return ((*alphar < 0.0) && (*beta > 0.0)) ||
           ((*alphar > 0.0) && (*beta < 0.0));
}

static int select_unstable_cont_gen(const f64* alphar, const f64* alphai, const f64* beta) {
    (void)alphai;
    return ((*alphar < 0.0) && (*beta < 0.0)) ||
           ((*alphar > 0.0) && (*beta > 0.0)) ||
           (*alphar == 0.0);
}

static int select_unstable_disc_gen(const f64* alphar, const f64* alphai, const f64* beta) {
    f64 absalpha = sqrt((*alphar) * (*alphar) + (*alphai) * (*alphai));
    return absalpha >= fabs(*beta);
}

void sg02ad(
    const char* dico_str,
    const char* jobb_str,
    const char* fact_str,
    const char* uplo_str,
    const char* jobl_str,
    const char* scal_str,
    const char* sort_str,
    const char* acc_str,
    const i32 n,
    const i32 m,
    const i32 p,
    f64* a,
    const i32 lda,
    f64* e,
    const i32 lde,
    f64* b,
    const i32 ldb,
    f64* q,
    const i32 ldq,
    f64* r,
    const i32 ldr,
    f64* l,
    const i32 ldl,
    f64* rcondu,
    f64* x,
    const i32 ldx,
    f64* alfar,
    f64* alfai,
    f64* beta,
    f64* s,
    const i32 lds,
    f64* t,
    const i32 ldt,
    f64* u,
    const i32 ldu,
    const f64 tol,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* iwarn,
    i32* info)
{
    const f64 ZERO = 0.0;
    const f64 HALF = 0.5;
    const f64 ONE = 1.0;
    const f64 FOUR = 4.0;
    const f64 P1 = 0.1;
    const i32 int1 = 1;
    const i32 int0 = 0;

    char dico = toupper((unsigned char)dico_str[0]);
    char jobb = toupper((unsigned char)jobb_str[0]);
    char fact = toupper((unsigned char)fact_str[0]);
    char uplo = toupper((unsigned char)uplo_str[0]);
    char jobl = toupper((unsigned char)jobl_str[0]);
    char scal = toupper((unsigned char)scal_str[0]);
    char sort = toupper((unsigned char)sort_str[0]);
    char acc = toupper((unsigned char)acc_str[0]);

    bool discr = (dico == 'D');
    bool ljobb = (jobb == 'B');
    bool lfacn = (fact == 'N');
    bool lfacq = (fact == 'C');
    bool lfacr = (fact == 'D');
    bool lfacb = (fact == 'B');
    bool luplo = (uplo == 'U');
    bool lsort = (sort == 'S');
    bool refine = (acc == 'R');

    i32 nn = 2 * n;
    i32 nnm, ldw;
    bool ljobl = false, ljobln = false, lscal = false;

    if (ljobb) {
        ljobl = (jobl == 'Z');
        ljobln = (jobl == 'N');
        lscal = (scal == 'G');
        nnm = nn + m;
        ldw = (nnm > 3*m) ? nnm : 3*m;
    } else {
        lscal = false;
        nnm = nn;
        ldw = 1;
    }
    i32 np1 = n;

    *iwarn = 0;
    *info = 0;

    if (!discr && dico != 'C') {
        *info = -1;
    } else if (!ljobb && jobb != 'G') {
        *info = -2;
    } else if (!lfacq && !lfacr && !lfacb && !lfacn) {
        *info = -3;
    } else if (!ljobb || lfacn) {
        if (!luplo && uplo != 'L') {
            *info = -4;
        }
    }
    if (*info == 0 && ljobb) {
        if (!ljobl && !ljobln) {
            *info = -5;
        } else if (!lscal && scal != 'N') {
            *info = -6;
        }
    }
    if (*info == 0) {
        if (!lsort && sort != 'U') {
            *info = -7;
        } else if (!refine && acc != 'N') {
            *info = -8;
        } else if (n < 0) {
            *info = -9;
        } else if (ljobb && m < 0) {
            *info = -10;
        }
    }
    if (*info == 0 && !lfacn) {
        if (p < 0) {
            *info = -11;
        }
    }
    if (*info == 0) {
        i32 min_lda = n > 1 ? n : 1;
        if (lda < min_lda) {
            *info = -13;
        } else if (lde < min_lda) {
            *info = -15;
        } else if (ldb < min_lda) {
            *info = -17;
        } else if (((lfacn || lfacr) && ldq < min_lda) ||
                   ((lfacq || lfacb) && ldq < (p > 1 ? p : 1))) {
            *info = -19;
        } else if (ldr < 1) {
            *info = -21;
        } else if (ldl < 1) {
            *info = -23;
        } else if (ljobb) {
            if (((lfacn || lfacq) && ldr < m) ||
                ((lfacr || lfacb) && ldr < p)) {
                *info = -21;
            } else if (ljobln && ldl < n) {
                *info = -23;
            }
        }
    }
    if (*info == 0) {
        i32 min_ldx = n > 1 ? n : 1;
        if (ldx < min_ldx) {
            *info = -26;
        } else if (lds < (nnm > 1 ? nnm : 1)) {
            *info = -31;
        } else if (ldt < (nnm > 1 ? nnm : 1)) {
            *info = -33;
        } else if (ldu < (nn > 1 ? nn : 1)) {
            *info = -35;
        } else {
            i32 req1 = 14*n + 23;
            i32 req2 = 16*n;
            i32 req = req1 > req2 ? req1 : req2;
            req = req > ldw ? req : ldw;
            if (ldwork < req) {
                *info = -39;
            }
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        dwork[0] = FOUR;
        dwork[3] = ONE;
        return;
    }

    lscal = lscal && ljobb;
    i32 info1 = 0;
    i32 wrkopt;
    f64 scale = ONE, rcondl = ZERO;
    i32 np = 0, mp = 0;
    char qtype = 'G', rtype = 'G';
    f64 rnorm = ZERO, qscal = ONE, rscal = ONE;

    if (lscal) {
        if (lfacn || lfacr) {
            scale = SLC_DLANSY("1", &uplo, &n, q, &ldq, dwork);
            qtype = uplo;
            np = n;
        } else {
            scale = SLC_DLANGE("1", &p, &n, q, &ldq, dwork);
            qtype = 'G';
            np = p;
        }

        if (lfacn || lfacq) {
            rnorm = SLC_DLANSY("1", &uplo, &m, r, &ldr, dwork);
            rtype = uplo;
            mp = m;
        } else {
            rnorm = SLC_DLANGE("1", &p, &m, r, &ldr, dwork);
            rtype = 'G';
            mp = p;
        }
        scale = scale + rnorm;

        if (ljobln) {
            scale = scale + SLC_DLANGE("1", &n, &m, l, &ldl, dwork);
        }
        if (scale == ZERO) {
            scale = ONE;
        }

        // Factors C, D enter as C'C, D'D, so they are scaled by sqrt(scale).
        qscal = (lfacn || lfacr) ? scale : sqrt(scale);
        rscal = (lfacn || lfacq) ? scale : sqrt(scale);

        char qtype_str[2] = {qtype, '\0'};
        char rtype_str[2] = {rtype, '\0'};
        SLC_DLASCL(qtype_str, &int0, &int0, &qscal, &ONE, &np, &n, q, &ldq, &info1);
        SLC_DLASCL(rtype_str, &int0, &int0, &rscal, &ONE, &mp, &m, r, &ldr, &info1);
        if (ljobln) {
            SLC_DLASCL("G", &int0, &int0, &scale, &ONE, &n, &m, l, &ldl, &info1);
        }
    } else {
        scale = ONE;
    }

    sb02oy("O", dico_str, jobb_str, fact_str, uplo_str, jobl_str, "N",
           n, m, p, a, lda, b, ldb, q, ldq, r, ldr, l, ldl, e, lde,
           s, lds, t, ldt, tol, iwork, dwork, ldwork, info);

    if (lscal) {
        char qtype_str[2] = {qtype, '\0'};
        char rtype_str[2] = {rtype, '\0'};
        SLC_DLASCL(qtype_str, &int0, &int0, &ONE, &qscal, &np, &n, q, &ldq, &info1);
        SLC_DLASCL(rtype_str, &int0, &int0, &ONE, &rscal, &mp, &m, r, &ldr, &info1);
        if (ljobln) {
            SLC_DLASCL("G", &int0, &int0, &ONE, &scale, &n, &m, l, &ldl, &info1);
        }
    }

    if (*info != 0) {
        return;
    }
    wrkopt = (i32)dwork[0];
    if (ljobb) {
        rcondl = dwork[1];
    }

    i32 ndim;
    i32 bwork_arr[256];
    i32* bwork = (nn <= 256) ? bwork_arr : (i32*)malloc(nn * sizeof(i32));

    if (discr) {
        if (lsort) {
            SLC_DGGES("N", "V", "S", select_unstable_disc_gen, &nn, t, &ldt, s, &lds,
                      &ndim, alfar, alfai, beta, u, &ldu, u, &ldu,
                      dwork, &ldwork, bwork, &info1);
            SLC_DSWAP(&n, &alfar[np1], &int1, alfar, &int1);
            SLC_DSWAP(&n, &alfai[np1], &int1, alfai, &int1);
            SLC_DSWAP(&n, &beta[np1], &int1, beta, &int1);
        } else {
            SLC_DGGES("N", "V", "S", select_unstable_disc_gen, &nn, s, &lds, t, &ldt,
                      &ndim, alfar, alfai, beta, u, &ldu, u, &ldu,
                      dwork, &ldwork, bwork, &info1);
        }
    } else {
        if (lsort) {
            SLC_DGGES("N", "V", "S", select_stable_cont_gen, &nn, s, &lds, t, &ldt,
                      &ndim, alfar, alfai, beta, u, &ldu, u, &ldu,
                      dwork, &ldwork, bwork, &info1);
        } else {
            SLC_DGGES("N", "V", "S", select_unstable_cont_gen, &nn, s, &lds, t, &ldt,
                      &ndim, alfar, alfai, beta, u, &ldu, u, &ldu,
                      dwork, &ldwork, bwork, &info1);
        }
    }

    if (bwork != bwork_arr) {
        free(bwork);
    }

    if (info1 > 0 && info1 <= nn + 1) {
        *info = 2;
    } else if (info1 == nn + 2) {
        *info = 4;
    } else if (info1 == nn + 3) {
        *info = 3;
    } else if (ndim != n) {
        *info = 5;
    }
    if (*info != 0) {
        return;
    }
    i32 opt_val = (i32)dwork[0];
    wrkopt = wrkopt > opt_val ? wrkopt : opt_val;

    SLC_DGEMM("N", "N", &n, &n, &n, &ONE, e, &lde, u, &ldu, &ZERO, x, &ldx);
    SLC_DLACPY("F", &n, &n, x, &ldx, u, &ldu);

    f64* tau = x;
    SLC_DGEQRF(&nn, &n, u, &ldu, tau, dwork, &ldwork, &info1);
    opt_val = (i32)dwork[0];
    wrkopt = wrkopt > opt_val ? wrkopt : opt_val;

    SLC_DORGQR(&nn, &n, &n, u, &ldu, tau, dwork, &ldwork, &info1);
    opt_val = (i32)dwork[0];
    wrkopt = wrkopt > opt_val ? wrkopt : opt_val;

    SLC_DGEMM("T", "N", &n, &n, &n, &ONE, u, &ldu, &u[np1], &ldu, &ZERO, x, &ldx);

    f64 u12m = ZERO;
    f64 asym = ZERO;
    for (i32 j = 0; j < n; j++) {
        for (i32 i = 0; i < n; i++) {
            f64 val = fabs(x[i + j*ldx]);
            if (val > u12m) u12m = val;
            f64 diff = fabs(x[i + j*ldx] - x[j + i*ldx]);
            if (diff > asym) asym = diff;
        }
    }

    f64 eps = SLC_DLAMCH("E");
    f64 seps = sqrt(eps);
    asym = asym - seps;
    if (asym > P1 * u12m) {
        *info = 6;
        return;
    } else if (asym > seps) {
        *iwarn = 1;
    }

    f64 pivotu = ONE;

    if (refine) {
        for (i32 i = 0; i < n - 1; i++) {
            i32 len = n - i - 1;
            SLC_DSWAP(&len, &u[np1 + i + (i+1)*ldu], &ldu, &u[np1 + i + 1 + i*ldu], &int1);
        }

        i32 iwr = 0;
        i32 iwc = iwr + n;
        i32 iwf = iwc + n;
        i32 iwb = iwf + n;
        i32 iw = iwb + n;

        char equed;
        bool rowequ, colequ;

        mb02pd("E", "T", n, n, u, ldu, &s[np1], lds, iwork, &equed,
               &dwork[iwr], &dwork[iwc], &u[np1], ldu, x, ldx, rcondu,
               &dwork[iwf], &dwork[iwb], &iwork[np1], &dwork[iw], &info1);

        for (i32 i = 0; i < n - 1; i++) {
            i32 len = n - i - 1;
            SLC_DSWAP(&len, &u[np1 + i + (i+1)*ldu], &ldu, &u[np1 + i + 1 + i*ldu], &int1);
        }

        if (equed != 'N') {
            rowequ = (equed == 'R') || (equed == 'B');
            colequ = (equed == 'C') || (equed == 'B');

            if (rowequ) {
                for (i32 i = 0; i < n; i++) {
                    dwork[iwr + i] = ONE / dwork[iwr + i];
                }
                mb01sd('R', n, n, u, ldu, &dwork[iwr], &dwork[iwc]);
            }

            if (colequ) {
                for (i32 i = 0; i < n; i++) {
                    dwork[iwc + i] = ONE / dwork[iwc + i];
                }
                mb01sd('C', nn, n, u, ldu, &dwork[iwr], &dwork[iwc]);
            }
        }

        pivotu = dwork[iw];

        if (info1 > 0) {
            *info = 7;
            goto finish;
        }
    } else {
        SLC_DLACPY("F", &n, &n, u, &ldu, &s[np1], &lds);
        SLC_DLACPY("F", &n, &n, &u[np1], &ldu, x, &ldx);

        mb02vd("N", n, n, &s[np1], lds, iwork, x, ldx, &info1);

        if (info1 != 0) {
            *info = 7;
            *rcondu = ZERO;
            goto finish;
        } else {
            f64 unorm = SLC_DLANGE("1", &n, &n, u, &ldu, dwork);
            SLC_DGECON("1", &n, &s[np1], &lds, &unorm, rcondu, dwork, &iwork[np1], &info1);

            if (*rcondu < eps) {
                *iwarn = 1;
            }
            wrkopt = wrkopt > 4*n ? wrkopt : 4*n;
        }
    }

    SLC_DLASET("F", &n, &n, &ZERO, &ZERO, &s[np1], &lds);

    for (i32 i = 0; i < n - 1; i++) {
        i32 len = n - i - 1;
        SLC_DAXPY(&len, &ONE, &x[i + (i+1)*ldx], &ldx, &x[i+1 + i*ldx], &int1);
        SLC_DSCAL(&len, &HALF, &x[i+1 + i*ldx], &int1);
        SLC_DCOPY(&len, &x[i+1 + i*ldx], &int1, &x[i + (i+1)*ldx], &ldx);
    }

    if (lscal) {
        SLC_DLASCL("G", &int0, &int0, &ONE, &scale, &n, &n, x, &ldx, &info1);
    }

    dwork[0] = (f64)wrkopt;

finish:
    if (ljobb) {
        dwork[1] = rcondl;
    }
    if (refine) {
        dwork[2] = pivotu;
    }
    dwork[3] = scale;
}
