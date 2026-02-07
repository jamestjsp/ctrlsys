/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB02OD - Algebraic Riccati Equation Solver
 *
 * Solves continuous-time or discrete-time algebraic Riccati equations
 * using the method of deflating subspaces.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <math.h>
#include <ctype.h>

static int select_stable_cont(const f64* reig, const f64* ieig) {
    (void)ieig;
    return *reig < 0.0;
}

static int select_unstable_cont(const f64* reig, const f64* ieig) {
    (void)ieig;
    return *reig >= 0.0;
}

static int select_unstable_disc(const f64* alphar, const f64* alphai, const f64* beta) {
    f64 absalpha = sqrt((*alphar) * (*alphar) + (*alphai) * (*alphai));
    return absalpha >= fabs(*beta);
}

static int select_stable_cont_gen(const f64* alphar, const f64* alphai, const f64* beta) {
    (void)alphai;
    return ((*alphar < 0.0) && (*beta > 0.0)) ||
           ((*alphar > 0.0) && (*beta < 0.0));
}

static int select_unstable_cont_gen(const f64* alphar, const f64* alphai, const f64* beta) {
    (void)alphai;
    return ((*alphar < 0.0) && (*beta < 0.0)) ||
           ((*alphar > 0.0) && (*beta > 0.0));
}

void sb02od(
    const char* dico_str,
    const char* jobb_str,
    const char* fact_str,
    const char* uplo_str,
    const char* jobl_str,
    const char* sort_str,
    const i32 n,
    const i32 m,
    const i32 p,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    f64* q,
    const i32 ldq,
    f64* r,
    const i32 ldr,
    f64* l,
    const i32 ldl,
    f64* rcond,
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
    i32* info)
{
    const f64 zero = 0.0;
    const f64 half = 0.5;
    const f64 one = 1.0;
    const f64 mone = -1.0;
    const f64 three = 3.0;
    const i32 int1 = 1;
    const i32 int0 = 0;

    char dico = toupper((unsigned char)dico_str[0]);
    char jobb = toupper((unsigned char)jobb_str[0]);
    char fact = toupper((unsigned char)fact_str[0]);
    char uplo = toupper((unsigned char)uplo_str[0]);
    char jobl = toupper((unsigned char)jobl_str[0]);
    char sort = toupper((unsigned char)sort_str[0]);

    bool discr = (dico == 'D');
    bool ljobb = (jobb == 'B');
    bool lfacn = (fact == 'N');
    bool lfacq = (fact == 'C');
    bool lfacr = (fact == 'D');
    bool lfacb = (fact == 'B');
    bool luplo = (uplo == 'U');
    bool lsort = (sort == 'S');

    i32 nn = 2 * n;
    i32 nnm, ldw;
    bool ljobl = false, ljobln = false;

    if (ljobb) {
        ljobl = (jobl == 'Z');
        ljobln = (jobl == 'N');
        nnm = nn + m;
        ldw = (nnm > 3*m) ? nnm : 3*m;
    } else {
        nnm = nn;
        ldw = 1;
    }
    i32 np1 = n;

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
        }
    }
    if (*info == 0) {
        if (!lsort && sort != 'U') {
            *info = -6;
        } else if (n < 0) {
            *info = -7;
        } else if (ljobb) {
            if (m < 0) {
                *info = -8;
            }
        }
    }
    if (*info == 0 && !lfacn) {
        if (p < 0) {
            *info = -9;
        }
    }
    if (*info == 0) {
        i32 min_lda = n > 1 ? n : 1;
        if (lda < min_lda) {
            *info = -11;
        } else if (ldb < min_lda) {
            *info = -13;
        } else if (((lfacn || lfacr) && ldq < min_lda) ||
                   ((lfacq || lfacb) && ldq < (p > 1 ? p : 1))) {
            *info = -15;
        } else if (ldr < 1) {
            *info = -17;
        } else if (ldl < 1) {
            *info = -19;
        } else if (ljobb) {
            if (((lfacn || lfacq) && ldr < m) ||
                ((lfacr || lfacb) && ldr < p)) {
                *info = -17;
            } else if (ljobln && ldl < n) {
                *info = -19;
            }
        }
    }
    if (*info == 0) {
        i32 min_ldx = n > 1 ? n : 1;
        if (ldx < min_ldx) {
            *info = -22;
        } else if (lds < (nnm > 1 ? nnm : 1)) {
            *info = -27;
        } else if (ldt < 1) {
            *info = -29;
        } else if (ldu < (nn > 1 ? nn : 1)) {
            *info = -31;
        } else if (ldwork < (6*n > 3 ? 6*n : 3)) {
            *info = -35;
        } else if (discr || ljobb) {
            if (ldt < nnm) {
                *info = -29;
            } else {
                i32 req1 = 14*n + 23;
                i32 req2 = 16*n;
                i32 req = req1 > req2 ? req1 : req2;
                req = req > ldw ? req : ldw;
                if (ldwork < req) {
                    *info = -35;
                }
            }
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        *rcond = one;
        dwork[0] = three;
        dwork[2] = one;
        return;
    }

    bool lscal = true;
    i32 info1 = 0;
    i32 wrkopt;
    f64 scale = one, qscal = one, rscal = one, rnorm = zero, rcondl = zero;
    i32 np = 0, mp = 0;
    char qtype = 'G', rtype = 'G';

    if (lscal && ljobb) {
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
        if (scale == zero) {
            scale = one;
        }

        if (lfacn || lfacr) {
            qscal = scale;
        } else {
            qscal = sqrt(scale);
        }

        if (lfacn || lfacq) {
            rscal = scale;
        } else {
            rscal = sqrt(scale);
        }

        char qtype_str[2] = {qtype, '\0'};
        char rtype_str[2] = {rtype, '\0'};
        SLC_DLASCL(qtype_str, &int0, &int0, &qscal, &one, &np, &n, q, &ldq, &info1);
        SLC_DLASCL(rtype_str, &int0, &int0, &rscal, &one, &mp, &m, r, &ldr, &info1);
        if (ljobln) {
            SLC_DLASCL("G", &int0, &int0, &scale, &one, &n, &m, l, &ldl, &info1);
        }
    }

    sb02oy("O", dico_str, jobb_str, fact_str, uplo_str, jobl_str, "I",
           n, m, p, a, lda, b, ldb, q, ldq, r, ldr, l, ldl, u, 1,
           s, lds, t, ldt, tol, iwork, dwork, ldwork, info);

    if (lscal && ljobb) {
        char qtype_str[2] = {qtype, '\0'};
        char rtype_str[2] = {rtype, '\0'};
        SLC_DLASCL(qtype_str, &int0, &int0, &one, &qscal, &np, &n, q, &ldq, &info1);
        SLC_DLASCL(rtype_str, &int0, &int0, &one, &rscal, &mp, &m, r, &ldr, &info1);
        if (ljobln) {
            SLC_DLASCL("G", &int0, &int0, &one, &scale, &n, &m, l, &ldl, &info1);
        }
    }

    if (*info != 0) {
        return;
    }
    wrkopt = (i32)dwork[0];
    if (ljobb) {
        rcondl = dwork[1];
    }

    bool lscl = false;
    if (lscal && !ljobb) {
        if (lfacn || lfacr) {
            scale = sqrt(SLC_DLANSY("1", &uplo, &n, q, &ldq, dwork));
        } else {
            scale = SLC_DLANGE("1", &p, &n, q, &ldq, dwork);
        }
        rnorm = sqrt(SLC_DLANSY("1", &uplo, &n, b, &ldb, dwork));

        f64 minval = scale < rnorm ? scale : rnorm;
        lscl = (minval > zero) && (scale != rnorm);

        if (lscl) {
            if (discr) {
                SLC_DLASCL("G", &int0, &int0, &scale, &rnorm, &n, &n, &s[np1], &lds, &info1);
                SLC_DLASCL("G", &int0, &int0, &rnorm, &scale, &n, &n, &t[np1*ldt], &ldt, &info1);
            } else {
                f64 mrnorm = -rnorm;
                SLC_DLASCL("G", &int0, &int0, &scale, &mrnorm, &n, &n, &s[np1], &lds, &info1);
                SLC_DLASCL("G", &int0, &int0, &rnorm, &scale, &n, &n, &s[np1*lds], &lds, &info1);
                SLC_DLASCL("G", &int0, &int0, &one, &mone, &n, &n, &s[np1 + np1*lds], &lds, &info1);
            }
        } else {
            if (!discr) {
                SLC_DLASCL("G", &int0, &int0, &one, &mone, &n, &nn, &s[np1], &lds, &info1);
            }
        }
    } else {
        lscl = false;
    }

    i32 ndim;
    i32 bwork_arr[256];
    i32* bwork = (nn <= 256) ? bwork_arr : (i32*)malloc(nn * sizeof(i32));

    if (discr) {
        if (lsort) {
            SLC_DGGES("N", "V", "S", select_unstable_disc, &nn, t, &ldt, s, &lds,
                      &ndim, alfar, alfai, beta, u, &ldu, u, &ldu,
                      dwork, &ldwork, bwork, &info1);
            SLC_DSWAP(&n, &alfar[np1], &int1, alfar, &int1);
            SLC_DSWAP(&n, &alfai[np1], &int1, alfai, &int1);
            SLC_DSWAP(&n, &beta[np1], &int1, beta, &int1);
        } else {
            SLC_DGGES("N", "V", "S", select_unstable_disc, &nn, s, &lds, t, &ldt,
                      &ndim, alfar, alfai, beta, u, &ldu, u, &ldu,
                      dwork, &ldwork, bwork, &info1);
        }
    } else {
        if (ljobb) {
            if (lsort) {
                SLC_DGGES("N", "V", "S", select_stable_cont_gen, &nn, s, &lds, t, &ldt,
                          &ndim, alfar, alfai, beta, u, &ldu, u, &ldu,
                          dwork, &ldwork, bwork, &info1);
            } else {
                SLC_DGGES("N", "V", "S", select_unstable_cont_gen, &nn, s, &lds, t, &ldt,
                          &ndim, alfar, alfai, beta, u, &ldu, u, &ldu,
                          dwork, &ldwork, bwork, &info1);
            }
        } else {
            if (lsort) {
                SLC_DGEES("V", "S", select_stable_cont, &nn, s, &lds, &ndim,
                          alfar, alfai, u, &ldu, dwork, &ldwork, bwork, &info1);
            } else {
                SLC_DGEES("V", "S", select_unstable_cont, &nn, s, &lds, &ndim,
                          alfar, alfai, u, &ldu, dwork, &ldwork, bwork, &info1);
            }
            f64 dum = one;
            SLC_DCOPY(&nn, &dum, &int0, beta, &int1);
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

    for (i32 j = 0; j < n; j++) {
        SLC_DCOPY(&n, &u[np1 + j*ldu], &int1, &x[j], &ldx);
    }

    SLC_DLACPY("F", &n, &n, u, &ldu, &s[np1], &lds);

    f64 unorm = SLC_DLANGE("1", &n, &n, &s[np1], &lds, dwork);

    SLC_DGETRF(&n, &n, &s[np1], &lds, iwork, &info1);
    if (info1 != 0) {
        *info = 6;
        dwork[2] = one;
        if (lscal) {
            if (ljobb) {
                dwork[2] = scale;
            } else if (lscl) {
                dwork[2] = scale / rnorm;
            }
        }
        return;
    }

    SLC_DGECON("1", &n, &s[np1], &lds, &unorm, rcond, dwork, &iwork[np1], &info1);

    if (*rcond < SLC_DLAMCH("E")) {
        *info = 6;
        return;
    }
    wrkopt = wrkopt > 3*n ? wrkopt : 3*n;

    SLC_DGETRS("T", &n, &n, &s[np1], &lds, iwork, x, &ldx, &info1);

    SLC_DLASET("F", &n, &n, &zero, &zero, &s[np1], &lds);

    if (lscal) {
        if (!ljobb) {
            if (lscl) {
                scale = scale / rnorm;
            } else {
                scale = one;
            }
        }
        dwork[2] = scale;
        scale = half * scale;
    } else {
        dwork[2] = one;
        scale = half;
    }

    for (i32 i = 0; i < n; i++) {
        i32 len = n - i;
        SLC_DAXPY(&len, &one, &x[i + i*ldx], &ldx, &x[i + i*ldx], &int1);
        SLC_DSCAL(&len, &scale, &x[i + i*ldx], &int1);
        SLC_DCOPY(&len, &x[i + i*ldx], &int1, &x[i + i*ldx], &ldx);
    }

    dwork[0] = (f64)wrkopt;
    if (ljobb) {
        dwork[1] = rcondl;
    }
}
