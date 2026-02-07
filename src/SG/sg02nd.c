/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SG02ND - Optimal gain matrix for discrete/continuous algebraic Riccati problems
 *
 * Computes:
 * - Discrete: K = (R + B'XB)^{-1} (B'X*op(A) + L')
 * - Continuous: K = R^{-1} (B'X*op(E) + L')
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <math.h>
#include <ctype.h>

void sg02nd(
    const char* dico_str,
    const char* jobe_str,
    const char* job_str,
    const char* jobx_str,
    const char* fact_str,
    const char* uplo_str,
    const char* jobl_str,
    const char* trans_str,
    const i32 n,
    const i32 m,
    const i32 p,
    const f64* a,
    const i32 lda,
    const f64* e,
    const i32 lde,
    f64* b,
    const i32 ldb,
    f64* r,
    const i32 ldr,
    i32* ipiv,
    const f64* l,
    const i32 ldl,
    f64* x,
    const i32 ldx,
    const f64 rnorm,
    f64* k,
    const i32 ldk,
    f64* h,
    const i32 ldh,
    f64* xe,
    const i32 ldxe,
    i32* oufact,
    f64* dwork,
    const i32 ldwork,
    i32* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const i32 int1 = 1;
    char dico = toupper((unsigned char)dico_str[0]);
    char jobe = toupper((unsigned char)jobe_str[0]);
    char job = toupper((unsigned char)job_str[0]);
    char jobx = toupper((unsigned char)jobx_str[0]);
    char fact = toupper((unsigned char)fact_str[0]);
    char uplo = toupper((unsigned char)uplo_str[0]);
    char jobl = toupper((unsigned char)jobl_str[0]);
    char trans = toupper((unsigned char)trans_str[0]);

    bool discr = (dico == 'D');
    bool ljobe = (jobe == 'G');
    bool lfactc = (fact == 'C');
    bool lfactd = (fact == 'D');
    bool lfactu = (fact == 'U');
    bool luplou = (uplo == 'U');
    bool ltrans = (trans == 'T' || trans == 'C');
    bool withc = (job == 'C');
    bool withd = (job == 'D');
    bool withf = (job == 'F');
    bool withh = (job == 'H');
    bool withl = (jobl == 'N');
    bool withxe = (jobx == 'C');
    bool lfacta = lfactc || lfactd || lfactu;
    bool withcd = withc || withd;
    bool lnfact = !lfacta;

    withh = withh || withf || withcd;

    *info = 0;

    if (!discr && dico != 'C') {
        *info = -1;
    } else if (!ljobe && jobe != 'I' && !discr) {
        *info = -2;
    } else if (!withh && job != 'K') {
        *info = -3;
    } else if (!withxe && jobx != 'N') {
        if (discr || ljobe) {
            *info = -4;
        }
    } else if ((lnfact && fact != 'N') || (discr && lfactu) || (!lfactc && withcd)) {
        *info = -5;
    } else if (!luplou && uplo != 'L') {
        *info = -6;
    } else if (!withl && jobl != 'Z') {
        *info = -7;
    } else if (!ltrans && trans != 'N') {
        *info = -8;
    } else if (n < 0) {
        *info = -9;
    } else if (m < 0) {
        *info = -10;
    } else if (lfactd && (p < 0 || (!discr && p < m))) {
        *info = -11;
    } else if (lda < 1 || (lda < n && discr)) {
        *info = -13;
    } else if (lde < 1 || (lde < n && !discr && ljobe)) {
        *info = -15;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -17;
    } else if (ldr < (m > 1 ? m : 1) || (lfactd && ldr < p)) {
        *info = -19;
    } else if (ldl < 1 || (withl && ldl < n)) {
        *info = -22;
    } else if (ldx < (n > 1 ? n : 1)) {
        *info = -24;
    } else if (lfactu && rnorm < ZERO) {
        *info = -25;
    }

    if (*info == 0) {
        if (ldk < (m > 1 ? m : 1)) {
            *info = -27;
        } else if (ldh < 1 || (withh && ldh < n)) {
            *info = -29;
        } else if (ldxe < 1 || (withxe && (discr || ljobe) && ldxe < n)) {
            *info = -31;
        }
    }

    i32 nm = n * m;
    i32 wrkmin;
    char nuplo;
    const char* nuplo_str;

    if (luplou) {
        nuplo = 'L';
        nuplo_str = "L";
    } else {
        nuplo = 'U';
        nuplo_str = "U";
    }

    if (*info == 0) {
        if (discr && lfacta && !withcd) {
            wrkmin = 2;
            if (3 * m > wrkmin) wrkmin = 3 * m;
            if (4 * n + 1 > wrkmin) wrkmin = 4 * n + 1;
        } else {
            if (!withxe && (discr || ljobe)) {
                wrkmin = (2 > n) ? 2 : n;
            } else {
                wrkmin = 2;
            }
            if (lfactu) {
                if (2 * m > wrkmin) wrkmin = 2 * m;
            } else {
                if (3 * m > wrkmin) wrkmin = 3 * m;
            }
        }

        if (ldwork == -1) {
            i32 wrkopt = wrkmin;
            if (lfacta) {
                if (lfactd) {
                    i32 pmin = (p < m) ? p : m;
                    wrkopt = (wrkopt > nm + pmin) ? wrkopt : (nm + pmin);
                }
                if (discr && !withcd) {
                    wrkopt = (wrkopt > nm + n + 2) ? wrkopt : (nm + n + 2);
                }
            }
            dwork[0] = (f64)wrkopt;
            return;
        } else if (ldwork == -2) {
            dwork[0] = (f64)wrkmin;
            return;
        } else if (ldwork < wrkmin) {
            *info = -35;
            dwork[0] = (f64)wrkmin;
            return;
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 || m == 0) {
        dwork[0] = TWO;
        if (n == 0) {
            dwork[1] = ONE;
        } else {
            dwork[1] = ZERO;
        }
        return;
    }

    f64 eps = SLC_DLAMCH("Precision");

    bool sufwrk = ldwork >= nm;

    const char* nt = "N";
    const char* tr = "T";
    const char* ntrans;
    const char* side;

    if (ltrans) {
        ntrans = nt;
        side = "R";
    } else {
        ntrans = tr;
        side = "L";
    }

    f64 temp = withl ? ONE : ZERO;

    i32 wrkopt = wrkmin;
    bool lastcs = false;
    f64 rnormp;
    i32 ifail;

    if (withxe && (discr || ljobe)) {
        lastcs = false;
        if (discr) {
            if (lnfact) {
                if (m <= n) {
                    mb01ru(uplo_str, tr, m, n, ONE, ONE, r, ldr, b, ldb, x, ldx, xe, nm, info);
                    wrkopt = 0;
                } else if (sufwrk) {
                    SLC_DSYMM(side, uplo_str, &n, &m, &ONE, x, &ldx, b, &ldb, &ZERO, dwork, &n);
                    mb01rb("L", uplo_str, tr, m, n, ONE, ONE, r, ldr, b, ldb, dwork, n, &ifail);
                    wrkopt = nm;
                } else {
                    SLC_DGEMM(tr, nt, &m, &n, &n, &ONE, b, &ldb, x, &ldx, &ZERO, k, &ldk);
                    mb01rb("L", uplo_str, nt, m, n, ONE, ONE, r, ldr, k, ldk, b, ldb, &ifail);
                    wrkopt = 0;
                }
            }
            SLC_DSYMM(side, uplo_str, &n, &n, &ONE, x, &ldx, a, &lda, &ZERO, xe, &ldxe);
        } else {
            SLC_DSYMM(side, uplo_str, &n, &n, &ONE, x, &ldx, e, &lde, &ZERO, xe, &ldxe);
            wrkopt = 0;
        }

        if (withh) {
            if (withl) {
                SLC_DLACPY("A", &n, &m, l, &ldl, h, &ldh);
            }
            SLC_DGEMM(ntrans, nt, &n, &m, &n, &ONE, xe, &ldxe, b, &ldb, &temp, h, &ldh);
            ma02ad("A", n, m, h, ldh, k, ldk);
        } else {
            if (withl) {
                ma02ad("A", n, m, l, ldl, k, ldk);
            }
            SLC_DGEMM(tr, trans_str, &m, &n, &n, &ONE, b, &ldb, xe, &ldxe, &temp, k, &ldk);
        }
    } else if (((discr && lfacta) || (!discr && ljobe)) && n < m && ldwork >= n * n) {
        lastcs = false;
        if (discr) {
            SLC_DSYMM(side, uplo_str, &n, &n, &ONE, x, &ldx, a, &lda, &ZERO, dwork, &n);
        } else {
            SLC_DSYMM(side, uplo_str, &n, &n, &ONE, x, &ldx, e, &lde, &ZERO, dwork, &n);
        }

        if (withh) {
            if (withl) {
                SLC_DLACPY("A", &n, &m, l, &ldl, h, &ldh);
            }
            SLC_DGEMM(ntrans, nt, &n, &m, &n, &ONE, dwork, &n, b, &ldb, &temp, h, &ldh);
            ma02ad("A", n, m, h, ldh, k, ldk);
        } else {
            if (withl) {
                ma02ad("A", n, m, l, ldl, k, ldk);
            }
            SLC_DGEMM(tr, trans_str, &m, &n, &n, &ONE, b, &ldb, dwork, &n, &temp, k, &ldk);
        }
        wrkopt = n * n;
    } else if (!(discr || ljobe) && withh) {
        lastcs = false;

        if (withh) {
            if (withl) {
                SLC_DLACPY("A", &n, &m, l, &ldl, h, &ldh);
            }
            SLC_DSYMM("L", uplo_str, &n, &m, &ONE, x, &ldx, b, &ldb, &temp, h, &ldh);
            ma02ad("A", n, m, h, ldh, k, ldk);
            wrkopt = 0;
        }
    } else if (sufwrk) {
        lastcs = false;
        if (discr || ljobe) {
            SLC_DSYMM("L", uplo_str, &n, &m, &ONE, x, &ldx, b, &ldb, &ZERO, dwork, &n);
            if (withh) {
                if (withl) {
                    SLC_DLACPY("A", &n, &m, l, &ldl, h, &ldh);
                }
                if (discr) {
                    SLC_DGEMM(ntrans, nt, &n, &m, &n, &ONE, a, &lda, dwork, &n, &temp, h, &ldh);
                } else {
                    SLC_DGEMM(ntrans, nt, &n, &m, &n, &ONE, e, &lde, dwork, &n, &temp, h, &ldh);
                }
                ma02ad("A", n, m, h, ldh, k, ldk);
            } else {
                if (withl) {
                    ma02ad("A", n, m, l, ldl, k, ldk);
                }
                if (discr) {
                    SLC_DGEMM(tr, trans_str, &m, &n, &n, &ONE, dwork, &n, a, &lda, &temp, k, &ldk);
                } else {
                    SLC_DGEMM(tr, trans_str, &m, &n, &n, &ONE, dwork, &n, e, &lde, &temp, k, &ldk);
                }
            }
            wrkopt = nm;
        } else if (!withh || ldwork >= nm) {
            if (withh) {
                if (withl) {
                    SLC_DLACPY("A", &n, &m, l, &ldl, h, &ldh);
                }
                SLC_DSYMM("L", uplo_str, &n, &m, &ONE, x, &ldx, b, &ldb, &temp, h, &ldh);
                ma02ad("A", n, m, h, ldh, k, ldk);
                wrkopt = 0;
            } else {
                if (withl) {
                    SLC_DLACPY("A", &n, &m, l, &ldl, dwork, &n);
                }
                SLC_DSYMM("L", uplo_str, &n, &m, &ONE, x, &ldx, b, &ldb, &temp, dwork, &n);
                ma02ad("A", n, m, dwork, n, k, ldk);
                wrkopt = nm;
            }
        }
    } else {
        lastcs = true;

        if (withh) {
            SLC_DGEMM(nt, nt, &n, &m, &n, &ONE, x, &ldx, b, &ldb, &ZERO, h, &ldh);
        } else {
            SLC_DGEMM(tr, nt, &m, &n, &n, &ONE, b, &ldb, x, &ldx, &ZERO, k, &ldk);
        }
        wrkopt = 0;
    }

    if (lnfact) {
        if (discr && !withxe) {
            if (sufwrk) {
                mb01rb("L", uplo_str, tr, m, n, ONE, ONE, r, ldr, dwork, n, b, ldb, &ifail);
            } else if (withh) {
                mb01rb("L", uplo_str, tr, m, n, ONE, ONE, r, ldr, b, ldb, h, ldh, &ifail);
            } else {
                mb01rb("L", uplo_str, nt, m, n, ONE, ONE, r, ldr, k, ldk, b, ldb, &ifail);
            }
        }

        rnormp = SLC_DLANSY("1", uplo_str, &m, r, &ldr, dwork);
    }

    if (lastcs) {
        i32 ms = ldwork / n;
        if (ms < 1) ms = 1;

        if (withh) {
            if (discr) {
                for (i32 i = 0; i < m; i += ms) {
                    i32 nr = (ms < m - i) ? ms : (m - i);
                    SLC_DLACPY("A", &n, &nr, &h[i * ldh], &ldh, dwork, &n);
                    if (withl) {
                        SLC_DLACPY("A", &n, &nr, &l[i * ldl], &ldl, &h[i * ldh], &ldh);
                    }
                    SLC_DGEMM(ntrans, nt, &n, &nr, &n, &ONE, a, &lda, dwork, &n, &temp, &h[i * ldh], &ldh);
                }
            } else if (ljobe) {
                for (i32 i = 0; i < m; i += ms) {
                    i32 nr = (ms < m - i) ? ms : (m - i);
                    SLC_DLACPY("A", &n, &nr, &h[i * ldh], &ldh, dwork, &n);
                    if (withl) {
                        SLC_DLACPY("A", &n, &nr, &l[i * ldl], &ldl, &h[i * ldh], &ldh);
                    }
                    SLC_DGEMM(ntrans, nt, &n, &nr, &n, &ONE, e, &lde, dwork, &n, &temp, &h[i * ldh], &ldh);
                }
            } else if (withl) {
                for (i32 i = 0; i < m; i++) {
                    SLC_DAXPY(&n, &ONE, &l[i * ldl], &int1, &k[i], &ldk);
                }
            }
            ma02ad("A", n, m, h, ldh, k, ldk);
        } else {
            if (discr) {
                for (i32 i = 0; i < m; i += ms) {
                    i32 nr = (ms < m - i) ? ms : (m - i);
                    SLC_DLACPY("A", &nr, &n, &k[i], &ldk, dwork, &nr);
                    if (withl) {
                        ma02ad("A", n, nr, &l[i * ldl], ldl, &k[i], ldk);
                    }
                    SLC_DGEMM(nt, trans_str, &nr, &n, &n, &ONE, dwork, &nr, a, &lda, &temp, &k[i], &ldk);
                }
            } else if (ljobe) {
                for (i32 i = 0; i < m; i += ms) {
                    i32 nr = (ms < m - i) ? ms : (m - i);
                    SLC_DLACPY("A", &nr, &n, &k[i], &ldk, dwork, &nr);
                    if (withl) {
                        ma02ad("A", n, nr, &l[i * ldl], ldl, &k[i], ldk);
                    }
                    SLC_DGEMM(nt, trans_str, &nr, &n, &n, &ONE, dwork, &nr, e, &lde, &temp, &k[i], &ldk);
                }
            } else if (withl) {
                for (i32 i = 0; i < m; i++) {
                    SLC_DAXPY(&n, &ONE, &l[i * ldl], &int1, &k[i], &ldk);
                }
            }
        }
    }

    if (wrkmin > wrkopt) wrkopt = wrkmin;

    const char* trl;
    if (luplou) {
        trl = nt;
    } else {
        trl = tr;
    }

    f64 rcond;
    i32 jz = 0;

    if (lfacta) {
        if (lfactd) {
            i32 jw = (p < m ? p : m) + 1;
            i32 ldw_rem = ldwork - jw + 1;
            SLC_DGEQRF(&p, &m, r, &ldr, dwork, &dwork[jw - 1], &ldw_rem, &ifail);
            wrkopt = (wrkopt > (i32)dwork[jw - 1] + jw - 1) ? wrkopt : ((i32)dwork[jw - 1] + jw - 1);
            if (p < m) {
                i32 rows = m - p;
                SLC_DLASET("F", &rows, &m, &ZERO, &ZERO, &r[p], &ldr);
            }

            for (i32 i = 0; i < m; i++) {
                if (!luplou) {
                    SLC_DCOPY(&i, &r[i * ldr], &int1, &r[i], &ldr);
                }
                if (r[i + i * ldr] < ZERO) {
                    i32 cnt = m - i;
                    f64 neg1 = -ONE;
                    SLC_DSCAL(&cnt, &neg1, &r[i + i * ldr], &ldr);
                }
            }
        }

        if (discr && !withcd) {
            jz = 0;

            SLC_DCOPY(&n, x, &(i32){ldx + 1}, dwork, &int1);
            SLC_DPOTRF(uplo_str, &n, x, &ldx, &ifail);

            if (ifail == 0) {
                oufact[1] = 1;
                SLC_DTRMM("L", uplo_str, trl, "N", &n, &m, &ONE, x, &ldx, b, &ldb);
            } else {
                oufact[1] = 2;
                SLC_DCOPY(&n, dwork, &int1, x, &(i32){ldx + 1});

                i32 jw = n + 3;
                i32 ldw_rem = ldwork - jw + 1;
                SLC_DSYEV("V", nuplo_str, &n, x, &ldx, &dwork[2], &dwork[jw - 1], &ldw_rem, &ifail);

                if (ifail > 0) {
                    *info = m + 2;
                    return;
                }
                wrkopt = (wrkopt > (i32)dwork[jw - 1] + jw - 1) ? wrkopt : ((i32)dwork[jw - 1] + jw - 1);
                temp = fabs(dwork[n + 1]) * eps * (f64)n;

                while (jz < n && fabs(dwork[jz + 2]) <= temp) {
                    jz++;
                }

                if (lfactd && n - jz + p < m) {
                    oufact[0] = 1;
                    dwork[1] = ZERO;
                    *info = m + 1;
                    return;
                }

                if (dwork[jz + 2] < ZERO) {
                    *info = m + 3;
                    return;
                }

                i32 ms_inner = ldwork >= (jw - 1 + n * m) ? m : (ldwork - jw + 1) / n;
                if (ms_inner < 1) ms_inner = 1;
                wrkopt = (wrkopt > nm + jw - 1) ? wrkopt : (nm + jw - 1);

                i32 nz = n - jz;
                for (i32 i = 0; i < m; i += ms_inner) {
                    i32 nr = (ms_inner < m - i) ? ms_inner : (m - i);
                    SLC_DLACPY("A", &n, &nr, &b[i * ldb], &ldb, &dwork[jw - 1], &n);
                    SLC_DGEMM(tr, nt, &nz, &nr, &n, &ONE, &x[jz * ldx], &ldx, &dwork[jw - 1], &n, &ZERO, &b[jz + i * ldb], &ldb);
                }

                for (i32 i = jz; i < n; i++) {
                    f64 scale = sqrt(dwork[i + 2]);
                    SLC_DSCAL(&m, &scale, &b[i], &ldb);
                }
            }

            if (!luplou) {
                ma02ed(uplo, m, r, ldr);
            }

            i32 nz = n - jz;
            f64 dummy;
            mb04kd('F', m, 0, nz, r, ldr, &b[jz], ldb, &dummy, n, &dummy, m, dwork, &dwork[m]);

            for (i32 i = 0; i < m; i++) {
                if (r[i + i * ldr] < ZERO) {
                    i32 cnt = m - i;
                    f64 neg1 = -ONE;
                    SLC_DSCAL(&cnt, &neg1, &r[i + i * ldr], &ldr);
                }
            }

            if (!luplou) {
                ma02ed(nuplo, m, r, ldr);
            }
        }

        if (!lfactu) {
            SLC_DTRCON("1", uplo_str, "N", &m, r, &ldr, &rcond, dwork, (i32*)dwork, &ifail);
            oufact[0] = 1;
        } else {
            SLC_DSYCON(uplo_str, &m, r, &ldr, ipiv, &rnorm, &rcond, dwork, (i32*)dwork, &ifail);
            oufact[0] = 2;
        }
    } else {
        SLC_DCOPY(&m, r, &(i32){ldr + 1}, dwork, &int1);
        ma02ed(uplo, m, r, ldr);
        SLC_DPOTRF(uplo_str, &m, r, &ldr, &ifail);

        if (ifail == 0) {
            oufact[0] = 1;
            SLC_DPOCON(uplo_str, &m, r, &ldr, &rnormp, &rcond, dwork, (i32*)dwork, &ifail);
        } else {
            oufact[0] = 2;
            SLC_DCOPY(&m, dwork, &int1, r, &(i32){ldr + 1});
            ma02ed(nuplo, m, r, ldr);

            SLC_DSYTRF(uplo_str, &m, r, &ldr, ipiv, dwork, &ldwork, info);
            if (*info > 0) {
                return;
            }
            wrkopt = (wrkopt > (i32)dwork[0]) ? wrkopt : (i32)dwork[0];

            SLC_DSYCON(uplo_str, &m, r, &ldr, ipiv, &rnormp, &rcond, dwork, (i32*)dwork, &ifail);
        }
    }

    dwork[1] = rcond;
    if (rcond < eps) {
        *info = m + 1;
        return;
    }

    if (oufact[0] == 1) {
        if (withf) {
            SLC_DTRSM("R", uplo_str, trl, "N", &n, &m, &ONE, r, &ldr, h, &ldh);
        } else if (withd) {
            SLC_DTRMM("R", uplo_str, trl, "N", &n, &m, &ONE, r, &ldr, h, &ldh);
            SLC_DTRSM("L", uplo_str, trl, "N", &m, &n, &ONE, r, &ldr, k, &ldk);
        } else if (!withc) {
            SLC_DPOTRS(uplo_str, &m, &n, r, &ldr, k, &ldk, &ifail);
        }
    } else {
        SLC_DSYTRS(uplo_str, &m, &n, r, &ldr, ipiv, k, &ldk, &ifail);
    }

    dwork[0] = (f64)wrkopt;
}
