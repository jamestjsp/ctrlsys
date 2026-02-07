/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB03OZ - Complex Lyapunov equation solver for Cholesky factor
 *
 * Solves for X = op(U)^H * op(U) either the stable continuous-time Lyapunov equation:
 *     op(A)^H * X + X * op(A) = -scale^2 * op(B)^H * op(B)
 * or the convergent discrete-time Lyapunov equation:
 *     op(A)^H * X * op(A) - X = -scale^2 * op(B)^H * op(B)
 *
 * where A is N-by-N complex, op(B) is M-by-N complex, U is upper triangular (Cholesky factor).
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>
#include <string.h>

static bool lsame(const char* ca, const char* cb) {
    char a = ca[0];
    char b = cb[0];
    if (a >= 'a' && a <= 'z') a -= 32;
    if (b >= 'a' && b <= 'z') b -= 32;
    return a == b;
}

static int select_dummy(const c128* w) {
    (void)w;
    return 0;
}

static bool ma02hz_is_identity(i32 n, c128 diag, const c128* a, i32 lda) {
    const c128 czero = 0.0;
    if (n <= 0) return false;

    for (i32 j = 0; j < n; j++) {
        for (i32 i = 0; i < n; i++) {
            if (i == j) {
                if (a[i + j*lda] != diag) return false;
            } else {
                if (a[i + j*lda] != czero) return false;
            }
        }
    }
    return true;
}

void sb03oz(
    const char* dico, const char* fact, const char* trans,
    const i32 n, const i32 m,
    c128* a, const i32 lda,
    c128* q, const i32 ldq,
    c128* b, const i32 ldb,
    f64* scale,
    c128* w,
    f64* dwork,
    c128* zwork, const i32 lzwork,
    i32* info)
{
    const c128 czero = 0.0;
    const c128 cone = 1.0;
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 p95 = 0.95;
    const f64 neg_one = -1.0;

    bool cont = lsame(dico, "C");
    bool nofact = lsame(fact, "N");
    bool istran = lsame(trans, "C");
    bool lquery = (lzwork == -1);

    i32 minmn = (m < n) ? m : n;
    i32 maxmn = (m > n) ? m : n;

    *info = 0;

    if (!cont && !lsame(dico, "D")) {
        *info = -1;
    } else if (!nofact && !lsame(fact, "F")) {
        *info = -2;
    } else if (!istran && !lsame(trans, "N")) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (m < 0) {
        *info = -5;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldq < (n > 1 ? n : 1)) {
        *info = -9;
    } else if ((istran && ldb < (n > 1 ? n : 1)) ||
               (!istran && ldb < (maxmn > 1 ? maxmn : 1))) {
        *info = -11;
    } else {
        i32 minwrk;
        if (minmn == 0) {
            minwrk = 1;
        } else {
            i32 extra = (minmn - 2 > 0) ? (minmn - 2) : 0;
            minwrk = 2*n + extra;
        }

        if (lquery) {
            i32 wrkopt = minwrk;
            if (nofact) {
                i32 sdim, ifail;
                i32 bwork_tmp;
                i32 lwork_query = -1;
                SLC_ZGEES("V", "N", select_dummy, &n, a, &lda, &sdim, w,
                         q, &ldq, zwork, &lwork_query, dwork, &bwork_tmp, &ifail);
                i32 zgees_opt = (i32)creal(zwork[0]);
                wrkopt = (wrkopt > zgees_opt) ? wrkopt : zgees_opt;
            }

            i32 ifail;
            i32 lwork_query = -1;
            if (istran) {
                SLC_ZGERQF(&n, &maxmn, b, &ldb, zwork, zwork, &lwork_query, &ifail);
            } else {
                SLC_ZGEQRF(&maxmn, &n, b, &ldb, zwork, zwork, &lwork_query, &ifail);
            }
            i32 qr_opt = (i32)creal(zwork[0]) + n;
            wrkopt = (wrkopt > qr_opt) ? wrkopt : qr_opt;

            zwork[0] = (f64)wrkopt;
            return;
        } else if (lzwork < minwrk) {
            zwork[0] = (f64)minwrk;
            *info = -16;
        }
    }

    if (*info != 0) {
        i32 abs_info = -*info;
        SLC_XERBLA("SB03OZ", &abs_info);
        return;
    }

    *scale = one;

    i32 k, l;
    if (istran) {
        k = n;
        l = m;
    } else {
        k = m;
        l = n;
    }

    f64 mb = SLC_ZLANGE("M", &k, &l, b, &ldb, dwork);
    if (mb == zero) {
        if (n > 0) {
            SLC_ZLASET("F", &n, &n, &czero, &czero, b, &ldb);
        }
        zwork[0] = cone;
        return;
    }

    f64 eps = SLC_DLAMCH("P");
    f64 safmin = SLC_DLAMCH("S");
    f64 smlnum = safmin;
    f64 bignms = one / smlnum;
    SLC_DLABAD(&smlnum, &bignms);
    smlnum = sqrt(smlnum) / eps;
    f64 bignum = one / smlnum;

    i32 wrkopt = 0;
    i32 sdim, ifail;

    if (nofact) {
        i32* bwork = (i32*)dwork;
        SLC_ZGEES("V", "N", select_dummy, &n, a, &lda, &sdim, w,
                 q, &ldq, zwork, &lzwork, dwork, bwork, &ifail);
        if (ifail != 0) {
            *info = 6;
            return;
        }
        wrkopt = (i32)creal(zwork[0]);
    } else {
        i32 int1 = 1;
        i32 ldap1 = lda + 1;
        SLC_ZCOPY(&n, a, &ldap1, w, &int1);
        wrkopt = 0;
    }

    bool nunitq = !ma02hz_is_identity(n, cone, q, ldq);

    f64 emax;
    if (cont) {
        emax = creal(w[0]);
        for (i32 j = 1; j < n; j++) {
            f64 tmp = creal(w[j]);
            if (tmp > emax) emax = tmp;
        }
    } else {
        emax = cabs(w[0]);
        for (i32 j = 1; j < n; j++) {
            f64 tmp = cabs(w[j]);
            if (tmp > emax) emax = tmp;
        }
    }

    if ((cont && emax >= zero) || (!cont && emax >= one)) {
        if (nofact) {
            *info = 2;
        } else {
            *info = 3;
        }
        return;
    }

    f64 ma = SLC_ZLANTR("M", "U", "N", &n, &n, a, &lda, dwork);
    ma = (ma < bignms) ? ma : bignms;
    f64 mn = (ma < mb) ? ma : mb;
    f64 mx = (ma > mb) ? ma : mb;

    bool lscl = false;
    if (cont) {
        lscl = (mn < mx*smlnum || mx < smlnum || mn > bignum);
    }

    f64 mato = one, mbto = one;
    bool lascl = false, lbscl = false;

    if (lscl) {
        mato = one;
        mbto = one;
        lascl = true;
        lbscl = true;
    } else {
        if (ma > zero && ma < smlnum) {
            mato = smlnum;
            lascl = true;
        } else if (ma > bignum) {
            mato = bignum;
            lascl = true;
        }

        if (mb > zero && mb < smlnum) {
            mbto = smlnum;
            lbscl = true;
        } else if (mb > bignum) {
            mbto = bignum;
            lbscl = true;
        }
    }

    if (!cont && mato == one) {
        mato = p95;
    }

    if (lascl) {
        i32 kl = 0, ku = 0, info_scl;
        SLC_ZLASCL("U", &kl, &ku, &ma, &mato, &n, &n, a, &lda, &info_scl);
    }

    bool scalb = (mb > bignms);
    mb = (mb < bignms) ? mb : bignms;

    if (lbscl && scalb) {
        i32 kl = 0, ku = 0, info_scl;
        SLC_ZLASCL("G", &kl, &ku, &mb, &mbto, &k, &l, b, &ldb, &info_scl);
    }

    i32 itau = 0;
    i32 jwork = itau + minmn;
    i32 nm;
    bool smallm_effective = (6*m <= 7*n);

    if (istran) {
        nm = m;
        if (nunitq) {
            if (smallm_effective) {
                i32 nc = lzwork / n;
                for (i32 j = 0; j < m; j += nc) {
                    i32 bl = (m - j < nc) ? (m - j) : nc;
                    c128 alpha = cone, beta = czero;
                    SLC_ZGEMM("C", "N", &n, &bl, &n, &alpha, q, &ldq, &b[j*ldb], &ldb, &beta, zwork, &n);
                    SLC_ZLACPY("A", &n, &bl, zwork, &n, &b[j*ldb], &ldb);
                }
            } else {
                nm = n;
                i32 ldw_avail = lzwork - jwork;
                SLC_ZGERQF(&n, &m, b, &ldb, &zwork[itau], &zwork[jwork], &ldw_avail, &ifail);
                wrkopt = (wrkopt > (i32)creal(zwork[jwork]) + jwork) ? wrkopt : ((i32)creal(zwork[jwork]) + jwork);
                wrkopt = (wrkopt > minmn*n) ? wrkopt : minmn*n;

                if (lzwork >= minmn*n) {
                    i32 jj = 0;
                    for (i32 i = 0; i < minmn; i++) {
                        i32 int1 = 1;
                        SLC_ZCOPY(&n, &q[(n - minmn + i)], &ldq, &zwork[jj], &int1);
                        SLC_ZLACGV(&n, &zwork[jj], &int1);
                        jj += n;
                    }
                    SLC_ZTRMM("R", "U", "N", "N", &n, &minmn, &cone, &b[(n - minmn) + (m - minmn)*ldb], &ldb, zwork, &n);
                    SLC_ZLACPY("F", &n, &minmn, zwork, &n, b, &ldb);
                } else {
                    for (i32 j = 0; j < minmn; j++) {
                        i32 jp1 = j + 1;
                        i32 int1 = 1;
                        SLC_ZCOPY(&jp1, &b[(m - minmn + j)*ldb], &int1, zwork, &int1);
                        c128 alpha = cone, beta = czero;
                        SLC_ZGEMV("C", &jp1, &n, &alpha, q, &ldq, zwork, &int1, &beta, &b[j*ldb], &int1);
                    }
                }
            }
        }

        i32 ldw_avail = lzwork - jwork;
        SLC_ZGERQF(&n, &nm, b, &ldb, &zwork[itau], &zwork[jwork], &ldw_avail, &ifail);

        if (n > nm) {
            if (lbscl && !scalb) {
                i32 kl = 0, ku = 0, info_scl;
                i32 nrows = n - m;
                SLC_ZLASCL("G", &kl, &ku, &mb, &mbto, &nrows, &m, b, &ldb, &info_scl);
                SLC_ZLASCL("U", &kl, &ku, &mb, &mbto, &m, &m, &b[(n - m)*ldb], &ldb, &info_scl);
            }
            for (i32 i = m - 1; i >= 0; i--) {
                i32 len = n - m + i + 1;
                i32 int1 = 1;
                SLC_ZCOPY(&len, &b[i*ldb], &int1, &b[(n - m + i)*ldb], &int1);
            }
            i32 nm_diff = n - m;
            SLC_ZLASET("F", &n, &nm_diff, &czero, &czero, b, &ldb);
            if (m > 1) {
                i32 m1 = m - 1;
                SLC_ZLASET("L", &m1, &m1, &czero, &czero, &b[(n - m + 1) + (n - m)*ldb], &ldb);
            }
        } else {
            if (m > n && m == nm) {
                SLC_ZLACPY("U", &n, &n, &b[(m - n)*ldb], &ldb, b, &ldb);
            }
            if (lbscl && !scalb) {
                i32 kl = 0, ku = 0, info_scl;
                SLC_ZLASCL("U", &kl, &ku, &mb, &mbto, &n, &n, b, &ldb, &info_scl);
            }
        }

        for (i32 i = n - minmn; i < n; i++) {
            if (creal(b[i + i*ldb]) < zero) {
                i32 len = i + 1;
                i32 int1 = 1;
                SLC_ZDSCAL(&len, &neg_one, &b[i*ldb], &int1);
            }
        }
    } else {
        nm = m;
        if (nunitq) {
            if (smallm_effective) {
                i32 nr = lzwork / n;
                for (i32 i = 0; i < m; i += nr) {
                    i32 bl = (m - i < nr) ? (m - i) : nr;
                    c128 alpha = cone, beta = czero;
                    SLC_ZGEMM("N", "N", &bl, &n, &n, &alpha, &b[i], &ldb, q, &ldq, &beta, zwork, &bl);
                    SLC_ZLACPY("A", &bl, &n, zwork, &bl, &b[i], &ldb);
                }
            } else {
                i32 ldw_avail = lzwork - jwork;
                SLC_ZGEQRF(&m, &n, b, &ldb, &zwork[itau], &zwork[jwork], &ldw_avail, &ifail);
                wrkopt = (wrkopt > (i32)creal(zwork[jwork]) + jwork) ? wrkopt : ((i32)creal(zwork[jwork]) + jwork);
                wrkopt = (wrkopt > n*n) ? wrkopt : n*n;

                if (lzwork >= n*n) {
                    SLC_ZLACPY("F", &n, &n, q, &ldq, zwork, &n);
                    SLC_ZTRMM("L", "U", "N", "N", &n, &n, &cone, b, &ldb, zwork, &n);
                    SLC_ZLACPY("F", &n, &n, zwork, &minmn, b, &ldb);
                } else {
                    i32 info_uy;
                    mb01uz("L", "U", "N", n, n, cone, b, ldb, q, ldq, zwork, lzwork, &info_uy);
                }
                nm = n;
            }
        }

        i32 ldw_avail = lzwork - jwork;
        SLC_ZGEQRF(&nm, &n, b, &ldb, &zwork[itau], &zwork[jwork], &ldw_avail, &ifail);

        if (lbscl && !scalb) {
            i32 kl = 0, ku = 0, info_scl;
            SLC_ZLASCL("U", &kl, &ku, &mb, &mbto, &nm, &n, b, &ldb, &info_scl);
        }

        if (m < n) {
            i32 nmm = n - m;
            SLC_ZLASET("U", &nmm, &nmm, &czero, &czero, &b[m + m*ldb], &ldb);
        }

        for (i32 i = 0; i < minmn; i++) {
            if (creal(b[i + i*ldb]) < zero) {
                i32 len = n - i;
                SLC_ZDSCAL(&len, &neg_one, &b[i + i*ldb], &ldb);
            }
        }
    }

    if (minmn > 1) {
        i32 m1 = minmn - 1;
        SLC_ZLASET("L", &m1, &m1, &czero, &czero, &b[1], &ldb);
    }

    f64 scaloc;
    sb03os(!cont, istran, n, a, lda, b, ldb, &scaloc, dwork, zwork, info);
    *scale *= scaloc;

    if (istran) {
        if (nunitq) {
            i32 info_uy;
            mb01uz("R", "U", "N", n, n, cone, b, ldb, q, ldq, zwork, lzwork, &info_uy);

            i32 ldw_avail = lzwork - n;
            SLC_ZGERQF(&n, &n, b, &ldb, zwork, &zwork[n], &ldw_avail, &ifail);

            if (n > 1) {
                i32 n1 = n - 1;
                SLC_ZLASET("L", &n1, &n1, &czero, &czero, &b[1], &ldb);
            }

            for (i32 i = 0; i < n; i++) {
                if (creal(b[i + i*ldb]) < zero) {
                    i32 len = i + 1;
                    i32 int1 = 1;
                    SLC_ZDSCAL(&len, &neg_one, &b[i*ldb], &int1);
                }
            }
        }
    } else {
        if (nunitq) {
            i32 info_uy;
            mb01uz("R", "U", "C", n, n, cone, b, ldb, q, ldq, zwork, lzwork, &info_uy);

            for (i32 i = 0; i < n; i++) {
                i32 len = i + 1;
                i32 int1 = 1;
                SLC_ZSWAP(&len, &b[i], &ldb, &b[i*ldb], &int1);
            }

            for (i32 i = 0; i < n; i++) {
                i32 int1 = 1;
                SLC_ZLACGV(&n, &b[i*ldb], &int1);
            }

            i32 ldw_avail = lzwork - n;
            SLC_ZGEQRF(&n, &n, b, &ldb, zwork, &zwork[n], &ldw_avail, &ifail);

            if (n > 1) {
                i32 n1 = n - 1;
                SLC_ZLASET("L", &n1, &n1, &czero, &czero, &b[1], &ldb);
            }

            for (i32 i = 0; i < n; i++) {
                if (creal(b[i + i*ldb]) < zero) {
                    i32 len = n - i;
                    SLC_ZDSCAL(&len, &neg_one, &b[i + i*ldb], &ldb);
                }
            }
        }
    }

    f64 tmp = one;
    if (lascl) {
        i32 kl = 0, ku = 0, info_scl;
        SLC_ZLASCL("U", &kl, &ku, &mato, &ma, &n, &n, a, &lda, &info_scl);
        tmp = sqrt(mato / ma);
    }

    if (lbscl) {
        f64 mx_local = SLC_ZLANTR("M", "U", "N", &n, &n, b, &ldb, dwork);
        f64 mn_local = (tmp < mb) ? tmp : mb;
        f64 t = (tmp > mb) ? tmp : mb;

        if (t > one) {
            if (mn_local > bignms / t) {
                *scale = *scale / t;
                tmp = tmp / t;
            }
        }
        tmp = tmp * mb;
        if (tmp > one) {
            if (mx_local > bignms / tmp) {
                *scale = *scale / mx_local;
                tmp = tmp / mx_local;
            }
        }
    }

    if (lascl || lbscl) {
        i32 kl = 0, ku = 0, info_scl;
        SLC_ZLASCL("U", &kl, &ku, &mbto, &tmp, &n, &n, b, &ldb, &info_scl);
    }

    zwork[0] = (f64)wrkopt;
}
