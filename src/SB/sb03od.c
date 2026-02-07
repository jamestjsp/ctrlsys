/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB03OD - Solve Lyapunov equations for Cholesky factor of solution
 *
 * Solves for X = op(U)'*op(U) either the stable continuous-time Lyapunov equation:
 *     op(A)'*X + X*op(A) = -scale^2*op(B)'*op(B)
 * or the convergent discrete-time Lyapunov equation:
 *     op(A)'*X*op(A) - X = -scale^2*op(B)'*op(B)
 *
 * where A is N-by-N, op(B) is M-by-N, U is upper triangular (Cholesky factor).
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

static int select_dummy(const f64* wr, const f64* wi) {
    (void)wr; (void)wi;
    return 0;
}

static bool ma02hd_is_identity(i32 n, f64 diag, const f64* a, i32 lda) {
    const f64 zero = 0.0;
    if (n <= 0) return false;
    
    for (i32 j = 0; j < n; j++) {
        for (i32 i = 0; i < n; i++) {
            if (i == j) {
                if (a[i + j*lda] != diag) return false;
            } else {
                if (a[i + j*lda] != zero) return false;
            }
        }
    }
    return true;
}

void sb03od(
    const char* dico, const char* fact, const char* trans,
    const i32 n, const i32 m,
    f64* a, const i32 lda,
    f64* q, const i32 ldq,
    f64* b, const i32 ldb,
    f64* scale,
    f64* wr, f64* wi,
    f64* dwork, const i32 ldwork,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 p95 = 0.95;

    bool cont = lsame(dico, "C");
    bool nofact = lsame(fact, "N");
    bool istran = lsame(trans, "T");
    bool lquery = (ldwork == -1);
    
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
            minwrk = 4*n;
        }
        
        if (lquery) {
            i32 wrkopt = minwrk;
            if (nofact) {
                i32 sdim, ifail;
                i32 bwork_tmp;
                i32 lwork_query = -1;
                SLC_DGEES("V", "N", select_dummy, &n, a, &lda, &sdim, wr, wi,
                         q, &ldq, dwork, &lwork_query, &bwork_tmp, &ifail);
                i32 dgees_opt = (i32)dwork[0];
                wrkopt = (wrkopt > dgees_opt) ? wrkopt : dgees_opt;
            }
            
            i32 ifail;
            i32 lwork_query = -1;
            if (istran) {
                SLC_DGERQF(&n, &maxmn, b, &ldb, dwork, dwork, &lwork_query, &ifail);
            } else {
                SLC_DGEQRF(&maxmn, &n, b, &ldb, dwork, dwork, &lwork_query, &ifail);
            }
            i32 qr_opt = (i32)dwork[0] + n;
            wrkopt = (wrkopt > qr_opt) ? wrkopt : qr_opt;
            
            dwork[0] = (f64)wrkopt;
            return;
        } else if (ldwork < minwrk) {
            dwork[0] = (f64)minwrk;
            *info = -16;
        }
    }

    if (*info != 0) {
        i32 abs_info = -*info;
        SLC_XERBLA("SB03OD", &abs_info);
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
    
    f64 mb = SLC_DLANGE("M", &k, &l, b, &ldb, dwork);
    if (mb == zero) {
        if (n > 0) {
            SLC_DLASET("F", &n, &n, &zero, &zero, b, &ldb);
        }
        dwork[0] = one;
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
        SLC_DGEES("V", "N", select_dummy, &n, a, &lda, &sdim, wr, wi,
                 q, &ldq, dwork, &ldwork, bwork, &ifail);
        if (ifail != 0) {
            *info = 6;
            return;
        }
        wrkopt = (i32)dwork[0];
    } else {
        i32 i = 0;
        while (i < n) {
            if (i < n - 1) {
                if (a[(i+1) + i*lda] != zero) {
                    f64 a_ii = a[i + i*lda];
                    f64 a_i1i = a[(i+1) + i*lda];
                    f64 a_ii1 = a[i + (i+1)*lda];
                    f64 a_i1i1 = a[(i+1) + (i+1)*lda];
                    f64 cs, sn;
                    SLC_DLANV2(&a_ii, &a_ii1, &a_i1i, &a_i1i1,
                              &wr[i], &wi[i], &wr[i+1], &wi[i+1], &cs, &sn);
                    i += 2;
                } else {
                    wr[i] = a[i + i*lda];
                    wi[i] = zero;
                    i++;
                }
            } else {
                wr[i] = a[i + i*lda];
                wi[i] = zero;
                i++;
            }
        }
        wrkopt = 0;
    }

    bool nunitq = !ma02hd_is_identity(n, one, q, ldq);

    f64 emax;
    if (cont) {
        emax = wr[0];
        for (i32 j = 1; j < n; j++) {
            if (wr[j] > emax) emax = wr[j];
        }
    } else {
        emax = SLC_DLAPY2(&wr[0], &wi[0]);
        for (i32 j = 1; j < n; j++) {
            f64 tmp = SLC_DLAPY2(&wr[j], &wi[j]);
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

    f64 ma = SLC_DLANHS("M", &n, a, &lda, dwork);
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
        SLC_DLASCL("H", &kl, &ku, &ma, &mato, &n, &n, a, &lda, &info_scl);
    }

    bool scalb = (mb > bignms);
    mb = (mb < bignms) ? mb : bignms;
    
    if (lbscl && scalb) {
        i32 kl = 0, ku = 0, info_scl;
        SLC_DLASCL("G", &kl, &ku, &mb, &mbto, &k, &l, b, &ldb, &info_scl);
    }

    i32 itau = 0;
    i32 jwork = itau + minmn;
    i32 nm;
    bool smallm_effective = (6*m <= 7*n);

    if (istran) {
        nm = m;
        if (nunitq) {
            if (smallm_effective) {
                i32 nc = ldwork / n;
                for (i32 j = 0; j < m; j += nc) {
                    i32 bl = (m - j < nc) ? (m - j) : nc;
                    f64 alpha = one, beta = zero;
                    SLC_DGEMM("T", "N", &n, &bl, &n, &alpha, q, &ldq, &b[j*ldb], &ldb, &beta, dwork, &n);
                    SLC_DLACPY("A", &n, &bl, dwork, &n, &b[j*ldb], &ldb);
                }
            } else {
                nm = n;
                i32 ldw_avail = ldwork - jwork;
                SLC_DGERQF(&n, &m, b, &ldb, &dwork[itau], &dwork[jwork], &ldw_avail, &ifail);
                wrkopt = (wrkopt > (i32)dwork[jwork] + jwork) ? wrkopt : ((i32)dwork[jwork] + jwork);
                wrkopt = (wrkopt > minmn*n) ? wrkopt : minmn*n;

                if (ldwork >= minmn*n) {
                    i32 jj = 0;
                    for (i32 i = 0; i < minmn; i++) {
                        i32 inc = 1;
                        SLC_DCOPY(&n, &q[n - minmn + i], &ldq, &dwork[jj], &inc);
                        jj += n;
                    }
                    SLC_DTRMM("R", "U", "N", "N", &n, &minmn, &one, &b[n - minmn + (m - minmn)*ldb], &ldb, dwork, &n);
                    SLC_DLACPY("F", &n, &minmn, dwork, &n, b, &ldb);
                } else {
                    for (i32 j = 0; j < minmn; j++) {
                        i32 jp1 = j + 1;
                        i32 inc = 1;
                        SLC_DCOPY(&jp1, &b[(m - minmn + j)*ldb], &inc, dwork, &inc);
                        f64 alpha = one, beta = zero;
                        SLC_DGEMV("T", &jp1, &n, &alpha, q, &ldq, dwork, &inc, &beta, &b[j*ldb], &inc);
                    }
                }
            }
        }

        i32 ldw_avail = ldwork - jwork;
        SLC_DGERQF(&n, &nm, b, &ldb, &dwork[itau], &dwork[jwork], &ldw_avail, &ifail);
        
        if (n > nm) {
            if (lbscl && !scalb) {
                i32 kl = 0, ku = 0, info_scl;
                i32 nrows = n - m;
                SLC_DLASCL("G", &kl, &ku, &mb, &mbto, &nrows, &m, b, &ldb, &info_scl);
                SLC_DLASCL("U", &kl, &ku, &mb, &mbto, &m, &m, &b[(n - m)*ldb], &ldb, &info_scl);
            }
            for (i32 i = m - 1; i >= 0; i--) {
                i32 len = n - m + i + 1;
                i32 inc = 1;
                SLC_DCOPY(&len, &b[i*ldb], &inc, &b[(n - m + i)*ldb], &inc);
            }
            i32 nm_diff = n - m;
            SLC_DLASET("F", &n, &nm_diff, &zero, &zero, b, &ldb);
            if (m > 1) {
                i32 m1 = m - 1;
                SLC_DLASET("L", &m1, &m1, &zero, &zero, &b[(n - m + 1) + (n - m)*ldb], &ldb);
            }
        } else {
            if (m > n && m == nm) {
                SLC_DLACPY("U", &n, &n, &b[(m - n)*ldb], &ldb, b, &ldb);
            }
            if (lbscl && !scalb) {
                i32 kl = 0, ku = 0, info_scl;
                SLC_DLASCL("U", &kl, &ku, &mb, &mbto, &n, &n, b, &ldb, &info_scl);
            }
        }

        for (i32 i = n - minmn; i < n; i++) {
            if (b[i + i*ldb] < zero) {
                i32 len = i + 1;
                i32 inc = 1;
                f64 neg_one = -one;
                SLC_DSCAL(&len, &neg_one, &b[i*ldb], &inc);
            }
        }
    } else {
        nm = m;
        if (nunitq) {
            if (smallm_effective) {
                i32 nr = ldwork / n;
                for (i32 i = 0; i < m; i += nr) {
                    i32 bl = (m - i < nr) ? (m - i) : nr;
                    f64 alpha = one, beta = zero;
                    SLC_DGEMM("N", "N", &bl, &n, &n, &alpha, &b[i], &ldb, q, &ldq, &beta, dwork, &bl);
                    SLC_DLACPY("A", &bl, &n, dwork, &bl, &b[i], &ldb);
                }
            } else {
                i32 ldw_avail = ldwork - jwork;
                SLC_DGEQRF(&m, &n, b, &ldb, &dwork[itau], &dwork[jwork], &ldw_avail, &ifail);
                wrkopt = (wrkopt > (i32)dwork[jwork] + jwork) ? wrkopt : ((i32)dwork[jwork] + jwork);
                wrkopt = (wrkopt > n*n) ? wrkopt : n*n;

                if (ldwork >= n*n) {
                    SLC_DLACPY("F", &n, &n, q, &ldq, dwork, &n);
                    SLC_DTRMM("L", "U", "N", "N", &n, &n, &one, b, &ldb, dwork, &n);
                    SLC_DLACPY("F", &n, &n, dwork, &minmn, b, &ldb);
                } else {
                    i32 info_uy;
                    mb01uy("L", "U", "N", n, n, one, b, ldb, q, ldq, dwork, ldwork, &info_uy);
                }
                nm = n;
            }
        }

        i32 ldw_avail = ldwork - jwork;
        SLC_DGEQRF(&nm, &n, b, &ldb, &dwork[itau], &dwork[jwork], &ldw_avail, &ifail);
        
        if (lbscl && !scalb) {
            i32 kl = 0, ku = 0, info_scl;
            SLC_DLASCL("U", &kl, &ku, &mb, &mbto, &nm, &n, b, &ldb, &info_scl);
        }

        if (m < n) {
            i32 nmm = n - m;
            SLC_DLASET("U", &nmm, &nmm, &zero, &zero, &b[m + m*ldb], &ldb);
        }

        for (i32 i = 0; i < minmn; i++) {
            if (b[i + i*ldb] < zero) {
                i32 len = n - i;
                f64 neg_one = -one;
                SLC_DSCAL(&len, &neg_one, &b[i + i*ldb], &ldb);
            }
        }
    }
    
    if (minmn > 1) {
        i32 m1 = minmn - 1;
        SLC_DLASET("L", &m1, &m1, &zero, &zero, &b[1], &ldb);
    }

    f64 scaloc;
    sb03ot(!cont, istran, n, a, lda, b, ldb, &scaloc, dwork, info);
    *scale *= scaloc;

    if (istran) {
        if (nunitq) {
            i32 info_uy;
            mb01uy("R", "U", "N", n, n, one, b, ldb, q, ldq, dwork, ldwork, &info_uy);

            i32 ldw_avail = ldwork - n;
            SLC_DGERQF(&n, &n, b, &ldb, dwork, &dwork[n], &ldw_avail, &ifail);
            
            if (n > 1) {
                i32 n1 = n - 1;
                SLC_DLASET("L", &n1, &n1, &zero, &zero, &b[1], &ldb);
            }

            for (i32 i = 0; i < n; i++) {
                if (b[i + i*ldb] < zero) {
                    i32 len = i + 1;
                    i32 inc = 1;
                    f64 neg_one = -one;
                    SLC_DSCAL(&len, &neg_one, &b[i*ldb], &inc);
                }
            }
        }
    } else {
        if (nunitq) {
            i32 info_uy;
            mb01uy("R", "U", "T", n, n, one, b, ldb, q, ldq, dwork, ldwork, &info_uy);

            for (i32 i = 0; i < n; i++) {
                i32 len = i + 1;
                i32 inc = 1;
                SLC_DSWAP(&len, &b[i], &ldb, &b[i*ldb], &inc);
            }

            i32 ldw_avail = ldwork - n;
            SLC_DGEQRF(&n, &n, b, &ldb, dwork, &dwork[n], &ldw_avail, &ifail);
            
            if (n > 1) {
                i32 n1 = n - 1;
                SLC_DLASET("L", &n1, &n1, &zero, &zero, &b[1], &ldb);
            }

            for (i32 i = 0; i < n; i++) {
                if (b[i + i*ldb] < zero) {
                    i32 len = n - i;
                    f64 neg_one = -one;
                    SLC_DSCAL(&len, &neg_one, &b[i + i*ldb], &ldb);
                }
            }
        }
    }

    f64 tmp = one;
    if (lascl) {
        i32 kl = 0, ku = 0, info_scl;
        SLC_DLASCL("H", &kl, &ku, &mato, &ma, &n, &n, a, &lda, &info_scl);
        tmp = sqrt(mato / ma);
    }
    
    if (lbscl) {
        f64 mx_local = SLC_DLANTR("M", "U", "N", &n, &n, b, &ldb, dwork);
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
        SLC_DLASCL("U", &kl, &ku, &mbto, &tmp, &n, &n, b, &ldb, &info_scl);
    }

    dwork[0] = (f64)wrkopt;
}
