// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <math.h>

void mb04vd(const char *mode, const char *jobq, const char *jobz,
            i32 m, i32 n, i32 ranke,
            f64 *a, i32 lda, f64 *e, i32 lde,
            f64 *q, i32 ldq, f64 *z, i32 ldz,
            i32 *istair, i32 *nblcks, i32 *nblcki,
            i32 *imuk, i32 *inuk, i32 *imuk0, i32 *mnei,
            f64 tol, i32 *iwork, i32 *info) {
    const f64 zero = 0.0;
    const f64 one = 1.0;

    *info = 0;
    bool lmodeb = (mode[0] == 'B' || mode[0] == 'b');
    bool lmodet = (mode[0] == 'T' || mode[0] == 't');
    bool lmodes = (mode[0] == 'S' || mode[0] == 's');
    bool ljobqi = (jobq[0] == 'I' || jobq[0] == 'i');
    bool updatq = ljobqi || (jobq[0] == 'U' || jobq[0] == 'u');
    bool ljobzi = (jobz[0] == 'I' || jobz[0] == 'i');
    bool updatz = ljobzi || (jobz[0] == 'U' || jobz[0] == 'u');

    // Validate parameters
    if (!lmodeb && !lmodet && !lmodes) {
        *info = -1;
        return;
    }
    if (!updatq && !(jobq[0] == 'N' || jobq[0] == 'n')) {
        *info = -2;
        return;
    }
    if (!updatz && !(jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -3;
        return;
    }
    if (m < 0) {
        *info = -4;
        return;
    }
    if (n < 0) {
        *info = -5;
        return;
    }
    if (ranke < 0) {
        *info = -6;
        return;
    }
    i32 max1m = (1 > m) ? 1 : m;
    if (lda < max1m) {
        *info = -8;
        return;
    }
    if (lde < max1m) {
        *info = -10;
        return;
    }
    if ((!updatq && ldq < 1) || (updatq && ldq < max1m)) {
        *info = -12;
        return;
    }
    i32 max1n = (1 > n) ? 1 : n;
    if ((!updatz && ldz < 1) || (updatz && ldz < max1n)) {
        *info = -14;
        return;
    }

    // Initialize Q and Z to identity if requested
    if (ljobqi && m > 0) {
        SLC_DLASET("Full", &m, &m, &zero, &one, q, &ldq);
    }
    if (ljobzi && n > 0) {
        SLC_DLASET("Full", &n, &n, &zero, &one, z, &ldz);
    }

    // Quick return if possible
    *nblcks = 0;
    *nblcki = 0;

    if (n == 0) {
        mnei[0] = 0;
        mnei[1] = 0;
        mnei[2] = 0;
        return;
    }

    if (m == 0) {
        *nblcks = 1;
        imuk[0] = n;
        inuk[0] = 0;
        mnei[0] = 0;
        mnei[1] = n;
        mnei[2] = 0;
        return;
    }

    // Compute tolerance if not provided
    f64 toler = tol;
    if (toler <= zero) {
        f64 eps = SLC_DLAMCH("Epsilon");
        f64 anorm = SLC_DLANGE("M", &m, &n, a, &lda, NULL);
        f64 enorm = SLC_DLANGE("M", &m, &n, e, &lde, NULL);
        f64 maxnorm = (anorm > enorm) ? anorm : enorm;
        toler = eps * maxnorm;
    }

    // Initialize indices
    // IFIRA, IFICA: first row and column index of A(k) in A (1-based)
    // NCA: number of columns in A(k)
    i32 ifira = 1;
    i32 ifica = 1;
    i32 nca = n - ranke;
    i32 isnuk = 0;
    i32 ismuk = 0;
    i32 k = 0;

    // Initialize INUK and IMUK arrays
    for (i32 i = 0; i < m + 1; i++) {
        if (i < m + 1) inuk[i] = -1;
    }
    for (i32 i = 0; i < n; i++) {
        imuk[i] = -1;
    }

    // Compress rows of A while keeping E in column echelon form
    do {
        i32 ranka;
        mb04tt(updatq, updatz, m, n, ifira, ifica, nca,
               a, lda, e, lde, q, ldq, z, ldz, istair, &ranka, toler, iwork);

        imuk[k] = nca;
        ismuk += nca;

        inuk[k] = ranka;
        isnuk += ranka;
        (*nblcks)++;

        // If rank(A(k)) = nra then A has full row rank
        // JK = first column index (in A) after the rightmost column of A(k+1)
        ifira = 1 + isnuk;
        ifica = 1 + ismuk;

        i32 jk;
        if (ifira > m) {
            jk = n + 1;
        } else {
            jk = abs(istair[ifira - 1]);
        }
        nca = jk - 1 - ismuk;

        k++;
    } while (nca > 0);

    // Store dimensions of sE(eps,inf)-A(eps,inf)
    mnei[0] = isnuk;
    mnei[1] = ismuk;
    mnei[2] = 0;

    if (lmodeb) {
        return;
    }

    // Triangularization of submatrices in A and E
    mb04ty(updatq, updatz, m, n, *nblcks, inuk, imuk,
           a, lda, e, lde, q, ldq, z, ldz, info);

    if (*info > 0 || lmodet) {
        return;
    }

    // Save row dimensions of diagonal submatrices in pencil sE(eps,inf)-A(eps,inf)
    for (i32 i = 0; i < *nblcks; i++) {
        imuk0[i] = inuk[i];
    }

    // Reduction to square submatrices E(k)'s in E
    mb04vx(updatq, updatz, m, n, *nblcks, inuk, imuk,
           a, lda, e, lde, q, ldq, z, ldz, mnei);

    // Determine dimensions of infinite diagonal submatrices
    // and update block numbers if necessary
    bool first = true;
    bool firsti = true;
    *nblcki = *nblcks;
    k = *nblcks;

    for (i32 i = k - 1; i >= 0; i--) {
        imuk0[i] = imuk0[i] - inuk[i];
        if (firsti && imuk0[i] == 0) {
            (*nblcki)--;
        } else {
            firsti = false;
        }
        if (first && imuk[i] == 0) {
            (*nblcks)--;
        } else {
            first = false;
        }
    }
}
