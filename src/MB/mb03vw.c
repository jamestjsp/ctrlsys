/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

void mb03vw(
    const char *compq,
    const i32 *qind,
    const char *triu,
    const i32 n,
    const i32 k,
    i32 *h,
    const i32 ilo,
    const i32 ihi,
    const i32 *s,
    f64 *a,
    const i32 lda1,
    const i32 lda2,
    f64 *q,
    const i32 ldq1,
    const i32 ldq2,
    i32 *iwork,
    const i32 liwork,
    f64 *dwork,
    const i32 ldwork,
    i32 *info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    bool liniq = (compq[0] == 'I' || compq[0] == 'i');
    bool lcmpq = (compq[0] == 'U' || compq[0] == 'u') || liniq;
    bool lparq = (compq[0] == 'P' || compq[0] == 'p');
    bool alltri = (triu[0] == 'A' || triu[0] == 'a');
    bool lquery = (ldwork == -1);

    *info = 0;

    // Parameter validation
    if (!lcmpq && !lparq && !(compq[0] == 'N' || compq[0] == 'n')) {
        *info = -1;
    } else if (!alltri && !(triu[0] == 'N' || triu[0] == 'n')) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (k < 0) {
        *info = -5;
    } else if (ilo < 1) {
        *info = -7;
    } else if (ihi > n || (n > 0 && ihi < ilo) || ihi < 0) {
        *info = -8;
    } else if (lda1 < (n > 1 ? n : 1)) {
        *info = -11;
    } else if (lda2 < (n > 1 ? n : 1)) {
        *info = -12;
    } else if (ldq1 < 1 || ((lcmpq || lparq) && ldq1 < n)) {
        *info = -14;
    } else if (ldq2 < 1 || ((lcmpq || lparq) && ldq2 < n)) {
        *info = -15;
    } else if (liwork < (k > 1 ? 3 * k : 1)) {
        *info = -17;
        if (iwork != NULL) {
            iwork[0] = (k > 1 ? 3 * k : 1);
        }
    } else {
        i32 minwrk;
        if (n < 1 || k < 1 || n == 1 || ilo == ihi) {
            minwrk = 1;
        } else {
            i32 m = ihi - ilo + 1;
            i32 maxval = (ihi > n - ilo + 1) ? ihi : (n - ilo + 1);
            minwrk = m + maxval;
        }

        i32 optwrk = minwrk;

        if (!lquery && ldwork < minwrk) {
            *info = -19;
            dwork[0] = (f64)minwrk;
        } else if (n > 2) {
            // Compute optimal workspace
            i32 m = ihi - ilo + 1;
            i32 ier;
            f64 dum[3];
            i32 neg1 = -1;

            SLC_DGEQRF(&m, &m, a, &lda1, dum, &dum[0], &neg1, &ier);
            optwrk = (i32)dum[0];
            SLC_DGERQF(&m, &m, a, &lda1, dum, &dum[1], &neg1, &ier);
            if ((i32)dum[1] > optwrk) optwrk = (i32)dum[1];

            if (ihi < n) {
                i32 nmihi = n - ihi;
                SLC_DORMQR("Left", "Trans", &m, &nmihi, &m, a, &lda1, dum,
                           a, &lda1, dum, &neg1, &ier);
                if ((i32)dum[0] > optwrk) optwrk = (i32)dum[0];
            }

            SLC_DORMQR("Right", "NoTran", &ihi, &m, &m, a, &lda1, dum,
                       a, &lda1, dum, &neg1, &ier);
            if ((i32)dum[0] > optwrk) optwrk = (i32)dum[0];

            i32 nmilop1 = n - ilo + 1;
            SLC_DORMQR("Left", "Trans", &m, &nmilop1, &m, a, &lda1, dum,
                       a, &lda1, &dum[1], &neg1, &ier);
            if ((i32)dum[0] > optwrk) optwrk = (i32)dum[0];
            if ((i32)dum[1] > optwrk) optwrk = (i32)dum[1];

            SLC_DORMRQ("Right", "Trans", &ihi, &m, &m, a, &lda1, dum,
                       a, &lda1, dum, &neg1, &ier);
            if ((i32)dum[0] > optwrk) optwrk = (i32)dum[0];

            SLC_DORMRQ("Left", "NoTran", &m, &nmilop1, &m, a, &lda1, dum,
                       a, &lda1, &dum[1], &neg1, &ier);
            if ((i32)dum[0] > optwrk) optwrk = (i32)dum[0];
            if ((i32)dum[1] > optwrk) optwrk = (i32)dum[1];

            if (liniq) {
                SLC_DORGQR(&m, &m, &m, q, &ldq1, dum, dum, &neg1, &ier);
                if ((i32)dum[0] > optwrk) optwrk = (i32)dum[0];
            }

            if (minwrk > m + optwrk) {
                optwrk = minwrk;
            } else {
                optwrk = m + optwrk;
            }
        }

        if (lquery) {
            dwork[0] = (f64)optwrk;
            return;
        }
    }

    if (*info != 0) {
        return;
    }

    // Set H if not in proper interval
    if (k == 0) {
        *h = 0;
    } else if (*h < 1 || *h > k) {
        *h = 1;
    }

    // Quick return if possible
    if (n < 1 || k < 1) {
        dwork[0] = ONE;
        return;
    }

    // Initialize Q if needed
    for (i32 i = 0; i < k; i++) {
        i32 j = 0;
        if (liniq) {
            j = i + 1;
        } else if (lparq && qind != NULL) {
            j = -qind[i];
        }
        if (j > 0) {
            i32 j0 = j - 1;
            SLC_DLASET("Full", &n, &n, &ZERO, &ONE, &q[j0 * ldq1 * ldq2], &ldq1);
        }
    }

    // If all factors already triangular, return
    if (ilo == ihi) {
        dwork[0] = ONE;
        return;
    }

    // Compute maps for accessing A and Q using mb03ba
    i32 mapa = k;
    i32 mapq = 2 * k;
    i32 smult;
    mb03ba(k, *h, s, &smult, &iwork[mapa], &iwork[mapq]);

    // Compute set of matrices with signature == smult
    for (i32 i = 0; i < k; i++) {
        i32 aind = iwork[mapa + i];
        if (s[aind - 1] != smult) {
            iwork[aind - 1] = 0;
        } else if (alltri && aind != *h) {
            iwork[aind - 1] = 0;
        } else {
            iwork[aind - 1] = 1;
        }
    }

    // Find maximal element in this set
    i32 maxset = 0;
    for (i32 i = k - 1; i >= 0; i--) {
        i32 aind = iwork[mapa + i];
        if (maxset == 0 && iwork[aind - 1] == 1) {
            maxset = i + 1;
        }
    }

    // Transform matrices not in set to upper triangular form
    i32 i2 = (n < ilo + 1) ? n : (ilo + 1);
    i32 i3 = (n < ihi + 1) ? n : (ihi + 1);
    i32 m = ihi - ilo + 1;
    i32 iwrk = m;
    i32 wmax = ldwork - m;
    bool lindq = false;

    for (i32 i = k - 1; i >= 0; i--) {
        i32 aind = iwork[mapa + i];

        if (iwork[aind - 1] == 0) {
            i32 indq = iwork[mapq + i];
            if (lparq && qind != NULL) {
                i32 qv = qind[indq - 1];
                indq = (qv >= 0) ? qv : -qv;
                lindq = (indq > 0);
            }

            i32 iprev = (i == 0) ? k - 1 : i - 1;
            i32 aindp = iwork[mapa + iprev];

            f64 *a_aind = &a[(aind - 1) * lda1 * lda2];
            f64 *a_aindp = &a[(aindp - 1) * lda1 * lda2];
            f64 *q_indq = (indq > 0 && q != NULL) ? &q[(indq - 1) * ldq1 * ldq2] : NULL;

            i32 ier;

            if (s[aind - 1] == smult) {
                // QR decomposition of A_AIND
                SLC_DGEQRF(&m, &m, &a_aind[(ilo - 1) + (ilo - 1) * lda1], &lda1,
                           dwork, &dwork[iwrk], &wmax, &ier);

                // Update rows ILO:IHI in columns IHI+1:N of A_AIND
                if (ihi < n) {
                    i32 nmihi = n - ihi;
                    SLC_DORMQR("Left", "Trans", &m, &nmihi, &m,
                               &a_aind[(ilo - 1) + (ilo - 1) * lda1], &lda1, dwork,
                               &a_aind[(ilo - 1) + ihi * lda1], &lda1,
                               &dwork[iwrk], &wmax, &ier);
                }

                // Update A_AINDP
                if (s[aindp - 1] == smult) {
                    SLC_DORMQR("Right", "NoTran", &ihi, &m, &m,
                               &a_aind[(ilo - 1) + (ilo - 1) * lda1], &lda1, dwork,
                               &a_aindp[0 + (ilo - 1) * lda1], &lda1,
                               &dwork[iwrk], &wmax, &ier);
                } else {
                    i32 nmilop1 = n - ilo + 1;
                    SLC_DORMQR("Left", "Trans", &m, &nmilop1, &m,
                               &a_aind[(ilo - 1) + (ilo - 1) * lda1], &lda1, dwork,
                               &a_aindp[(ilo - 1) + (ilo - 1) * lda1], &lda1,
                               &dwork[iwrk], &wmax, &ier);
                }

                // Update transformation matrix
                if (liniq && q_indq != NULL) {
                    i32 ilom1 = ilo - 1;
                    i32 nmilop1 = n - ilo + 1;
                    i32 nmihi = n - ihi;
                    i32 mm1 = m - 1;

                    SLC_DLASET("Full", &n, &ilom1, &ZERO, &ONE, q_indq, &ldq1);
                    SLC_DLASET("Full", &ilom1, &nmilop1, &ZERO, &ZERO, &q_indq[(ilo - 1) * ldq1], &ldq1);

                    if (mm1 > 0) {
                        SLC_DLACPY("Lower", &mm1, &mm1, &a_aind[(i2 - 1) + (ilo - 1) * lda1], &lda1,
                                   &q_indq[(i2 - 1) + (ilo - 1) * ldq1], &ldq1);
                    }

                    SLC_DORGQR(&m, &m, &m, &q_indq[(ilo - 1) + (ilo - 1) * ldq1], &ldq1,
                               dwork, &dwork[iwrk], &wmax, &ier);

                    SLC_DLASET("Full", &nmihi, &ihi, &ZERO, &ZERO, &q_indq[(i3 - 1) + (ilo - 1) * ldq1], &ldq1);
                    SLC_DLASET("Full", &ihi, &nmihi, &ZERO, &ZERO, &q_indq[0 + (i3 - 1) * ldq1], &ldq1);
                    SLC_DLASET("Full", &nmihi, &nmihi, &ZERO, &ONE, &q_indq[(i3 - 1) + (i3 - 1) * ldq1], &ldq1);
                } else if ((lcmpq || lindq) && q_indq != NULL) {
                    SLC_DORMQR("Right", "NoTran", &ihi, &m, &m,
                               &a_aind[(ilo - 1) + (ilo - 1) * lda1], &lda1, dwork,
                               &q_indq[0 + (ilo - 1) * ldq1], &ldq1,
                               &dwork[iwrk], &wmax, &ier);
                }
            } else {
                // RQ decomposition of A_AIND
                SLC_DGERQF(&m, &m, &a_aind[(ilo - 1) + (ilo - 1) * lda1], &lda1,
                           dwork, &dwork[iwrk], &wmax, &ier);

                // Update rows 1:ILO-1 in columns ILO:IHI of A_AIND
                if (ilo > 1) {
                    i32 ilom1 = ilo - 1;
                    SLC_DORMRQ("Right", "Trans", &ilom1, &m, &m,
                               &a_aind[(ilo - 1) + (ilo - 1) * lda1], &lda1, dwork,
                               &a_aind[0 + (ilo - 1) * lda1], &lda1,
                               &dwork[iwrk], &wmax, &ier);
                }

                // Update A_AINDP
                if (s[aindp - 1] == smult) {
                    SLC_DORMRQ("Right", "Trans", &ihi, &m, &m,
                               &a_aind[(ilo - 1) + (ilo - 1) * lda1], &lda1, dwork,
                               &a_aindp[0 + (ilo - 1) * lda1], &lda1,
                               &dwork[iwrk], &wmax, &ier);
                } else {
                    i32 nmilop1 = n - ilo + 1;
                    SLC_DORMRQ("Left", "NoTran", &m, &nmilop1, &m,
                               &a_aind[(ilo - 1) + (ilo - 1) * lda1], &lda1, dwork,
                               &a_aindp[(ilo - 1) + (ilo - 1) * lda1], &lda1,
                               &dwork[iwrk], &wmax, &ier);
                }

                // Update transformation matrix
                if ((lcmpq || lindq) && q_indq != NULL) {
                    SLC_DORMRQ("Right", "Trans", &ihi, &m, &m,
                               &a_aind[(ilo - 1) + (ilo - 1) * lda1], &lda1, dwork,
                               &q_indq[0 + (ilo - 1) * ldq1], &ldq1,
                               &dwork[iwrk], &wmax, &ier);
                }
            }

            // Zero out lower triangular part
            if (m > 1) {
                i32 mm1 = m - 1;
                SLC_DLASET("Lower", &mm1, &mm1, &ZERO, &ZERO, &a_aind[(i2 - 1) + (ilo - 1) * lda1], &lda1);
            }
        }
    }

    // Reduce A_1 to upper Hessenberg form
    lindq = false;
    i32 int1 = 1;

    for (i32 j = ilo - 1; j < ihi - 1; j++) {
        i32 upidx = j;

        for (i32 lt = k + maxset - 1; lt >= maxset; lt--) {
            i32 l = lt;
            if (l >= k) l = l - k;
            if (l == 0) upidx = j + 1;

            if (upidx < n - 1) {
                i32 aind = iwork[mapa + l];
                i32 indq = iwork[mapq + l];

                if (lparq && qind != NULL) {
                    i32 qv = qind[indq - 1];
                    indq = (qv >= 0) ? qv : -qv;
                    lindq = (indq > 0);
                }

                i32 aindp;
                if (l == 0) {
                    aindp = iwork[mapa + k - 1];
                } else {
                    aindp = iwork[mapa + l - 1];
                }

                f64 *a_aind = &a[(aind - 1) * lda1 * lda2];
                f64 *a_aindp = &a[(aindp - 1) * lda1 * lda2];
                f64 *q_indq = (indq > 0 && q != NULL) ? &q[(indq - 1) * ldq1 * ldq2] : NULL;

                if (iwork[aind - 1] == 1 && iwork[aindp - 1] == 1) {
                    // Case 1: Both in set - use Householder
                    i32 ihimupidx = ihi - upidx;
                    f64 tau;
                    SLC_DLARFG(&ihimupidx, &a_aind[upidx + j * lda1],
                               &a_aind[upidx + 1 + j * lda1], &int1, &tau);
                    f64 alpha = a_aind[upidx + j * lda1];
                    a_aind[upidx + j * lda1] = ONE;

                    // Apply transformations
                    i32 nmj = n - j - 1;
                    i32 nmupidx = n - upidx;
                    SLC_DLARF("Left", &ihimupidx, &nmj, &a_aind[upidx + j * lda1],
                              &int1, &tau, &a_aind[upidx + (j + 1) * lda1], &lda1, dwork);
                    SLC_DLARF("Right", &ihi, &nmupidx, &a_aind[upidx + j * lda1],
                              &int1, &tau, &a_aindp[0 + upidx * lda1], &lda1, dwork);

                    if ((lcmpq || lindq) && q_indq != NULL) {
                        SLC_DLARF("Right", &n, &nmupidx, &a_aind[upidx + j * lda1],
                                  &int1, &tau, &q_indq[0 + upidx * ldq1], &ldq1, dwork);
                    }

                    a_aind[upidx + j * lda1] = alpha;

                    // Zero out elements
                    for (i32 ii = upidx + 1; ii < ihi; ii++) {
                        a_aind[ii + j * lda1] = ZERO;
                    }

                } else if (iwork[aind - 1] == 1) {
                    // Case 2: Only AIND in set - use Givens
                    i32 pos = 0;

                    for (i32 ii = ihi - 1; ii > upidx; ii--) {
                        f64 temp = a_aind[ii - 1 + j * lda1];
                        SLC_DLARTG(&temp, &a_aind[ii + j * lda1],
                                   &dwork[pos], &dwork[pos + 1], &a_aind[ii - 1 + j * lda1]);
                        a_aind[ii + j * lda1] = ZERO;

                        i32 nmj = n - j - 1;
                        SLC_DROT(&nmj, &a_aind[ii - 1 + (j + 1) * lda1], &lda1,
                                 &a_aind[ii + (j + 1) * lda1], &lda1,
                                 &dwork[pos], &dwork[pos + 1]);
                        pos += 2;
                    }

                    if ((lcmpq || lindq) && q_indq != NULL) {
                        pos = 0;
                        for (i32 ii = ihi - 1; ii > upidx; ii--) {
                            SLC_DROT(&n, &q_indq[0 + (ii - 1) * ldq1], &int1,
                                     &q_indq[0 + ii * ldq1], &int1,
                                     &dwork[pos], &dwork[pos + 1]);
                            pos += 2;
                        }
                    }

                } else {
                    // Case 3: Neither in set - propagate rotations
                    i32 pos = 0;

                    if (s[aind - 1] == smult) {
                        for (i32 ii = ihi - 1; ii > upidx; ii--) {
                            i32 ip1 = ii + 1;
                            SLC_DROT(&ip1, &a_aind[0 + (ii - 1) * lda1], &int1,
                                     &a_aind[0 + ii * lda1], &int1,
                                     &dwork[pos], &dwork[pos + 1]);

                            f64 temp = a_aind[ii - 1 + (ii - 1) * lda1];
                            SLC_DLARTG(&temp, &a_aind[ii + (ii - 1) * lda1],
                                       &dwork[pos], &dwork[pos + 1],
                                       &a_aind[ii - 1 + (ii - 1) * lda1]);
                            a_aind[ii + (ii - 1) * lda1] = ZERO;

                            i32 nmip1 = n - ii;
                            SLC_DROT(&nmip1, &a_aind[ii - 1 + ii * lda1], &lda1,
                                     &a_aind[ii + ii * lda1], &lda1,
                                     &dwork[pos], &dwork[pos + 1]);
                            pos += 2;
                        }
                    } else {
                        for (i32 ii = ihi - 1; ii > upidx; ii--) {
                            i32 nmip2 = n - ii + 1;
                            SLC_DROT(&nmip2, &a_aind[ii - 1 + (ii - 1) * lda1], &lda1,
                                     &a_aind[ii + (ii - 1) * lda1], &lda1,
                                     &dwork[pos], &dwork[pos + 1]);

                            f64 temp = a_aind[ii + ii * lda1];
                            f64 negval = -a_aind[ii + (ii - 1) * lda1];
                            SLC_DLARTG(&temp, &negval,
                                       &dwork[pos], &dwork[pos + 1],
                                       &a_aind[ii + ii * lda1]);
                            a_aind[ii + (ii - 1) * lda1] = ZERO;

                            i32 im1 = ii;
                            SLC_DROT(&im1, &a_aind[0 + (ii - 1) * lda1], &int1,
                                     &a_aind[0 + ii * lda1], &int1,
                                     &dwork[pos], &dwork[pos + 1]);
                            pos += 2;
                        }
                    }

                    // Update transformation matrix
                    if ((lcmpq || lindq) && q_indq != NULL) {
                        pos = 0;
                        for (i32 ii = ihi - 1; ii > upidx; ii--) {
                            SLC_DROT(&n, &q_indq[0 + (ii - 1) * ldq1], &int1,
                                     &q_indq[0 + ii * ldq1], &int1,
                                     &dwork[pos], &dwork[pos + 1]);
                            pos += 2;
                        }
                    }

                    // If AINDP in set, apply rotations
                    if (iwork[aindp - 1] == 1) {
                        pos = 0;
                        for (i32 ii = ihi - 1; ii > upidx; ii--) {
                            SLC_DROT(&ihi, &a_aindp[0 + (ii - 1) * lda1], &int1,
                                     &a_aindp[0 + ii * lda1], &int1,
                                     &dwork[pos], &dwork[pos + 1]);
                            pos += 2;
                        }
                    }
                }
            }
        }
    }

    dwork[0] = (f64)(m + wmax);
}
