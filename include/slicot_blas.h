/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#ifndef SLICOT_BLAS_H
#define SLICOT_BLAS_H

#include "slicot_types.h"
#include <stdbool.h>

/**
 * @file slicot_blas.h
 * @brief BLAS/LAPACK function declarations with portable symbol naming
 *
 * This header provides a unified interface to BLAS/LAPACK routines that
 * automatically handles different symbol naming conventions:
 * - Lowercase with underscore: dgemm_ (most common)
 * - Lowercase only: dgemm (some systems)
 * - Uppercase: DGEMM (rare)
 *
 * The SLC_FC_* macro is defined by CMake during configuration based on
 * probing the BLAS/LAPACK library.
 *
 * Based on SLICUTLET's approach to BLAS/LAPACK integration.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Symbol mangling macros - defined by CMake configuration */
#ifndef SLC_FC_FUNC
  #if defined(SLC_FC_LOWER_US)
    #define SLC_FC_FUNC(lc, UC) lc##_
  #elif defined(SLC_FC_LOWER)
    #define SLC_FC_FUNC(lc, UC) lc
  #elif defined(SLC_FC_UPPER)
    #define SLC_FC_FUNC(lc, UC) UC
  #else
    /* Default: lowercase with underscore (most common) */
    #define SLC_FC_FUNC(lc, UC) lc##_
  #endif
#endif

/* Integer type for BLAS/LAPACK - future ILP64 support */
#ifdef SLICOT_ILP64
  typedef i64 sl_int;
#else
  typedef i32 sl_int;
#endif

/*
 * BLAS/LAPACK function prototypes
 * Using the "evil hack" from SLICUTLET: temporarily redefine 'int' to 'sl_int'
 * for cleaner prototypes that match Fortran INTEGER parameters.
 */
#define int sl_int

/* BLAS Level 1 - Vector operations */
void SLC_FC_FUNC(dcopy, DCOPY)(const int* n, const f64* x, const int* incx,
                                f64* y, const int* incy);

void SLC_FC_FUNC(daxpy, DAXPY)(const int* n, const f64* alpha, const f64* x,
                                const int* incx, f64* y, const int* incy);

void SLC_FC_FUNC(dscal, DSCAL)(const int* n, const f64* alpha, f64* x,
                                const int* incx);

void SLC_FC_FUNC(drscl, DRSCL)(const int* n, const f64* sa, f64* sx,
                                const int* incx);

void SLC_FC_FUNC(dswap, DSWAP)(const int* n, f64* x, const int* incx,
                                f64* y, const int* incy);

f64 SLC_FC_FUNC(ddot, DDOT)(const int* n, const f64* x, const int* incx,
                             const f64* y, const int* incy);

f64 SLC_FC_FUNC(dnrm2, DNRM2)(const int* n, const f64* x, const int* incx);

int SLC_FC_FUNC(idamax, IDAMAX)(const int* n, const f64* x, const int* incx);

f64 SLC_FC_FUNC(dasum, DASUM)(const int* n, const f64* x, const int* incx);

/* BLAS Level 2 - Matrix-vector operations */
void SLC_FC_FUNC(dgemv, DGEMV)(const char* trans, const int* m, const int* n,
                                const f64* alpha, const f64* a, const int* lda,
                                const f64* x, const int* incx, const f64* beta,
                                f64* y, const int* incy);

void SLC_FC_FUNC(dtrmv, DTRMV)(const char* uplo, const char* trans, const char* diag,
                                const int* n, const f64* a, const int* lda,
                                f64* x, const int* incx);

void SLC_FC_FUNC(dtpmv, DTPMV)(const char* uplo, const char* trans, const char* diag,
                                const int* n, const f64* ap, f64* x, const int* incx);

void SLC_FC_FUNC(dtrsv, DTRSV)(const char* uplo, const char* trans, const char* diag,
                                const int* n, const f64* a, const int* lda,
                                f64* x, const int* incx);

void SLC_FC_FUNC(dger, DGER)(const int* m, const int* n, const f64* alpha,
                              const f64* x, const int* incx, const f64* y,
                              const int* incy, f64* a, const int* lda);

void SLC_FC_FUNC(dsymv, DSYMV)(const char* uplo, const int* n, const f64* alpha,
                                const f64* a, const int* lda, const f64* x,
                                const int* incx, const f64* beta, f64* y,
                                const int* incy);

void SLC_FC_FUNC(dsyr2, DSYR2)(const char* uplo, const int* n, const f64* alpha,
                                const f64* x, const int* incx, const f64* y,
                                const int* incy, f64* a, const int* lda);

/* BLAS Level 3 - Matrix-matrix operations */
void SLC_FC_FUNC(dtrmm, DTRMM)(const char* side, const char* uplo, const char* transa,
                                const char* diag, const int* m, const int* n,
                                const f64* alpha, const f64* a, const int* lda,
                                f64* b, const int* ldb);

void SLC_FC_FUNC(dtrsm, DTRSM)(const char* side, const char* uplo, const char* transa,
                                const char* diag, const int* m, const int* n,
                                const f64* alpha, const f64* a, const int* lda,
                                f64* b, const int* ldb);

void SLC_FC_FUNC(dgemm, DGEMM)(const char* transa, const char* transb,
                                const int* m, const int* n, const int* k,
                                const f64* alpha, const f64* a, const int* lda,
                                const f64* b, const int* ldb, const f64* beta,
                                f64* c, const int* ldc);

void SLC_FC_FUNC(dsyrk, DSYRK)(const char* uplo, const char* trans,
                                const int* n, const int* k, const f64* alpha,
                                const f64* a, const int* lda, const f64* beta,
                                f64* c, const int* ldc);

void SLC_FC_FUNC(dsyr2k, DSYR2K)(const char* uplo, const char* trans,
                                  const int* n, const int* k, const f64* alpha,
                                  const f64* a, const int* lda, const f64* b,
                                  const int* ldb, const f64* beta, f64* c,
                                  const int* ldc);

void SLC_FC_FUNC(dsymm, DSYMM)(const char* side, const char* uplo,
                                const int* m, const int* n, const f64* alpha,
                                const f64* a, const int* lda, const f64* b,
                                const int* ldb, const f64* beta, f64* c,
                                const int* ldc);

/* LAPACK - Utilities */
f64 SLC_FC_FUNC(dlamch, DLAMCH)(const char* cmach);

f64 SLC_FC_FUNC(dlamc3, DLAMC3)(const f64* a, const f64* b);

void SLC_FC_FUNC(dlacpy, DLACPY)(const char* uplo, const int* m, const int* n,
                                  const f64* a, const int* lda, f64* b,
                                  const int* ldb);

void SLC_FC_FUNC(dlapmt, DLAPMT)(const int* forwrd, const int* m, const int* n,
                                  f64* x, const int* ldx, int* k);

void SLC_FC_FUNC(dlarnv, DLARNV)(const int* idist, int* seed, const int* n,
                                  f64* x);

void SLC_FC_FUNC(dlaset, DLASET)(const char* uplo, const int* m, const int* n,
                                  const f64* alpha, const f64* beta, f64* a,
                                  const int* lda);

/* LAPACK - Factorizations */
void SLC_FC_FUNC(dgetrf, DGETRF)(const int* m, const int* n, f64* a,
                                  const int* lda, int* ipiv, int* info);

void SLC_FC_FUNC(dgetrs, DGETRS)(const char* trans, const int* n, const int* nrhs,
                                  const f64* a, const int* lda, const int* ipiv,
                                  f64* b, const int* ldb, int* info);

void SLC_FC_FUNC(dgetri, DGETRI)(const int* n, f64* a, const int* lda,
                                  const int* ipiv, f64* work, const int* lwork,
                                  int* info);

void SLC_FC_FUNC(dgesv, DGESV)(const int* n, const int* nrhs, f64* a,
                                const int* lda, int* ipiv, f64* b,
                                const int* ldb, int* info);

void SLC_FC_FUNC(dgesvx, DGESVX)(const char* fact, const char* trans,
                                  const int* n, const int* nrhs, f64* a,
                                  const int* lda, f64* af, const int* ldaf,
                                  int* ipiv, char* equed, f64* r, f64* c,
                                  f64* b, const int* ldb, f64* x, const int* ldx,
                                  f64* rcond, f64* ferr, f64* berr, f64* work,
                                  int* iwork, int* info);

void SLC_FC_FUNC(dgecon, DGECON)(const char* norm, const int* n, const f64* a,
                                  const int* lda, const f64* anorm, f64* rcond,
                                  f64* work, int* iwork, int* info);

void SLC_FC_FUNC(dgeequ, DGEEQU)(const int* m, const int* n, const f64* a,
                                  const int* lda, f64* r, f64* c,
                                  f64* rowcnd, f64* colcnd, f64* amax,
                                  int* info);

void SLC_FC_FUNC(dgerfs, DGERFS)(const char* trans, const int* n, const int* nrhs,
                                  const f64* a, const int* lda, const f64* af,
                                  const int* ldaf, const int* ipiv, const f64* b,
                                  const int* ldb, f64* x, const int* ldx,
                                  f64* ferr, f64* berr, f64* work, int* iwork,
                                  int* info);

void SLC_FC_FUNC(dlaqge, DLAQGE)(const int* m, const int* n, f64* a,
                                  const int* lda, const f64* r, const f64* c,
                                  const f64* rowcnd, const f64* colcnd,
                                  const f64* amax, char* equed);

void SLC_FC_FUNC(dpotrf, DPOTRF)(const char* uplo, const int* n, f64* a,
                                  const int* lda, int* info);

void SLC_FC_FUNC(dposv, DPOSV)(const char* uplo, const int* n, const int* nrhs,
                                f64* a, const int* lda, f64* b, const int* ldb,
                                int* info);

void SLC_FC_FUNC(dpptrf, DPPTRF)(const char* uplo, const int* n, f64* ap,
                                  int* info);

void SLC_FC_FUNC(dtrtri, DTRTRI)(const char* uplo, const char* diag, const int* n,
                                  f64* a, const int* lda, int* info);

void SLC_FC_FUNC(dsytrf, DSYTRF)(const char* uplo, const int* n, f64* a,
                                  const int* lda, int* ipiv, f64* work,
                                  const int* lwork, int* info);

void SLC_FC_FUNC(dsytri, DSYTRI)(const char* uplo, const int* n, f64* a,
                                  const int* lda, const int* ipiv, f64* work,
                                  int* info);

void SLC_FC_FUNC(dsysv, DSYSV)(const char* uplo, const int* n, const int* nrhs,
                                f64* a, const int* lda, int* ipiv, f64* b,
                                const int* ldb, f64* work, const int* lwork,
                                int* info);

void SLC_FC_FUNC(dpocon, DPOCON)(const char* uplo, const int* n, const f64* a,
                                  const int* lda, const f64* anorm, f64* rcond,
                                  f64* work, int* iwork, int* info);

void SLC_FC_FUNC(dsycon, DSYCON)(const char* uplo, const int* n, const f64* a,
                                  const int* lda, const int* ipiv, const f64* anorm,
                                  f64* rcond, f64* work, int* iwork, int* info);

void SLC_FC_FUNC(dsytrs, DSYTRS)(const char* uplo, const int* n, const int* nrhs,
                                  const f64* a, const int* lda, const int* ipiv,
                                  f64* b, const int* ldb, int* info);

void SLC_FC_FUNC(dpotrs, DPOTRS)(const char* uplo, const int* n, const int* nrhs,
                                  const f64* a, const int* lda,
                                  f64* b, const int* ldb, int* info);

void SLC_FC_FUNC(dpptrs, DPPTRS)(const char* uplo, const int* n, const int* nrhs,
                                  const f64* ap, f64* b, const int* ldb, int* info);

void SLC_FC_FUNC(dpptri, DPPTRI)(const char* uplo, const int* n, f64* ap, int* info);

void SLC_FC_FUNC(dpttrf, DPTTRF)(const int* n, f64* d, f64* e, int* info);

void SLC_FC_FUNC(dpttrs, DPTTRS)(const int* n, const int* nrhs, const f64* d,
                                  const f64* e, f64* b, const int* ldb, int* info);

void SLC_FC_FUNC(dspmv, DSPMV)(const char* uplo, const int* n, const f64* alpha,
                                const f64* ap, const f64* x, const int* incx,
                                const f64* beta, f64* y, const int* incy);

void SLC_FC_FUNC(dspr, DSPR)(const char* uplo, const int* n, const f64* alpha,
                              const f64* x, const int* incx, f64* ap);

void SLC_FC_FUNC(dtrcon, DTRCON)(const char* norm, const char* uplo,
                                  const char* diag, const int* n, const f64* a,
                                  const int* lda, f64* rcond, f64* work,
                                  int* iwork, int* info);

f64 SLC_FC_FUNC(dlansy, DLANSY)(const char* norm, const char* uplo, const int* n,
                                 const f64* a, const int* lda, f64* work);

void SLC_FC_FUNC(dgeqrf, DGEQRF)(const int* m, const int* n, f64* a,
                                  const int* lda, f64* tau, f64* work,
                                  const int* lwork, int* info);

void SLC_FC_FUNC(dgeqp3, DGEQP3)(const int* m, const int* n, f64* a,
                                  const int* lda, int* jpvt, f64* tau,
                                  f64* work, const int* lwork, int* info);

void SLC_FC_FUNC(dgerqf, DGERQF)(const int* m, const int* n, f64* a,
                                  const int* lda, f64* tau, f64* work,
                                  const int* lwork, int* info);

void SLC_FC_FUNC(dgerq2, DGERQ2)(const int* m, const int* n, f64* a,
                                  const int* lda, f64* tau, f64* work,
                                  int* info);

void SLC_FC_FUNC(dormr2, DORMR2)(const char* side, const char* trans,
                                  const int* m, const int* n, const int* k,
                                  const f64* a, const int* lda, const f64* tau,
                                  f64* c, const int* ldc, f64* work, int* info);

void SLC_FC_FUNC(dlarfg, DLARFG)(const int* n, f64* alpha, f64* x,
                                  const int* incx, f64* tau);

void SLC_FC_FUNC(dlarf, DLARF)(const char* side, const int* m, const int* n,
                                const f64* v, const int* incv, const f64* tau,
                                f64* c, const int* ldc, f64* work);

void SLC_FC_FUNC(dlarfx, DLARFX)(const char* side, const int* m, const int* n,
                                  const f64* v, const f64* tau, f64* c,
                                  const int* ldc, f64* work);

void SLC_FC_FUNC(dlarft, DLARFT)(const char* direct, const char* storev,
                                  const int* n, const int* k, f64* v,
                                  const int* ldv, const f64* tau, f64* t,
                                  const int* ldt);

void SLC_FC_FUNC(dlarfb, DLARFB)(const char* side, const char* trans,
                                  const char* direct, const char* storev,
                                  const int* m, const int* n, const int* k,
                                  const f64* v, const int* ldv, const f64* t,
                                  const int* ldt, f64* c, const int* ldc,
                                  f64* work, const int* ldwork);

/* LAPACK - SVD */
void SLC_FC_FUNC(dgesvd, DGESVD)(const char* jobu, const char* jobvt,
                                  const int* m, const int* n, f64* a, const int* lda,
                                  f64* s, f64* u, const int* ldu, f64* vt,
                                  const int* ldvt, f64* work, const int* lwork,
                                  int* info);

/* LAPACK - Singular values and condition estimation */
void SLC_FC_FUNC(dlaic1, DLAIC1)(const int* job, const int* j, const f64* x,
                                  const f64* sest, const f64* w, const f64* gamma,
                                  f64* sestpr, f64* s, f64* c);

f64 SLC_FC_FUNC(dlange, DLANGE)(const char* norm, const int* m, const int* n,
                                  const f64* a, const int* lda, f64* work);

void SLC_FC_FUNC(dlassq, DLASSQ)(const int* n, const f64* x, const int* incx,
                                  f64* scale, f64* sumsq);

f64 SLC_FC_FUNC(dlanhs, DLANHS)(const char* norm, const int* n,
                                 const f64* a, const int* lda, f64* work);

f64 SLC_FC_FUNC(dlapy2, DLAPY2)(const f64* x, const f64* y);

f64 SLC_FC_FUNC(dlapy3, DLAPY3)(const f64* x, const f64* y, const f64* z);
f64 SLC_FC_FUNC(dlantr, DLANTR)(const char* norm, const char* uplo, const char* diag,
                                const int* m, const int* n, const f64* a,
                                const int* lda, f64* work);

void SLC_FC_FUNC(dlascl, DLASCL)(const char* type, const int* kl, const int* ku,
                                  const f64* cfrom, const f64* cto, const int* m,
                                  const int* n, f64* a, const int* lda, int* info);

void SLC_FC_FUNC(dlabad, DLABAD)(f64* small, f64* large);

void SLC_FC_FUNC(dlag2, DLAG2)(const f64* a, const int* lda, const f64* b, const int* ldb,
                                const f64* safmin, f64* scale1, f64* scale2,
                                f64* wr1, f64* wr2, f64* wi);

void SLC_FC_FUNC(dlas2, DLAS2)(const f64* f, const f64* g, const f64* h,
                                f64* ssmin, f64* ssmax);

void SLC_FC_FUNC(dlasv2, DLASV2)(const f64* f, const f64* g, const f64* h,
                                  f64* ssmin, f64* ssmax, f64* snr, f64* csr,
                                  f64* snl, f64* csl);

void SLC_FC_FUNC(dladiv, DLADIV)(const f64* a, const f64* b, const f64* c, const f64* d,
                                  f64* p, f64* q);

void SLC_FC_FUNC(dlanv2, DLANV2)(f64* a, f64* b, f64* c, f64* d,
                                  f64* rt1r, f64* rt1i, f64* rt2r, f64* rt2i,
                                  f64* cs, f64* sn);

void SLC_FC_FUNC(dlagv2, DLAGV2)(f64* a, const int* lda, f64* b, const int* ldb,
                                  f64* alphar, f64* alphai, f64* beta,
                                  f64* csl, f64* snl, f64* csr, f64* snr);

void SLC_FC_FUNC(zlarfg, ZLARFG)(const int* n, double complex* alpha, double complex* x,
                                  const int* incx, double complex* tau);

void SLC_FC_FUNC(zlarf, ZLARF)(const char* side, const int* m, const int* n,
                                const double complex* v, const int* incv,
                                const double complex* tau, double complex* c,
                                const int* ldc, double complex* work);

void SLC_FC_FUNC(zlaic1, ZLAIC1)(const int* job, const int* j, const double complex* x,
                                  const f64* sest, const double complex* w,
                                  const double complex* gamma, f64* sestpr,
                                  double complex* s, double complex* c);

void SLC_FC_FUNC(zlacgv, ZLACGV)(const int* n, double complex* x, const int* incx);

void SLC_FC_FUNC(zlatzm, ZLATZM)(const char* side, const int* m, const int* n,
                                  const double complex* v, const int* incv,
                                  const double complex* tau, double complex* c1,
                                  double complex* c2, const int* ldc,
                                  double complex* work);

void SLC_FC_FUNC(zlapmt, ZLAPMT)(const int* forwrd, const int* m, const int* n,
                                  double complex* x, const int* ldx, int* k);

void SLC_FC_FUNC(zunmqr, ZUNMQR)(const char* side, const char* trans, const int* m,
                                  const int* n, const int* k, const double complex* a,
                                  const int* lda, const double complex* tau,
                                  double complex* c, const int* ldc,
                                  double complex* work, const int* lwork, int* info);

void SLC_FC_FUNC(zunmrq, ZUNMRQ)(const char* side, const char* trans, const int* m,
                                  const int* n, const int* k, const double complex* a,
                                  const int* lda, const double complex* tau,
                                  double complex* c, const int* ldc,
                                  double complex* work, const int* lwork, int* info);

void SLC_FC_FUNC(ztzrzf, ZTZRZF)(const int* m, const int* n, double complex* a,
                                  const int* lda, double complex* tau,
                                  double complex* work, const int* lwork, int* info);

void SLC_FC_FUNC(zunmrz, ZUNMRZ)(const char* side, const char* trans, const int* m,
                                  const int* n, const int* k, const int* l,
                                  const double complex* a, const int* lda,
                                  const double complex* tau, double complex* c,
                                  const int* ldc, double complex* work,
                                  const int* lwork, int* info);

void SLC_FC_FUNC(zstein, ZSTEIN)(const int* n, const f64* d, const f64* e, const int* m,
                                  const f64* w, const int* iblock, const int* isplit,
                                  double complex* z, const int* ldz, f64* work,
                                  int* iwork, int* ifail, int* info);

/* Complex BLAS */
void SLC_FC_FUNC(zcopy, ZCOPY)(const int* n, const double complex* x, const int* incx,
                                double complex* y, const int* incy);

void SLC_FC_FUNC(zswap, ZSWAP)(const int* n, double complex* x, const int* incx,
                                double complex* y, const int* incy);

void SLC_FC_FUNC(zaxpy, ZAXPY)(const int* n, const double complex* alpha,
                                const double complex* x, const int* incx,
                                double complex* y, const int* incy);

void SLC_FC_FUNC(zgemv, ZGEMV)(const char* trans, const int* m, const int* n,
                                const double complex* alpha, const double complex* a,
                                const int* lda, const double complex* x, const int* incx,
                                const double complex* beta, double complex* y, const int* incy);

void SLC_FC_FUNC(ztrsm, ZTRSM)(const char* side, const char* uplo, const char* transa,
                                const char* diag, const int* m, const int* n,
                                const double complex* alpha, const double complex* a,
                                const int* lda, double complex* b, const int* ldb);

void SLC_FC_FUNC(ztrmm, ZTRMM)(const char* side, const char* uplo, const char* transa,
                                const char* diag, const int* m, const int* n,
                                const double complex* alpha, const double complex* a,
                                const int* lda, double complex* b, const int* ldb);

void SLC_FC_FUNC(ztrmv, ZTRMV)(const char* uplo, const char* trans, const char* diag,
                                const int* n, const double complex* a, const int* lda,
                                double complex* x, const int* incx);

void SLC_FC_FUNC(zgeqrf, ZGEQRF)(const int* m, const int* n, double complex* a,
                                  const int* lda, double complex* tau,
                                  double complex* work, const int* lwork, int* info);

void SLC_FC_FUNC(zgerqf, ZGERQF)(const int* m, const int* n, double complex* a,
                                  const int* lda, double complex* tau,
                                  double complex* work, const int* lwork, int* info);

void SLC_FC_FUNC(zgemm, ZGEMM)(const char* transa, const char* transb,
                                const int* m, const int* n, const int* k,
                                const double complex* alpha, const double complex* a,
                                const int* lda, const double complex* b, const int* ldb,
                                const double complex* beta, double complex* c, const int* ldc);

int SLC_FC_FUNC(izamax, IZAMAX)(const int* n, const double complex* x, const int* incx);

f64 SLC_FC_FUNC(dzasum, DZASUM)(const int* n, const double complex* x, const int* incx);

void SLC_FC_FUNC(zdscal, ZDSCAL)(const int* n, const f64* da, double complex* zx, const int* incx);

void SLC_FC_FUNC(zscal, ZSCAL)(const int* n, const double complex* za, double complex* zx, const int* incx);

f64 SLC_FC_FUNC(dznrm2, DZNRM2)(const int* n, const double complex* x, const int* incx);

void SLC_FC_FUNC(zlassq, ZLASSQ)(const int* n, const double complex* x, const int* incx,
                                  f64* scale, f64* sumsq);

/* Complex LAPACK */
void SLC_FC_FUNC(zlaset, ZLASET)(const char* uplo, const int* m, const int* n,
                                  const double complex* alpha, const double complex* beta,
                                  double complex* a, const int* lda);

void SLC_FC_FUNC(zlacpy, ZLACPY)(const char* uplo, const int* m, const int* n,
                                  const double complex* a, const int* lda,
                                  double complex* b, const int* ldb);

void SLC_FC_FUNC(zlacon, ZLACON)(const int* n, double complex* v, double complex* x,
                                  f64* est, int* kase);

void SLC_FC_FUNC(zlatrs, ZLATRS)(const char* uplo, const char* trans, const char* diag,
                                  const char* normin, const int* n, const double complex* a,
                                  const int* lda, double complex* x, f64* scale, f64* cnorm,
                                  int* info);

void SLC_FC_FUNC(zdrscl, ZDRSCL)(const int* n, const f64* sa, double complex* sx,
                                  const int* incx);

void SLC_FC_FUNC(zlascl, ZLASCL)(const char* type, const int* kl, const int* ku,
                                  const f64* cfrom, const f64* cto, const int* m,
                                  const int* n, double complex* a, const int* lda, int* info);

f64 SLC_FC_FUNC(zlange, ZLANGE)(const char* norm, const int* m, const int* n,
                                 const double complex* a, const int* lda, f64* work);

void SLC_FC_FUNC(zgetrf, ZGETRF)(const int* m, const int* n, double complex* a,
                                  const int* lda, int* ipiv, int* info);

void SLC_FC_FUNC(zgetrs, ZGETRS)(const char* trans, const int* n, const int* nrhs,
                                  const double complex* a, const int* lda,
                                  const int* ipiv, double complex* b, const int* ldb,
                                  int* info);

void SLC_FC_FUNC(zgecon, ZGECON)(const char* norm, const int* n,
                                  const double complex* a, const int* lda,
                                  const f64* anorm, f64* rcond,
                                  double complex* work, f64* rwork, int* info);

void SLC_FC_FUNC(zgetri, ZGETRI)(const int* n, double complex* a, const int* lda,
                                  const int* ipiv, double complex* work, const int* lwork,
                                  int* info);

void SLC_FC_FUNC(zgees, ZGEES)(const char* jobvs, const char* sort,
                                int (*select)(const double complex*),
                                const int* n, double complex* a, const int* lda,
                                int* sdim, double complex* w, double complex* vs,
                                const int* ldvs, double complex* work, const int* lwork,
                                f64* rwork, int* bwork, int* info);

void SLC_FC_FUNC(zgges, ZGGES)(const char* jobvsl, const char* jobvsr, const char* sort,
                                int (*selctg)(const double complex*, const double complex*),
                                const int* n, double complex* a, const int* lda,
                                double complex* b, const int* ldb, int* sdim,
                                double complex* alpha, double complex* beta,
                                double complex* vsl, const int* ldvsl,
                                double complex* vsr, const int* ldvsr,
                                double complex* work, const int* lwork,
                                f64* rwork, int* bwork, int* info);

void SLC_FC_FUNC(zgesvd, ZGESVD)(const char* jobu, const char* jobvt,
                                  const int* m, const int* n, double complex* a,
                                  const int* lda, f64* s, double complex* u, const int* ldu,
                                  double complex* vt, const int* ldvt, double complex* work,
                                  const int* lwork, f64* rwork, int* info);

void SLC_FC_FUNC(zgeqp3, ZGEQP3)(const int* m, const int* n, double complex* a,
                                  const int* lda, int* jpvt, double complex* tau,
                                  double complex* work, const int* lwork, f64* rwork,
                                  int* info);

void SLC_FC_FUNC(zungqr, ZUNGQR)(const int* m, const int* n, const int* k,
                                  double complex* a, const int* lda,
                                  const double complex* tau, double complex* work,
                                  const int* lwork, int* info);

void SLC_FC_FUNC(zgesv, ZGESV)(const int* n, const int* nrhs, double complex* a,
                                const int* lda, int* ipiv, double complex* b,
                                const int* ldb, int* info);

void SLC_FC_FUNC(zgetc2, ZGETC2)(const int* n, double complex* a, const int* lda,
                                  int* ipiv, int* jpiv, int* info);

void SLC_FC_FUNC(zgesc2, ZGESC2)(const int* n, const double complex* a, const int* lda,
                                  double complex* rhs, const int* ipiv, const int* jpiv,
                                  f64* scale);

void SLC_FC_FUNC(zlacp2, ZLACP2)(const char* uplo, const int* m, const int* n,
                                  const f64* a, const int* lda, double complex* b,
                                  const int* ldb);

f64 SLC_FC_FUNC(zlanhs, ZLANHS)(const char* norm, const int* n,
                                 const double complex* a, const int* lda, f64* work);

f64 SLC_FC_FUNC(zlantr, ZLANTR)(const char* norm, const char* uplo, const char* diag,
                                 const int* m, const int* n, const double complex* a,
                                 const int* lda, f64* work);

void SLC_FC_FUNC(zlarnv, ZLARNV)(const int* idist, int* iseed, const int* n,
                                  double complex* x);

double complex SLC_FC_FUNC(zdotu, ZDOTU)(const int* n, const double complex* x,
                                          const int* incx, const double complex* y,
                                          const int* incy);

double complex SLC_FC_FUNC(zladiv, ZLADIV)(const double complex* x,
                                            const double complex* y);

void SLC_FC_FUNC(xerbla, XERBLA)(const char* srname, const int* info);

void SLC_FC_FUNC(dgetc2, DGETC2)(const int* n, f64* a, const int* lda,
                                  int* ipiv, int* jpiv, int* info);

void SLC_FC_FUNC(dgesc2, DGESC2)(const int* n, const f64* a, const int* lda,
                                  f64* rhs, const int* ipiv, const int* jpiv,
                                  f64* scale);

/* LAPACK - Sylvester equations */
void SLC_FC_FUNC(dlaln2, DLALN2)(const int* ltrans, const int* na, const int* nw,
                                  const f64* smin, const f64* ca,
                                  const f64* a, const int* lda,
                                  const f64* d1, const f64* d2,
                                  const f64* b, const int* ldb,
                                  const f64* wr, const f64* wi,
                                  f64* x, const int* ldx,
                                  f64* scale, f64* xnorm, int* info);

void SLC_FC_FUNC(dlasy2, DLASY2)(const int* ltranl, const int* ltranr,
                                  const int* isgn, const int* n1, const int* n2,
                                  const f64* tl, const int* ldtl,
                                  const f64* tr, const int* ldtr,
                                  const f64* b, const int* ldb,
                                  f64* scale, f64* x, const int* ldx,
                                  f64* xnorm, int* info);

void SLC_FC_FUNC(dtrsyl, DTRSYL)(const char* trana, const char* tranb,
                                  const int* isgn, const int* m, const int* n,
                                  const f64* a, const int* lda,
                                  const f64* b, const int* ldb,
                                  f64* c, const int* ldc, f64* scale, int* info);

void SLC_FC_FUNC(dtgsyl, DTGSYL)(const char* trans, const int* ijob,
                                  const int* m, const int* n,
                                  const f64* a, const int* lda,
                                  const f64* b, const int* ldb,
                                  f64* c, const int* ldc,
                                  const f64* d, const int* ldd,
                                  const f64* e, const int* lde,
                                  f64* f, const int* ldf,
                                  f64* scale, f64* dif,
                                  f64* work, const int* lwork,
                                  int* iwork, int* info);

/* LAPACK - Orthogonal transformations */
void SLC_FC_FUNC(dormqr, DORMQR)(const char* side, const char* trans,
                                  const int* m, const int* n, const int* k,
                                  const f64* a, const int* lda, const f64* tau,
                                  f64* c, const int* ldc, f64* work,
                                  const int* lwork, int* info);

void SLC_FC_FUNC(dorgqr, DORGQR)(const int* m, const int* n, const int* k,
                                  f64* a, const int* lda, const f64* tau,
                                  f64* work, const int* lwork, int* info);

void SLC_FC_FUNC(dormrq, DORMRQ)(const char* side, const char* trans,
                                  const int* m, const int* n, const int* k,
                                  const f64* a, const int* lda, const f64* tau,
                                  f64* c, const int* ldc, f64* work,
                                  const int* lwork, int* info);

void SLC_FC_FUNC(dormbr, DORMBR)(const char* vect, const char* side,
                                  const char* trans, const int* m, const int* n,
                                  const int* k, const f64* a, const int* lda,
                                  const f64* tau, f64* c, const int* ldc,
                                  f64* work, const int* lwork, int* info);

void SLC_FC_FUNC(dorgrq, DORGRQ)(const int* m, const int* n, const int* k,
                                  f64* a, const int* lda, const f64* tau,
                                  f64* work, const int* lwork, int* info);

void SLC_FC_FUNC(dlatzm, DLATZM)(const char* side, const int* m, const int* n,
                                  const f64* v, const int* incv, const f64* tau,
                                  f64* c1, f64* c2, const int* ldc, f64* work);

void SLC_FC_FUNC(dormrz, DORMRZ)(const char* side, const char* trans,
                                  const int* m, const int* n, const int* k,
                                  const int* l, const f64* a, const int* lda,
                                  const f64* tau, f64* c, const int* ldc,
                                  f64* work, const int* lwork, int* info);

void SLC_FC_FUNC(dtzrzf, DTZRZF)(const int* m, const int* n, f64* a,
                                  const int* lda, f64* tau, f64* work,
                                  const int* lwork, int* info);

void SLC_FC_FUNC(dlartg, DLARTG)(const f64* f, const f64* g, f64* cs, f64* sn,
                                  f64* r);

void SLC_FC_FUNC(drotg, DROTG)(f64* da, f64* db, f64* c, f64* s);

void SLC_FC_FUNC(drot, DROT)(const int* n, f64* dx, const int* incx,
                              f64* dy, const int* incy, const f64* c,
                              const f64* s);

void SLC_FC_FUNC(dlasr, DLASR)(const char* side, const char* pivot,
                                const char* direct, const int* m, const int* n,
                                const f64* c, const f64* s, f64* a,
                                const int* lda);

void SLC_FC_FUNC(zlartg, ZLARTG)(const double complex* f, const double complex* g,
                                  f64* cs, double complex* sn, double complex* r);

void SLC_FC_FUNC(zrot, ZROT)(const int* n, double complex* cx, const int* incx,
                              double complex* cy, const int* incy, const f64* c,
                              const double complex* s);

void SLC_FC_FUNC(dsyevx, DSYEVX)(const char* jobz, const char* range, const char* uplo,
                                  const int* n, f64* a, const int* lda,
                                  const f64* vl, const f64* vu, const int* il,
                                  const int* iu, const f64* abstol, int* m,
                                  f64* w, f64* z, const int* ldz, f64* work,
                                  const int* lwork, int* iwork, int* ifail,
                                  int* info);

void SLC_FC_FUNC(dsyev, DSYEV)(const char* jobz, const char* uplo, const int* n,
                                f64* a, const int* lda, f64* w, f64* work,
                                const int* lwork, int* info);

void SLC_FC_FUNC(dgges, DGGES)(const char* jobvsl, const char* jobvsr,
                                const char* sort, int (*selctg)(),
                                const int* n, f64* a, const int* lda,
                                f64* b, const int* ldb, int* sdim,
                                f64* alphar, f64* alphai, f64* beta,
                                f64* vsl, const int* ldvsl, f64* vsr,
                                const int* ldvsr, f64* work, const int* lwork,
                                int* bwork, int* info);

void SLC_FC_FUNC(dgegs, DGEGS)(const char* jobvsl, const char* jobvsr,
                                const int* n, f64* a, const int* lda,
                                f64* b, const int* ldb,
                                f64* alphar, f64* alphai, f64* beta,
                                f64* vsl, const int* ldvsl, f64* vsr,
                                const int* ldvsr, f64* work, const int* lwork,
                                int* info);

void SLC_FC_FUNC(dgees, DGEES)(const char* jobvs, const char* sort,
                                int (*select)(const f64*, const f64*),
                                const int* n, f64* a, const int* lda,
                                int* sdim, f64* wr, f64* wi,
                                f64* vs, const int* ldvs, f64* work,
                                const int* lwork, int* bwork, int* info);

void SLC_FC_FUNC(dtrsen, DTRSEN)(const char* job, const char* compq,
                                 const int* select, const int* n, f64* t,
                                 const int* ldt, f64* q, const int* ldq,
                                 f64* wr, f64* wi, int* m, f64* s, f64* sep,
                                 f64* work, const int* lwork, int* iwork,
                                 const int* liwork, int* info);

void SLC_FC_FUNC(dgeev, DGEEV)(const char* jobvl, const char* jobvr,
                                const int* n, f64* a, const int* lda,
                                f64* wr, f64* wi,
                                f64* vl, const int* ldvl,
                                f64* vr, const int* ldvr,
                                f64* work, const int* lwork, int* info);

/* LAPACK - Hessenberg/Schur routines */
void SLC_FC_FUNC(dgebal, DGEBAL)(const char* job, const int* n, f64* a,
                                  const int* lda, int* ilo, int* ihi,
                                  f64* scale, int* info);

void SLC_FC_FUNC(dggbal, DGGBAL)(const char* job, const int* n, f64* a,
                                  const int* lda, f64* b, const int* ldb,
                                  int* ilo, int* ihi, f64* lscale, f64* rscale,
                                  f64* work, int* info);

void SLC_FC_FUNC(dhgeqz, DHGEQZ)(const char* job, const char* compq, const char* compz,
                                  const int* n, const int* ilo, const int* ihi,
                                  f64* h, const int* ldh, f64* t, const int* ldt,
                                  f64* alphar, f64* alphai, f64* beta,
                                  f64* q, const int* ldq, f64* z, const int* ldz,
                                  f64* work, const int* lwork, int* info);

void SLC_FC_FUNC(zhgeqz, ZHGEQZ)(const char* job, const char* compq, const char* compz,
                                  const int* n, const int* ilo, const int* ihi,
                                  double complex* h, const int* ldh,
                                  double complex* t, const int* ldt,
                                  double complex* alpha, double complex* beta,
                                  double complex* q, const int* ldq,
                                  double complex* z, const int* ldz,
                                  double complex* work, const int* lwork,
                                  f64* rwork, int* info);

void SLC_FC_FUNC(dggev, DGGEV)(const char* jobvl, const char* jobvr,
                                const int* n, f64* a, const int* lda,
                                f64* b, const int* ldb,
                                f64* alphar, f64* alphai, f64* beta,
                                f64* vl, const int* ldvl,
                                f64* vr, const int* ldvr,
                                f64* work, const int* lwork, int* info);

void SLC_FC_FUNC(dgghrd, DGGHRD)(const char* compq, const char* compz,
                                  const int* n, const int* ilo, const int* ihi,
                                  f64* a, const int* lda, f64* b, const int* ldb,
                                  f64* q, const int* ldq, f64* z, const int* ldz,
                                  int* info);

void SLC_FC_FUNC(dtgevc, DTGEVC)(const char* side, const char* howmny,
                                  const int* select, const int* n,
                                  const f64* s, const int* lds,
                                  const f64* p, const int* ldp,
                                  f64* vl, const int* ldvl,
                                  f64* vr, const int* ldvr,
                                  const int* mm, int* m, f64* work, int* info);

void SLC_FC_FUNC(dggbak, DGGBAK)(const char* job, const char* side,
                                  const int* n, const int* ilo, const int* ihi,
                                  const f64* lscale, const f64* rscale,
                                  const int* m, f64* v, const int* ldv, int* info);

void SLC_FC_FUNC(dgelqf, DGELQF)(const int* m, const int* n, f64* a,
                                  const int* lda, f64* tau, f64* work,
                                  const int* lwork, int* info);

void SLC_FC_FUNC(dormlq, DORMLQ)(const char* side, const char* trans,
                                  const int* m, const int* n, const int* k,
                                  const f64* a, const int* lda, const f64* tau,
                                  f64* c, const int* ldc, f64* work,
                                  const int* lwork, int* info);

void SLC_FC_FUNC(dgehrd, DGEHRD)(const int* n, const int* ilo, const int* ihi,
                                  f64* a, const int* lda, f64* tau,
                                  f64* work, const int* lwork, int* info);

void SLC_FC_FUNC(dorghr, DORGHR)(const int* n, const int* ilo, const int* ihi,
                                  f64* a, const int* lda, const f64* tau,
                                  f64* work, const int* lwork, int* info);

void SLC_FC_FUNC(dormhr, DORMHR)(const char* side, const char* trans, const int* m,
                                  const int* n, const int* ilo, const int* ihi,
                                  const f64* a, const int* lda, const f64* tau,
                                  f64* c, const int* ldc, f64* work, const int* lwork,
                                  int* info);

void SLC_FC_FUNC(dhseqr, DHSEQR)(const char* job, const char* compz, const int* n,
                                  const int* ilo, const int* ihi, f64* h,
                                  const int* ldh, f64* wr, f64* wi, f64* z,
                                  const int* ldz, f64* work, const int* lwork,
                                  int* info);

void SLC_FC_FUNC(dlahqr, DLAHQR)(const int* wantt, const int* wantz, const int* n,
                                  const int* ilo, const int* ihi, f64* h,
                                  const int* ldh, f64* wr, f64* wi,
                                  const int* iloz, const int* ihiz, f64* z,
                                  const int* ldz, int* info);

void SLC_FC_FUNC(zlahqr, ZLAHQR)(const int* wantt, const int* wantz, const int* n,
                                  const int* ilo, const int* ihi, c128* h,
                                  const int* ldh, c128* w,
                                  const int* iloz, const int* ihiz, c128* z,
                                  const int* ldz, int* info);

void SLC_FC_FUNC(dtrevc, DTREVC)(const char* side, const char* howmny,
                                  const int* select, const int* n, const f64* t,
                                  const int* ldt, f64* vl, const int* ldvl,
                                  f64* vr, const int* ldvr, const int* mm,
                                  int* m, f64* work, int* info);

void SLC_FC_FUNC(dtrexc, DTREXC)(const char* compq, const int* n, f64* t,
                                  const int* ldt, f64* q, const int* ldq,
                                  int* ifst, int* ilst, f64* work, int* info);

void SLC_FC_FUNC(ztrexc, ZTREXC)(const char* compq, const int* n, c128* t,
                                  const int* ldt, c128* q, const int* ldq,
                                  int* ifst, int* ilst, int* info);

/* LAPACK - Bidiagonal decomposition and SVD */
void SLC_FC_FUNC(dgebrd, DGEBRD)(const int* m, const int* n, f64* a,
                                  const int* lda, f64* d, f64* e, f64* tauq,
                                  f64* taup, f64* work, const int* lwork,
                                  int* info);

void SLC_FC_FUNC(dorgbr, DORGBR)(const char* vect, const int* m, const int* n,
                                  const int* k, f64* a, const int* lda,
                                  const f64* tau, f64* work, const int* lwork,
                                  int* info);

void SLC_FC_FUNC(dbdsqr, DBDSQR)(const char* uplo, const int* n, const int* ncvt,
                                  const int* nru, const int* ncc, f64* d, f64* e,
                                  f64* vt, const int* ldvt, f64* u, const int* ldu,
                                  f64* c, const int* ldc, f64* work, int* info);

void SLC_FC_FUNC(dgels, DGELS)(const char* trans, const int* m, const int* n,
                                const int* nrhs, f64* a, const int* lda, f64* b,
                                const int* ldb, f64* work, const int* lwork, int* info);

void SLC_FC_FUNC(dgelss, DGELSS)(const int* m, const int* n, const int* nrhs,
                                  f64* a, const int* lda, f64* b, const int* ldb,
                                  f64* s, const f64* rcond, int* rank, f64* work,
                                  const int* lwork, int* info);

void SLC_FC_FUNC(dgelsy, DGELSY)(const int* m, const int* n, const int* nrhs,
                                  f64* a, const int* lda, f64* b, const int* ldb,
                                  int* jpvt, const f64* rcond, int* rank, f64* work,
                                  const int* lwork, int* info);

void SLC_FC_FUNC(dtrtrs, DTRTRS)(const char* uplo, const char* trans, const char* diag,
                                  const int* n, const int* nrhs, const f64* a,
                                  const int* lda, f64* b, const int* ldb, int* info);

void SLC_FC_FUNC(dormrz, DORMRZ)(const char* side, const char* trans,
                                  const int* m, const int* n, const int* k, const int* l,
                                  const f64* a, const int* lda, const f64* tau,
                                  f64* c, const int* ldc, f64* work, const int* lwork,
                                  int* info);

int SLC_FC_FUNC(ilaenv, ILAENV)(const int* ispec, const char* name, const char* opts,
                                const int* n1, const int* n2, const int* n3, const int* n4);

void SLC_FC_FUNC(dgeqlf, DGEQLF)(const int* m, const int* n, f64* a, const int* lda,
                                  f64* tau, f64* work, const int* lwork, int* info);

void SLC_FC_FUNC(dormql, DORMQL)(const char* side, const char* trans, const int* m,
                                  const int* n, const int* k, const f64* a, const int* lda,
                                  const f64* tau, f64* c, const int* ldc, f64* work,
                                  const int* lwork, int* info);

void SLC_FC_FUNC(dtrcon, DTRCON)(const char* norm, const char* uplo, const char* diag,
                                  const int* n, const f64* a, const int* lda, f64* rcond,
                                  f64* work, int* iwork, int* info);

void SLC_FC_FUNC(dgebak, DGEBAK)(const char* job, const char* side, const int* n,
                                  const int* ilo, const int* ihi, const f64* scale,
                                  const int* m, f64* v, const int* ldv, int* info);

void SLC_FC_FUNC(dlaexc, DLAEXC)(const int* wantq, const int* n, f64* t,
                                  const int* ldt, f64* q, const int* ldq,
                                  const int* j1, const int* n1, const int* n2,
                                  f64* work, int* info);

void SLC_FC_FUNC(dlacn2, DLACN2)(const int* n, f64* v, f64* x, int* isgn,
                                  f64* est, int* kase, int* isave);

void SLC_FC_FUNC(dlatrs, DLATRS)(const char* uplo, const char* trans,
                                  const char* diag, const char* normin,
                                  const int* n, const f64* a, const int* lda,
                                  f64* x, f64* scale, f64* cnorm, int* info);

void SLC_FC_FUNC(dtgsen, DTGSEN)(const int* ijob, const int* wantq, const int* wantz,
                                  const int* select, const int* n,
                                  f64* a, const int* lda, f64* b, const int* ldb,
                                  f64* alphar, f64* alphai, f64* beta,
                                  f64* q, const int* ldq, f64* z, const int* ldz,
                                  int* m, f64* pl, f64* pr, f64* dif,
                                  f64* work, const int* lwork,
                                  int* iwork, const int* liwork, int* info);

void SLC_FC_FUNC(dgeqr2, DGEQR2)(const int* m, const int* n, f64* a,
                                  const int* lda, f64* tau, f64* work, int* info);

void SLC_FC_FUNC(dorg2r, DORG2R)(const int* m, const int* n, const int* k,
                                  f64* a, const int* lda, const f64* tau,
                                  f64* work, int* info);

void SLC_FC_FUNC(dorgr2, DORGR2)(const int* m, const int* n, const int* k,
                                  f64* a, const int* lda, const f64* tau,
                                  f64* work, int* info);

void SLC_FC_FUNC(dorm2r, DORM2R)(const char* side, const char* trans,
                                  const int* m, const int* n, const int* k,
                                  const f64* a, const int* lda, const f64* tau,
                                  f64* c, const int* ldc, f64* work, int* info);

void SLC_FC_FUNC(dtgex2, DTGEX2)(const int* wantq, const int* wantz,
                                  const int* n, f64* a, const int* lda,
                                  f64* b, const int* ldb, f64* q, const int* ldq,
                                  f64* z, const int* ldz, const int* j1,
                                  const int* n1, const int* n2,
                                  f64* work, const int* lwork, int* info);

void SLC_FC_FUNC(dtgexc, DTGEXC)(const int* wantq, const int* wantz,
                                  const int* n, f64* a, const int* lda,
                                  f64* b, const int* ldb, f64* q, const int* ldq,
                                  f64* z, const int* ldz, int* ifst, int* ilst,
                                  f64* work, const int* lwork, int* info);

void SLC_FC_FUNC(ztgexc, ZTGEXC)(const int* wantq, const int* wantz,
                                  const int* n, c128* a, const int* lda,
                                  c128* b, const int* ldb, c128* q, const int* ldq,
                                  c128* z, const int* ldz, int* ifst, int* ilst,
                                  int* info);

void SLC_FC_FUNC(zggbal, ZGGBAL)(const char* job, const int* n,
                                  c128* a, const int* lda, c128* b, const int* ldb,
                                  int* ilo, int* ihi, f64* lscale, f64* rscale,
                                  f64* work, int* info);

void SLC_FC_FUNC(zgghrd, ZGGHRD)(const char* compq, const char* compz,
                                  const int* n, const int* ilo, const int* ihi,
                                  c128* a, const int* lda, c128* b, const int* ldb,
                                  c128* q, const int* ldq, c128* z, const int* ldz,
                                  int* info);

void SLC_FC_FUNC(zggbak, ZGGBAK)(const char* job, const char* side,
                                  const int* n, const int* ilo, const int* ihi,
                                  const f64* lscale, const f64* rscale,
                                  const int* m, c128* v, const int* ldv, int* info);

void SLC_FC_FUNC(ztgevc, ZTGEVC)(const char* side, const char* howmny,
                                  const int* select, const int* n,
                                  const c128* s, const int* lds,
                                  const c128* p, const int* ldp,
                                  c128* vl, const int* ldvl,
                                  c128* vr, const int* ldvr,
                                  const int* mm, int* m, c128* work,
                                  f64* rwork, int* info);

void SLC_FC_FUNC(dlasrt, DLASRT)(const char* id, const int* n, f64* d, int* info);

#undef int

/* Convenience macros for calling BLAS/LAPACK */
#define SLC_DCOPY    SLC_FC_FUNC(dcopy, DCOPY)
#define SLC_DAXPY    SLC_FC_FUNC(daxpy, DAXPY)
#define SLC_DSCAL    SLC_FC_FUNC(dscal, DSCAL)
#define SLC_DRSCL    SLC_FC_FUNC(drscl, DRSCL)
#define SLC_DSWAP    SLC_FC_FUNC(dswap, DSWAP)
#define SLC_DDOT     SLC_FC_FUNC(ddot, DDOT)
#define SLC_DNRM2    SLC_FC_FUNC(dnrm2, DNRM2)
#define SLC_IDAMAX   SLC_FC_FUNC(idamax, IDAMAX)
#define SLC_DASUM    SLC_FC_FUNC(dasum, DASUM)
#define SLC_DGEMV    SLC_FC_FUNC(dgemv, DGEMV)
#define SLC_DTRMV    SLC_FC_FUNC(dtrmv, DTRMV)
#define SLC_DTPMV    SLC_FC_FUNC(dtpmv, DTPMV)
#define SLC_DTRSV    SLC_FC_FUNC(dtrsv, DTRSV)
#define SLC_DGER     SLC_FC_FUNC(dger, DGER)
#define SLC_DSYMV    SLC_FC_FUNC(dsymv, DSYMV)
#define SLC_DSYR2    SLC_FC_FUNC(dsyr2, DSYR2)
#define SLC_DTRMM    SLC_FC_FUNC(dtrmm, DTRMM)
#define SLC_DTRSM    SLC_FC_FUNC(dtrsm, DTRSM)
#define SLC_DGEMM    SLC_FC_FUNC(dgemm, DGEMM)
#define SLC_DSYRK    SLC_FC_FUNC(dsyrk, DSYRK)
#define SLC_DSYR2K   SLC_FC_FUNC(dsyr2k, DSYR2K)
#define SLC_DSYMM    SLC_FC_FUNC(dsymm, DSYMM)
#define SLC_DLAMCH   SLC_FC_FUNC(dlamch, DLAMCH)
#define SLC_DLAMC3   SLC_FC_FUNC(dlamc3, DLAMC3)
#define SLC_DLACPY   SLC_FC_FUNC(dlacpy, DLACPY)
#define SLC_DLAPMT   SLC_FC_FUNC(dlapmt, DLAPMT)
#define SLC_DLARNV   SLC_FC_FUNC(dlarnv, DLARNV)
#define SLC_DLASET   SLC_FC_FUNC(dlaset, DLASET)
#define SLC_DGETRF   SLC_FC_FUNC(dgetrf, DGETRF)
#define SLC_DGETRS   SLC_FC_FUNC(dgetrs, DGETRS)
#define SLC_DGESVD   SLC_FC_FUNC(dgesvd, DGESVD)
#define SLC_DGETRI   SLC_FC_FUNC(dgetri, DGETRI)
#define SLC_DGESV    SLC_FC_FUNC(dgesv, DGESV)
#define SLC_DGESVX   SLC_FC_FUNC(dgesvx, DGESVX)
#define SLC_DGECON   SLC_FC_FUNC(dgecon, DGECON)
#define SLC_DGEEQU   SLC_FC_FUNC(dgeequ, DGEEQU)
#define SLC_DGERFS   SLC_FC_FUNC(dgerfs, DGERFS)
#define SLC_DLAQGE   SLC_FC_FUNC(dlaqge, DLAQGE)
#define SLC_DPOTRF   SLC_FC_FUNC(dpotrf, DPOTRF)
#define SLC_DPOSV    SLC_FC_FUNC(dposv, DPOSV)
#define SLC_DPPTRF   SLC_FC_FUNC(dpptrf, DPPTRF)
#define SLC_DSYTRF   SLC_FC_FUNC(dsytrf, DSYTRF)
#define SLC_DSYTRI   SLC_FC_FUNC(dsytri, DSYTRI)
#define SLC_DSYSV    SLC_FC_FUNC(dsysv, DSYSV)
#define SLC_DPOCON   SLC_FC_FUNC(dpocon, DPOCON)
#define SLC_DSYCON   SLC_FC_FUNC(dsycon, DSYCON)
#define SLC_DSYTRS   SLC_FC_FUNC(dsytrs, DSYTRS)
#define SLC_DLANSY   SLC_FC_FUNC(dlansy, DLANSY)
#define SLC_DGEQRF   SLC_FC_FUNC(dgeqrf, DGEQRF)
#define SLC_DGELQF   SLC_FC_FUNC(dgelqf, DGELQF)
#define SLC_DGEQP3   SLC_FC_FUNC(dgeqp3, DGEQP3)
#define SLC_DGERQF   SLC_FC_FUNC(dgerqf, DGERQF)
#define SLC_DGERQ2   SLC_FC_FUNC(dgerq2, DGERQ2)
#define SLC_DORMR2   SLC_FC_FUNC(dormr2, DORMR2)
#define SLC_DLARFG   SLC_FC_FUNC(dlarfg, DLARFG)
#define SLC_DLARF    SLC_FC_FUNC(dlarf, DLARF)
#define SLC_DLARFX   SLC_FC_FUNC(dlarfx, DLARFX)
#define SLC_DLARFT   SLC_FC_FUNC(dlarft, DLARFT)
#define SLC_DLARFB   SLC_FC_FUNC(dlarfb, DLARFB)
#define SLC_DLAIC1   SLC_FC_FUNC(dlaic1, DLAIC1)
#define SLC_DLANGE   SLC_FC_FUNC(dlange, DLANGE)
#define SLC_DLASSQ   SLC_FC_FUNC(dlassq, DLASSQ)
#define SLC_DLANHS   SLC_FC_FUNC(dlanhs, DLANHS)
#define SLC_DORMQR   SLC_FC_FUNC(dormqr, DORMQR)
#define SLC_DORMLQ   SLC_FC_FUNC(dormlq, DORMLQ)
#define SLC_DORGQR   SLC_FC_FUNC(dorgqr, DORGQR)
#define SLC_DORMRQ   SLC_FC_FUNC(dormrq, DORMRQ)
#define SLC_DORMBR   SLC_FC_FUNC(dormbr, DORMBR)
#define SLC_DORGRQ   SLC_FC_FUNC(dorgrq, DORGRQ)
#define SLC_DLATZM   SLC_FC_FUNC(dlatzm, DLATZM)
#define SLC_DORMRZ   SLC_FC_FUNC(dormrz, DORMRZ)
#define SLC_DTZRZF   SLC_FC_FUNC(dtzrzf, DTZRZF)
#define SLC_DLAPY2   SLC_FC_FUNC(dlapy2, DLAPY2)
#define SLC_DLAPY3   SLC_FC_FUNC(dlapy3, DLAPY3)
#define SLC_DLASCL   SLC_FC_FUNC(dlascl, DLASCL)
#define SLC_DLABAD   SLC_FC_FUNC(dlabad, DLABAD)
#define SLC_DLAG2    SLC_FC_FUNC(dlag2, DLAG2)
#define SLC_DLAS2    SLC_FC_FUNC(dlas2, DLAS2)
#define SLC_DLASV2   SLC_FC_FUNC(dlasv2, DLASV2)
#define SLC_DLADIV   SLC_FC_FUNC(dladiv, DLADIV)
#define SLC_DLANV2   SLC_FC_FUNC(dlanv2, DLANV2)
#define SLC_DLAGV2   SLC_FC_FUNC(dlagv2, DLAGV2)
#define SLC_ZLARFG   SLC_FC_FUNC(zlarfg, ZLARFG)
#define SLC_ZLARF    SLC_FC_FUNC(zlarf, ZLARF)
#define SLC_ZLAIC1   SLC_FC_FUNC(zlaic1, ZLAIC1)
#define SLC_ZLACGV   SLC_FC_FUNC(zlacgv, ZLACGV)
#define SLC_ZLATZM   SLC_FC_FUNC(zlatzm, ZLATZM)
#define SLC_ZLAPMT   SLC_FC_FUNC(zlapmt, ZLAPMT)
#define SLC_ZUNMQR   SLC_FC_FUNC(zunmqr, ZUNMQR)
#define SLC_ZUNMRQ   SLC_FC_FUNC(zunmrq, ZUNMRQ)
#define SLC_ZTZRZF   SLC_FC_FUNC(ztzrzf, ZTZRZF)
#define SLC_ZUNMRZ   SLC_FC_FUNC(zunmrz, ZUNMRZ)
#define SLC_ZSTEIN   SLC_FC_FUNC(zstein, ZSTEIN)
#define SLC_ZCOPY    SLC_FC_FUNC(zcopy, ZCOPY)
#define SLC_ZSWAP    SLC_FC_FUNC(zswap, ZSWAP)
#define SLC_ZAXPY    SLC_FC_FUNC(zaxpy, ZAXPY)
#define SLC_ZGEMV    SLC_FC_FUNC(zgemv, ZGEMV)
#define SLC_ZTRSM    SLC_FC_FUNC(ztrsm, ZTRSM)
#define SLC_ZTRMM    SLC_FC_FUNC(ztrmm, ZTRMM)
#define SLC_ZTRMV    SLC_FC_FUNC(ztrmv, ZTRMV)
#define SLC_ZGEQRF   SLC_FC_FUNC(zgeqrf, ZGEQRF)
#define SLC_ZGERQF   SLC_FC_FUNC(zgerqf, ZGERQF)
#define SLC_ZGEMM    SLC_FC_FUNC(zgemm, ZGEMM)
#define SLC_IZAMAX   SLC_FC_FUNC(izamax, IZAMAX)
#define SLC_DZASUM   SLC_FC_FUNC(dzasum, DZASUM)
#define SLC_ZDSCAL   SLC_FC_FUNC(zdscal, ZDSCAL)
#define SLC_ZSCAL    SLC_FC_FUNC(zscal, ZSCAL)
#define SLC_DZNRM2   SLC_FC_FUNC(dznrm2, DZNRM2)
#define SLC_ZLASSQ   SLC_FC_FUNC(zlassq, ZLASSQ)
#define SLC_ZLASET   SLC_FC_FUNC(zlaset, ZLASET)
#define SLC_ZLACPY   SLC_FC_FUNC(zlacpy, ZLACPY)
#define SLC_ZLACON   SLC_FC_FUNC(zlacon, ZLACON)
#define SLC_ZLATRS   SLC_FC_FUNC(zlatrs, ZLATRS)
#define SLC_ZDRSCL   SLC_FC_FUNC(zdrscl, ZDRSCL)
#define SLC_ZLASCL   SLC_FC_FUNC(zlascl, ZLASCL)
#define SLC_ZLANGE   SLC_FC_FUNC(zlange, ZLANGE)
#define SLC_ZGETRF   SLC_FC_FUNC(zgetrf, ZGETRF)
#define SLC_ZGETRS   SLC_FC_FUNC(zgetrs, ZGETRS)
#define SLC_ZGECON   SLC_FC_FUNC(zgecon, ZGECON)
#define SLC_ZGETRI   SLC_FC_FUNC(zgetri, ZGETRI)
#define SLC_ZGEES    SLC_FC_FUNC(zgees, ZGEES)
#define SLC_ZGGES    SLC_FC_FUNC(zgges, ZGGES)
#define SLC_ZGESVD   SLC_FC_FUNC(zgesvd, ZGESVD)
#define SLC_ZGEQP3   SLC_FC_FUNC(zgeqp3, ZGEQP3)
#define SLC_ZUNGQR   SLC_FC_FUNC(zungqr, ZUNGQR)
#define SLC_ZGESV    SLC_FC_FUNC(zgesv, ZGESV)
#define SLC_ZGETC2   SLC_FC_FUNC(zgetc2, ZGETC2)
#define SLC_ZGESC2   SLC_FC_FUNC(zgesc2, ZGESC2)
#define SLC_ZLACP2   SLC_FC_FUNC(zlacp2, ZLACP2)
#define SLC_XERBLA   SLC_FC_FUNC(xerbla, XERBLA)
#define SLC_DGETC2   SLC_FC_FUNC(dgetc2, DGETC2)
#define SLC_DGESC2   SLC_FC_FUNC(dgesc2, DGESC2)
#define SLC_DLALN2   SLC_FC_FUNC(dlaln2, DLALN2)
#define SLC_DLASY2   SLC_FC_FUNC(dlasy2, DLASY2)
#define SLC_DTRSYL   SLC_FC_FUNC(dtrsyl, DTRSYL)
#define SLC_DTGSYL   SLC_FC_FUNC(dtgsyl, DTGSYL)
#define SLC_DLARTG   SLC_FC_FUNC(dlartg, DLARTG)
#define SLC_DROTG    SLC_FC_FUNC(drotg, DROTG)
#define SLC_DROT     SLC_FC_FUNC(drot, DROT)
#define SLC_DLASR    SLC_FC_FUNC(dlasr, DLASR)
#define SLC_ZLARTG   SLC_FC_FUNC(zlartg, ZLARTG)
#define SLC_ZROT     SLC_FC_FUNC(zrot, ZROT)
#define SLC_DSYEVX   SLC_FC_FUNC(dsyevx, DSYEVX)
#define SLC_DSYEV    SLC_FC_FUNC(dsyev, DSYEV)
#define SLC_DGGES    SLC_FC_FUNC(dgges, DGGES)
#define SLC_DGGEV    SLC_FC_FUNC(dggev, DGGEV)
#define SLC_DGEGS    SLC_FC_FUNC(dgegs, DGEGS)
#define SLC_DGEES    SLC_FC_FUNC(dgees, DGEES)
#define SLC_DGEEV    SLC_FC_FUNC(dgeev, DGEEV)
#define SLC_DGEBAL   SLC_FC_FUNC(dgebal, DGEBAL)
#define SLC_DGGBAL   SLC_FC_FUNC(dggbal, DGGBAL)
#define SLC_DGEHRD   SLC_FC_FUNC(dgehrd, DGEHRD)
#define SLC_DORGHR   SLC_FC_FUNC(dorghr, DORGHR)
#define SLC_DORMHR   SLC_FC_FUNC(dormhr, DORMHR)
#define SLC_DHSEQR   SLC_FC_FUNC(dhseqr, DHSEQR)
#define SLC_DLAHQR   SLC_FC_FUNC(dlahqr, DLAHQR)
#define SLC_ZLAHQR   SLC_FC_FUNC(zlahqr, ZLAHQR)
#define SLC_DHGEQZ   SLC_FC_FUNC(dhgeqz, DHGEQZ)
#define SLC_ZHGEQZ   SLC_FC_FUNC(zhgeqz, ZHGEQZ)
#define SLC_DTREVC   SLC_FC_FUNC(dtrevc, DTREVC)
#define SLC_DTREXC   SLC_FC_FUNC(dtrexc, DTREXC)
#define SLC_ZTREXC   SLC_FC_FUNC(ztrexc, ZTREXC)
#define SLC_DGEBRD   SLC_FC_FUNC(dgebrd, DGEBRD)
#define SLC_DORGBR   SLC_FC_FUNC(dorgbr, DORGBR)
#define SLC_DBDSQR   SLC_FC_FUNC(dbdsqr, DBDSQR)
#define SLC_DGELS    SLC_FC_FUNC(dgels, DGELS)
#define SLC_DGELSS   SLC_FC_FUNC(dgelss, DGELSS)
#define SLC_DLANTR   SLC_FC_FUNC(dlantr, DLANTR)
#define SLC_DTRCON   SLC_FC_FUNC(dtrcon, DTRCON)
#define SLC_DGEBAK   SLC_FC_FUNC(dgebak, DGEBAK)
#define SLC_DPOTRS   SLC_FC_FUNC(dpotrs, DPOTRS)
#define SLC_DPPTRS   SLC_FC_FUNC(dpptrs, DPPTRS)
#define SLC_DPPTRI   SLC_FC_FUNC(dpptri, DPPTRI)
#define SLC_DPTTRF   SLC_FC_FUNC(dpttrf, DPTTRF)
#define SLC_DPTTRS   SLC_FC_FUNC(dpttrs, DPTTRS)
#define SLC_DSPMV    SLC_FC_FUNC(dspmv, DSPMV)
#define SLC_DSPR     SLC_FC_FUNC(dspr, DSPR)
#define SLC_DTRTRI   SLC_FC_FUNC(dtrtri, DTRTRI)
#define SLC_DTRTRS   SLC_FC_FUNC(dtrtrs, DTRTRS)
#define SLC_DGELSY   SLC_FC_FUNC(dgelsy, DGELSY)
#define SLC_DTZRZF   SLC_FC_FUNC(dtzrzf, DTZRZF)
#define SLC_DORMRZ   SLC_FC_FUNC(dormrz, DORMRZ)
#define SLC_ILAENV   SLC_FC_FUNC(ilaenv, ILAENV)
#define SLC_DGEQLF   SLC_FC_FUNC(dgeqlf, DGEQLF)
#define SLC_DORMQL   SLC_FC_FUNC(dormql, DORMQL)
#define SLC_DTRCON   SLC_FC_FUNC(dtrcon, DTRCON)
#define SLC_DLAEXC   SLC_FC_FUNC(dlaexc, DLAEXC)
#define SLC_DLACN2   SLC_FC_FUNC(dlacn2, DLACN2)
#define SLC_DLATRS   SLC_FC_FUNC(dlatrs, DLATRS)
#define SLC_DGGHRD   SLC_FC_FUNC(dgghrd, DGGHRD)
#define SLC_DTGEVC   SLC_FC_FUNC(dtgevc, DTGEVC)
#define SLC_DGGBAK   SLC_FC_FUNC(dggbak, DGGBAK)
#define SLC_ZLANHS   SLC_FC_FUNC(zlanhs, ZLANHS)
#define SLC_ZLANTR   SLC_FC_FUNC(zlantr, ZLANTR)
#define SLC_ZLARNV   SLC_FC_FUNC(zlarnv, ZLARNV)
#define SLC_ZDOTU    SLC_FC_FUNC(zdotu, ZDOTU)
#define SLC_ZLADIV   SLC_FC_FUNC(zladiv, ZLADIV)
#define SLC_DTGSEN   SLC_FC_FUNC(dtgsen, DTGSEN)
#define SLC_DGEQR2   SLC_FC_FUNC(dgeqr2, DGEQR2)
#define SLC_DORG2R   SLC_FC_FUNC(dorg2r, DORG2R)
#define SLC_DORGR2   SLC_FC_FUNC(dorgr2, DORGR2)
#define SLC_DORM2R   SLC_FC_FUNC(dorm2r, DORM2R)
#define SLC_DTGEX2   SLC_FC_FUNC(dtgex2, DTGEX2)
#define SLC_DTGEXC   SLC_FC_FUNC(dtgexc, DTGEXC)
#define SLC_ZTGEXC   SLC_FC_FUNC(ztgexc, ZTGEXC)
#define SLC_DLASRT   SLC_FC_FUNC(dlasrt, DLASRT)
#define SLC_DTRSEN   SLC_FC_FUNC(dtrsen, DTRSEN)
#define SLC_ZGGBAL   SLC_FC_FUNC(zggbal, ZGGBAL)
#define SLC_ZGGHRD   SLC_FC_FUNC(zgghrd, ZGGHRD)
#define SLC_ZGGBAK   SLC_FC_FUNC(zggbak, ZGGBAK)
#define SLC_ZTGEVC   SLC_FC_FUNC(ztgevc, ZTGEVC)

#ifdef __cplusplus
}
#endif

#endif /* SLICOT_BLAS_H */
