/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#ifndef SLICOT_MB03_H
#define SLICOT_MB03_H

#include "../slicot_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute smallest singular value of A - jwI.
 *
 * Helper routine for AB13FD. Computes sigma_min(A - jwI) using SVD.
 * If omega=0, uses real SVD (DGESVD). Otherwise uses complex SVD (ZGESVD).
 *
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in] omega Frequency constant w
 * @param[in,out] a N-by-N matrix A. Destroyed if omega=0.
 * @param[in] lda Leading dimension of a (lda >= max(1, n))
 * @param[out] s Singular values in decreasing order, dimension (n)
 * @param[out] dwork Real workspace, dimension (ldwork). dwork[0] returns optimal size.
 * @param[in] ldwork Real workspace size (>= max(1, 5*n))
 * @param[out] cwork Complex workspace, dimension (lcwork). cwork[0] returns optimal size.
 * @param[in] lcwork Complex workspace size (>= 1 if omega=0, >= max(1, n*n+3*n) otherwise)
 * @param[out] info 0=success, -i=param i invalid, 2=SVD failed to converge
 * @return Smallest singular value of A - jwI
 */
f64 mb03ny(i32 n, f64 omega, f64 *a, i32 lda, f64 *s, f64 *dwork, i32 ldwork,
           c128 *cwork, i32 lcwork, i32 *info);

/**
 * @brief Incremental rank estimation for QR factorization.
 *
 * Computes (optionally) a rank-revealing QR factorization of real M-by-N matrix A
 * and estimates effective rank using incremental condition estimation.
 *
 * Uses QR with column pivoting: A*P = Q*R, where R = [R11 R12; 0 R22].
 * R11 is largest leading submatrix with estimated condition < 1/RCOND.
 * Order of R11 (RANK) is effective rank of A.
 *
 * @param[in] jobqr 'Q' = perform QR factorization, 'N' = use existing R in A
 * @param[in] m Number of rows of A (m >= 0)
 * @param[in] n Number of columns of A (n >= 0)
 * @param[in,out] a DOUBLE PRECISION array, dimension (lda,n)
 *                  If JOBQR='Q': In: M-by-N matrix A, Out: QR factorization
 *                  If JOBQR='N': In/Out: Upper triangular R from QR
 * @param[in] lda Leading dimension of A (lda >= max(1,m))
 * @param[in,out] jpvt INTEGER array, dimension (n)
 *                     If JOBQR='Q': In: initial column flags (0=free),
 *                                   Out: pivot permutation
 *                     If JOBQR='N': not referenced
 * @param[in] rcond Rank threshold: condition < 1/RCOND (rcond >= 0)
 * @param[in] svlmax Largest singular value estimate of parent matrix (svlmax >= 0)
 *                   Use 0 if A is standalone
 * @param[out] tau DOUBLE PRECISION array, dimension (min(m,n))
 *                 Scalar factors of elementary reflectors (if JOBQR='Q')
 * @param[out] rank Effective rank of A
 * @param[out] sval DOUBLE PRECISION array, dimension (3)
 *                  sval[0]: largest singular value of R(1:rank,1:rank)
 *                  sval[1]: smallest singular value of R(1:rank,1:rank)
 *                  sval[2]: smallest singular value of R(1:rank+1,1:rank+1)
 * @param[out] dwork DOUBLE PRECISION array, dimension (ldwork)
 *                   On exit, dwork[0] = optimal ldwork
 * @param[in] ldwork Length of dwork
 *                   JOBQR='Q': ldwork >= 3*n+1 (prefer 2*n+(n+1)*NB)
 *                   JOBQR='N': ldwork >= max(1,2*min(m,n))
 *                   If ldwork=-1, workspace query
 * @param[out] info Exit code (0=success, <0=invalid parameter -info)
 */
void mb03od(
    const char* jobqr,
    const i32 m,
    const i32 n,
    f64* a,
    const i32 lda,
    i32* jpvt,
    const f64 rcond,
    const f64 svlmax,
    f64* tau,
    i32* rank,
    f64* sval,
    f64* dwork,
    const i32 ldwork,
    i32* info
);

/**
 * @brief Matrix rank determination by incremental condition estimation during QR factorization.
 *
 * Computes a rank-revealing QR factorization of a real general M-by-N matrix A,
 * which may be rank-deficient, and estimates its effective rank using incremental
 * condition estimation.
 *
 * The routine uses a truncated QR factorization with column pivoting
 * A * P = Q * R, where R = [R11 R12; 0 R22], with R11 defined as the largest
 * leading upper triangular submatrix whose estimated condition number is less
 * than 1/RCOND. The order of R11, RANK, is the effective rank of A.
 *
 * @param[in] m Number of rows of matrix A. m >= 0.
 * @param[in] n Number of columns of matrix A. n >= 0.
 * @param[in,out] a DOUBLE PRECISION array, dimension (lda, n)
 *                  On entry: M-by-N matrix A
 *                  On exit: RANK-by-RANK upper triangular R11 and QR factorization data
 * @param[in] lda Leading dimension of array A. lda >= max(1,m).
 * @param[in] rcond Threshold for rank determination (0 <= rcond <= 1)
 * @param[in] svlmax Estimate of largest singular value of parent matrix (>= 0)
 * @param[out] rank Effective rank of A
 * @param[out] sval DOUBLE PRECISION array, dimension (3)
 *                  Singular value estimates: [largest, rank-th, (rank+1)-th]
 * @param[out] jpvt INTEGER array, dimension (n)
 *                  Pivot indices (1-based, Fortran style)
 * @param[out] tau DOUBLE PRECISION array, dimension (min(m,n))
 *                 Scalar factors of elementary reflectors
 * @param[out] dwork DOUBLE PRECISION array, dimension (3*n-1)
 * @param[out] info Exit code (0 = success, <0 = invalid parameter)
 */
void mb03oy(i32 m, i32 n, f64* a, i32 lda, f64 rcond,
            f64 svlmax, i32* rank, f64* sval, i32* jpvt,
            f64* tau, f64* dwork, i32* info);

/**
 * @brief Rank-revealing RQ factorization with row pivoting.
 *
 * Computes a truncated RQ factorization with row pivoting:
 *
 *   P * A = R * Q,  where  R = [ R11  R12 ]
 *                              [  0   R22 ]
 *
 * with R22 defined as the largest trailing upper triangular submatrix
 * whose estimated condition number is less than 1/RCOND. The order of
 * R22, RANK, is the effective rank of A.
 *
 * @param[in] m Number of rows of A (m >= 0)
 * @param[in] n Number of columns of A (n >= 0)
 * @param[in,out] a On entry, m-by-n matrix. On exit, upper triangle of
 *                  A(m-rank+1:m, n-rank+1:n) contains R22.
 * @param[in] lda Leading dimension of a (lda >= max(1,m))
 * @param[in] rcond Tolerance for rank decisions (0 <= rcond <= 1)
 * @param[in] svlmax Estimate of largest singular value of parent matrix,
 *                   or 0 if A is standalone
 * @param[out] rank Effective rank of A (order of R22)
 * @param[out] sval Array dimension (3) with singular value estimates:
 *                  sval[0] = largest sv of R22,
 *                  sval[1] = smallest sv of R22,
 *                  sval[2] = smallest sv of R(m-rank:m, n-rank:n)
 * @param[out] jpvt INTEGER array, dimension (m). Row permutation indices.
 * @param[out] tau DOUBLE PRECISION array, dimension (min(m,n))
 * @param[out] dwork DOUBLE PRECISION array, dimension (3*m-1)
 * @param[out] info Exit code (0 = success, <0 = invalid parameter)
 */
void mb03py(i32 m, i32 n, f64* a, i32 lda, f64 rcond,
            f64 svlmax, i32* rank, f64* sval, i32* jpvt,
            f64* tau, f64* dwork, i32* info);

/**
 * @brief Rank-revealing RQ factorization with row pivoting.
 *
 * Computes (optionally) a rank-revealing RQ factorization of a real
 * M-by-N matrix A which may be rank-deficient, using incremental
 * condition estimation: P * A = R * Q, where R22 is the largest
 * trailing submatrix with estimated condition number < 1/RCOND.
 *
 * @param[in] jobrq 'R' = perform RQ factorization with row pivoting,
 *                  'N' = assume factorization already done
 * @param[in] m Number of rows of A (m >= 0)
 * @param[in] n Number of columns of A (n >= 0)
 * @param[in,out] a On entry (jobrq='R'), the M-by-N matrix A.
 *                  On exit, contains R and Householder reflectors.
 * @param[in] lda Leading dimension of a (lda >= max(1,m))
 * @param[in,out] jpvt On entry (jobrq='R'), if jpvt[i]!=0 row i is final.
 *                     On exit, jpvt[i]=k means row i of P*A was row k of A.
 * @param[in] rcond Tolerance for rank decisions (rcond >= 0)
 * @param[in] svlmax Estimate of largest singular value, or 0 if standalone
 * @param[out] tau Scalar factors of elementary reflectors, dim (min(m,n))
 * @param[out] rank Effective rank (order of R22)
 * @param[out] sval Singular value estimates, dimension (3)
 * @param[out] dwork Workspace, dim (3*m if jobrq='R', 3*min(m,n) if 'N')
 * @param[out] info 0 = success, <0 = invalid parameter
 */
void mb03pd(const char *jobrq, i32 m, i32 n, f64 *a, i32 lda, i32 *jpvt,
            f64 rcond, f64 svlmax, f64 *tau, i32 *rank, f64 *sval,
            f64 *dwork, i32 *info);

/**
 * @brief Reorder eigenvalues in quasi-triangular matrix.
 *
 * Reorders the diagonal blocks of a real upper quasi-triangular matrix
 * so that a selected group of eigenvalues appears in the leading diagonal
 * blocks. The leading columns of the orthogonal transformation matrix U
 * are updated to span an invariant subspace corresponding to the selected
 * eigenvalues.
 *
 * @param[in] dico Type of system: 'C' = continuous, 'D' = discrete
 * @param[in] stdom Stability domain: 'S' = stable, 'U' = unstable
 * @param[in] jobu Update transformation: 'I' = initialize, 'U' = update
 * @param[in] n Order of matrix A (n >= 1)
 * @param[in] nlow Lowest index of block to consider (1 <= nlow <= nsup)
 * @param[in] nsup Highest index of block (nlow <= nsup <= n)
 * @param[in] alpha Stability boundary:
 *                  Continuous: eigenvalues with Re(lambda) < alpha are stable
 *                  Discrete: eigenvalues with |lambda| < alpha are stable
 * @param[in,out] a N-by-N quasi-triangular matrix, dimension (lda,n)
 * @param[in] lda Leading dimension of A (lda >= n)
 * @param[in,out] u N-by-N orthogonal matrix U, dimension (ldu,n)
 * @param[in] ldu Leading dimension of U (ldu >= n)
 * @param[out] ndim Dimension of specified invariant subspace
 * @param[out] dwork Workspace, dimension at least (n)
 * @param[out] info Exit code: 0 = success, 1 = bad block boundary,
 *                  2 = reordering failed
 */
void mb03qd(const char* dico, const char* stdom, const char* jobu,
            i32 n, i32 nlow, i32 nsup, f64 alpha, f64* a, i32 lda,
            f64* u, i32 ldu, i32* ndim, f64* dwork, i32* info);

/**
 * @brief Compute eigenvalues of upper quasi-triangular matrix.
 *
 * Computes the eigenvalues of an upper quasi-triangular matrix T.
 * Complex conjugate pairs appear as 2x2 diagonal blocks.
 *
 * @param[in] n Order of the matrix T (N >= 0)
 * @param[in] t Upper quasi-triangular matrix, dimension (LDT,N)
 * @param[in] ldt Leading dimension of T (LDT >= max(1,N))
 * @param[out] wr Real parts of eigenvalues, dimension (N)
 * @param[out] wi Imaginary parts of eigenvalues, dimension (N)
 * @param[out] info 0=success, <0=-i means i-th argument invalid
 */
void mb03qx(i32 n, const f64* t, i32 ldt, f64* wr, f64* wi, i32* info);

/**
 * @brief Process 2x2 diagonal block of quasi-triangular matrix.
 *
 * Computes eigenvalues of a selected 2-by-2 diagonal block of an upper
 * quasi-triangular matrix. Reduces the block to standard form and splits
 * it in case of real eigenvalues by constructing an orthogonal transformation.
 *
 * @param[in] n Order of matrices A and U. N >= 2.
 * @param[in] l Position of the block (1 <= L < N). 1-based index.
 * @param[in,out] a Upper quasi-triangular matrix. On exit, transformed.
 * @param[in] lda Leading dimension of A (LDA >= N).
 * @param[in,out] u Transformation matrix. On exit, U*UT where UT is the
 *                  orthogonal transformation used.
 * @param[in] ldu Leading dimension of U (LDU >= N).
 * @param[out] e1 Real part (or first real eigenvalue).
 * @param[out] e2 Imaginary part (or second real eigenvalue).
 * @param[out] info 0=success, <0=-i means i-th argument invalid.
 */
void mb03qy(i32 n, i32 l, f64* a, i32 lda, f64* u, i32 ldu,
            f64* e1, f64* e2, i32* info);

/**
 * @brief Reduce real Schur form matrix to block-diagonal form.
 *
 * Reduces a matrix A in real Schur form to a block-diagonal form using
 * well-conditioned non-orthogonal similarity transformations. The condition
 * numbers of the transformations used for reduction are roughly bounded by
 * PMAX. The transformations are optionally postmultiplied in a given matrix X.
 * The real Schur form is optionally ordered, so that clustered eigenvalues
 * are grouped in the same block.
 *
 * @param[in] jobx Specifies whether transformations are accumulated:
 *                 'N': not accumulated
 *                 'U': accumulated in X (X is updated)
 * @param[in] sort Specifies whether diagonal blocks are reordered:
 *                 'N': not reordered
 *                 'S': reordered before each step for clustering
 *                 'C': not reordered but closest-neighbor strategy used
 *                 'B': both reordering and closest-neighbor strategy
 * @param[in] n Order of matrices A and X (n >= 0)
 * @param[in] pmax Upper bound for infinity norm of elementary submatrices
 *                 of individual transformations (pmax >= 1.0)
 * @param[in,out] a DOUBLE PRECISION array, dimension (lda,n)
 *                  On entry: matrix in real Schur form
 *                  On exit: block-diagonal matrix in real Schur canonical form
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] x DOUBLE PRECISION array, dimension (ldx,n)
 *                  On entry (if jobx='U'): given matrix X
 *                  On exit (if jobx='U'): product X * transformation matrix
 *                  Not referenced if jobx='N'
 * @param[in] ldx Leading dimension of X (ldx >= 1, or ldx >= max(1,n) if jobx='U')
 * @param[out] nblcks Number of diagonal blocks
 * @param[out] blsize INTEGER array, dimension (n)
 *                    First NBLCKS elements contain block orders
 * @param[out] wr DOUBLE PRECISION array, dimension (n)
 *                Real parts of eigenvalues
 * @param[out] wi DOUBLE PRECISION array, dimension (n)
 *                Imaginary parts of eigenvalues
 * @param[in] tol Tolerance for clustering (used if sort='S' or 'B'):
 *                > 0: absolute tolerance
 *                < 0: relative tolerance (|tol| * max|lambda_j|)
 *                = 0: default sqrt(sqrt(eps)) relative tolerance
 * @param[out] dwork Workspace, dimension (n)
 * @param[out] info Exit code:
 *                  = 0: success
 *                  < 0: if info = -i, the i-th argument had illegal value
 */
void mb03rd(
    const char* jobx,
    const char* sort,
    const i32 n,
    const f64 pmax,
    f64* a,
    const i32 lda,
    f64* x,
    const i32 ldx,
    i32* nblcks,
    i32* blsize,
    f64* wr,
    f64* wi,
    const f64 tol,
    f64* dwork,
    i32* info
);

/**
 * @brief Reorder diagonal blocks of a real Schur form matrix.
 *
 * Reorders the diagonal blocks of the principal submatrix between indices
 * KL and KU of a real Schur form matrix A using orthogonal similarity
 * transformations, such that the block specified by KU is moved to position KL.
 *
 * The transformations are optionally postmultiplied in a given matrix X.
 *
 * @param[in] jobv 'N': no accumulation, 'V': accumulate in X
 * @param[in] n Order of the matrices A and X (N >= 0)
 * @param[in] kl Lower boundary index and target position (1 <= KL <= KU <= N)
 * @param[in,out] ku Upper boundary index and source position. On exit, may be
 *                   incremented by 1 if a 2x2 block was split into two 1x1 blocks
 * @param[in,out] a Real Schur form matrix, dimension (LDA,N)
 * @param[in] lda Leading dimension of A (LDA >= max(1,N))
 * @param[in,out] x Transformation matrix if JOBV='V', dimension (LDX,N)
 * @param[in] ldx Leading dimension of X (LDX >= 1, or >= max(1,N) if JOBV='V')
 * @param[in,out] wr Real parts of eigenvalues, dimension (N)
 * @param[in,out] wi Imaginary parts of eigenvalues, dimension (N)
 * @param[out] dwork Workspace array, dimension (N)
 */
void mb03rx(const char* jobv, i32 n, i32 kl, i32* ku, f64* a, i32 lda,
            f64* x, i32 ldx, f64* wr, f64* wi, f64* dwork);

/**
 * @brief Solve Sylvester equation -AX + XB = C with norm bound.
 *
 * Solves the Sylvester equation -AX + XB = C, where A (M-by-M) and B (N-by-N)
 * are matrices in real Schur form. The solution X overwrites C.
 *
 * This routine is intended to be called only by MB03RD. For efficiency,
 * computations are aborted when the infinity norm of an elementary submatrix
 * of X exceeds PMAX.
 *
 * @param[in] m Order of matrix A and number of rows of C (m >= 0)
 * @param[in] n Order of matrix B and number of columns of C (n >= 0)
 * @param[in] pmax Upper bound for infinity norm of elementary submatrices of X
 * @param[in] a DOUBLE PRECISION array, dimension (lda,m)
 *              The matrix A in real Schur form
 * @param[in] lda Leading dimension of A (lda >= max(1,m))
 * @param[in] b DOUBLE PRECISION array, dimension (ldb,n)
 *              The matrix B in real Schur form
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in,out] c DOUBLE PRECISION array, dimension (ldc,n)
 *                  On entry: the right-hand side matrix C
 *                  On exit: the solution matrix X (if info=0)
 * @param[in] ldc Leading dimension of C (ldc >= max(1,m))
 * @param[out] info Exit code:
 *                  = 0: success
 *                  = 1: elementary submatrix had norm > PMAX (aborted)
 */
void mb03ry(
    const i32 m,
    const i32 n,
    const f64 pmax,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    i32* info
);

/**
 * @brief Compute singular value decomposition of upper triangular matrix.
 *
 * Computes SVD of N-by-N upper triangular matrix: A = Q*S*P', where Q and P
 * are orthogonal matrices and S is diagonal with non-negative singular values
 * in descending order. Uses bidiagonalization followed by QR algorithm.
 *
 * @param[in] jobq 'V' to compute left singular vectors Q, 'N' otherwise
 * @param[in] jobp 'V' to compute right singular vectors P', 'N' otherwise
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in,out] a DOUBLE PRECISION array, dimension (lda,n)
 *                  In: upper triangular matrix A
 *                  Out: if jobp='V', orthogonal matrix P'; otherwise workspace
 * @param[in] lda Leading dimension (lda >= max(1,n))
 * @param[out] q DOUBLE PRECISION array, dimension (ldq,n)
 *               If jobq='V', contains orthogonal matrix Q (left singular vectors)
 * @param[in] ldq Leading dimension (ldq >= max(1,n) if jobq='V', ldq >= 1 otherwise)
 * @param[out] sv DOUBLE PRECISION array, dimension (n)
 *                Singular values in descending order
 * @param[out] dwork DOUBLE PRECISION array, dimension (ldwork)
 *                   On exit: dwork[0] = optimal ldwork
 *                   If info > 0: dwork[1:n-1] = unconverged superdiagonals
 * @param[in] ldwork Workspace size (ldwork >= max(1,5*n), or -1 for query)
 * @param[out] info Exit code:
 *                  = 0: success
 *                  < 0: if info = -i, i-th argument invalid
 *                  > 0: QR algorithm failed to converge (info = # unconverged superdiagonals)
 */
i32 mb03ud(char jobq, char jobp, i32 n, f64 *a, i32 lda, f64 *q, i32 ldq,
           f64 *sv, f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Reduce product of p matrices to periodic Hessenberg form.
 *
 * Reduces a product of p real general matrices A = A_1*A_2*...*A_p
 * to upper Hessenberg form H = H_1*H_2*...*H_p, where H_1 is upper
 * Hessenberg, and H_2, ..., H_p are upper triangular, by using
 * orthogonal similarity transformations:
 *
 *   Q_1' * A_1 * Q_2 = H_1
 *   Q_2' * A_2 * Q_3 = H_2
 *   ...
 *   Q_p' * A_p * Q_1 = H_p
 *
 * @param[in] n Order of the square matrices A_j (n >= 0)
 * @param[in] p Number of matrices in the product (p >= 1)
 * @param[in] ilo Lower index for reduction (1 <= ilo <= max(1,n))
 * @param[in] ihi Upper index for reduction (min(ilo,n) <= ihi <= n)
 * @param[in,out] a Array dimension (lda1, lda2, p). On entry, A(*,*,j)
 *                  contains A_j. On exit, upper Hessenberg form in A(*,*,1)
 *                  and upper triangular in A(*,*,j) for j > 1. Below diagonal
 *                  elements contain Householder vectors.
 * @param[in] lda1 First leading dimension of a (lda1 >= max(1,n))
 * @param[in] lda2 Second leading dimension of a (lda2 >= max(1,n))
 * @param[out] tau Array dimension (ldtau, p). Contains scalar factors of
 *                 elementary reflectors for Q_j matrices.
 * @param[in] ldtau Leading dimension of tau (ldtau >= max(1,n-1))
 * @param[out] dwork Workspace array dimension (n)
 * @param[out] info 0 = success, < 0 = -i means i-th argument invalid
 */
void mb03vd(i32 n, i32 p, i32 ilo, i32 ihi, f64* a, i32 lda1, i32 lda2,
            f64* tau, i32 ldtau, f64* dwork, i32* info);

/**
 * @brief Generate orthogonal matrices from periodic Hessenberg reduction.
 *
 * Generates the real orthogonal matrices Q_1, Q_2, ..., Q_p, which are
 * defined as the product of ihi-ilo elementary reflectors of order n,
 * as returned by MB03VD:
 *
 *   Q_j = H_j(ilo) H_j(ilo+1) ... H_j(ihi-1)
 *
 * Uses LAPACK's DORGHR for Q_1 and DORGQR for Q_2, ..., Q_p.
 *
 * @param[in] n Order of the square matrices Q_j (n >= 0)
 * @param[in] p Number of transformation matrices (p >= 1)
 * @param[in] ilo Lower index from MB03VD (1 <= ilo <= max(1,n))
 * @param[in] ihi Upper index from MB03VD (min(ilo,n) <= ihi <= n)
 * @param[in,out] a Array dimension (lda1, lda2, p). On entry, contains
 *                  Householder vectors from MB03VD. On exit, A(*,*,j)
 *                  contains the orthogonal matrix Q_j.
 * @param[in] lda1 First leading dimension of a (lda1 >= max(1,n))
 * @param[in] lda2 Second leading dimension of a (lda2 >= max(1,n))
 * @param[in] tau Array dimension (ldtau, p). Scalar factors of reflectors.
 * @param[in] ldtau Leading dimension of tau (ldtau >= max(1,n-1))
 * @param[out] dwork Workspace array dimension (ldwork)
 * @param[in] ldwork Workspace size (ldwork >= max(1,n)). Use -1 for query.
 * @param[out] info 0 = success, < 0 = -i means i-th argument invalid
 */
void mb03vy(i32 n, i32 p, i32 ilo, i32 ihi, f64* a, i32 lda1, i32 lda2,
            f64* tau, i32 ldtau, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Reduce periodic matrix product to Hessenberg-triangular form (unblocked).
 *
 * Reduces the generalized matrix product A(:,:,1)^S(1) * ... * A(:,:,K)^S(K)
 * to upper Hessenberg-triangular form. The H-th matrix is reduced to upper
 * Hessenberg form while others are triangularized.
 *
 * @param[in] compq 'N'=no Q, 'U'=update Q, 'I'=init Q to I, 'P'=use QIND
 * @param[in] qind Array dim(k), controls partial Q generation when COMPQ='P'
 * @param[in] triu 'N'=only neg signature matrices, 'A'=all N-1 matrices
 * @param[in] n Order of each factor (n >= 0)
 * @param[in] k Number of factors (k >= 0)
 * @param[in,out] h On entry: which factor to make Hessenberg (1..K or auto).
 *                  On exit: index of Hessenberg factor
 * @param[in] ilo Lower bound of active submatrix (1-based)
 * @param[in] ihi Upper bound of active submatrix (1-based)
 * @param[in] s Signature array dim(k), each element is 1 or -1
 * @param[in,out] a Array dim(lda1,lda2,k). Input: factors. Output: reduced form.
 * @param[in] lda1 First leading dimension of a (lda1 >= max(1,n))
 * @param[in] lda2 Second leading dimension of a (lda2 >= max(1,n))
 * @param[in,out] q Array dim(ldq1,ldq2,k). Orthogonal factors.
 * @param[in] ldq1 First leading dimension of q
 * @param[in] ldq2 Second leading dimension of q
 * @param[out] iwork Integer workspace dim(liwork)
 * @param[in] liwork Length of iwork (liwork >= max(1,3*k))
 * @param[out] dwork Real workspace dim(ldwork)
 * @param[in] ldwork Length of dwork. Use -1 for query.
 * @param[out] info 0=success, <0=-i means i-th arg invalid
 */
void mb03vw(const char *compq, const i32 *qind, const char *triu,
            i32 n, i32 k, i32 *h, i32 ilo, i32 ihi, const i32 *s,
            f64 *a, i32 lda1, i32 lda2, f64 *q, i32 ldq1, i32 ldq2,
            i32 *iwork, i32 liwork, f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Periodic Schur decomposition of a product of matrices.
 *
 * Computes the Schur decomposition and eigenvalues of a product of matrices
 * H = H_1*H_2*...*H_p, with H_1 upper Hessenberg and H_2,...,H_p upper
 * triangular, without evaluating the product. Specifically, computes
 * orthogonal matrices Z_i such that:
 *
 *   Z_1' * H_1 * Z_2 = T_1   (real Schur form)
 *   Z_2' * H_2 * Z_3 = T_2   (upper triangular)
 *   ...
 *   Z_p' * H_p * Z_1 = T_p   (upper triangular)
 *
 * Uses a refined periodic QR algorithm.
 *
 * @param[in] job 'E' = eigenvalues only, 'S' = full Schur form
 * @param[in] compz 'N' = no Z, 'I' = initialize Z to I, 'V' = update Z
 * @param[in] n Order of the matrices (n >= 0)
 * @param[in] p Number of matrices in product (p >= 1)
 * @param[in] ilo Lower index of active submatrix (1 <= ilo <= max(1,n))
 * @param[in] ihi Upper index of active submatrix (min(ilo,n) <= ihi <= n)
 * @param[in] iloz Lower row index for Z updates (1 <= iloz <= ilo)
 * @param[in] ihiz Upper row index for Z updates (ihi <= ihiz <= n)
 * @param[in,out] h 3D array (ldh1,ldh2,p). On entry, H(:,:,1) is upper
 *                  Hessenberg, H(:,:,j) for j>1 are upper triangular.
 *                  On exit with job='S', contains Schur factors T_i.
 * @param[in] ldh1 First leading dimension of h (ldh1 >= max(1,n))
 * @param[in] ldh2 Second leading dimension of h (ldh2 >= max(1,n))
 * @param[in,out] z 3D array (ldz1,ldz2,p). If compz='V', contains initial
 *                  transformations on entry. On exit, contains accumulated
 *                  transformations if compz='I' or 'V'.
 * @param[in] ldz1 First leading dimension of z
 * @param[in] ldz2 Second leading dimension of z
 * @param[out] wr Real parts of eigenvalues (dimension n)
 * @param[out] wi Imaginary parts of eigenvalues (dimension n)
 * @param[out] dwork Workspace (dimension >= ihi-ilo+p-1)
 * @param[in] ldwork Workspace size
 * @param[out] info 0 = success, < 0 = -i means i-th argument invalid,
 *                  > 0 = QR failed to converge (value is failing index)
 */
void mb03wd(const char* job, const char* compz, i32 n, i32 p, i32 ilo, i32 ihi,
            i32 iloz, i32 ihiz, f64* h, i32 ldh1, i32 ldh2, f64* z, i32 ldz1,
            i32 ldz2, f64* wr, f64* wi, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Compute eigenvalues of Hamiltonian matrix.
 *
 * Computes the eigenvalues of a Hamiltonian matrix:
 *     H = [A, G; Q, -A'] where G=G' and Q=Q'
 *
 * Due to structure, eigenvalues appear in pairs (lambda, -lambda).
 * Uses symplectic URV and periodic Schur decompositions:
 *     U' * H * V = [T, G; 0, S]
 * where U, V are orthogonal symplectic, S is real Schur form, T is upper triangular.
 *
 * @param[in] balanc 'N' = no balancing, 'P' = permute, 'S' = scale, 'B' = both
 * @param[in] job 'E' = eigenvalues only, 'S' = Schur form (T,S), 'G' = full (T,S,G)
 * @param[in] jobu 'N' = don't compute U, 'U' = compute U
 * @param[in] jobv 'N' = don't compute V, 'V' = compute V
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in,out] a Matrix A, dimension (lda,n). On exit: Schur matrix S if JOB='S'/'G'
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] qg Matrix containing Q (cols 1:n lower) and G (cols 2:n+1 upper),
 *                   dimension (ldqg,n+1). On exit: G of decomposition if JOB='G'
 * @param[in] ldqg Leading dimension of QG (ldqg >= max(1,n))
 * @param[out] t Upper triangular T of decomposition, dimension (ldt,n)
 * @param[in] ldt Leading dimension of T (ldt >= max(1,n))
 * @param[out] u1 (1,1) block of orthogonal symplectic U, dimension (ldu1,n)
 * @param[in] ldu1 Leading dimension of U1 (ldu1 >= 1, >= n if JOBU='U')
 * @param[out] u2 (2,1) block of orthogonal symplectic U, dimension (ldu2,n)
 * @param[in] ldu2 Leading dimension of U2 (ldu2 >= 1, >= n if JOBU='U')
 * @param[out] v1 (1,1) block of orthogonal symplectic V, dimension (ldv1,n)
 * @param[in] ldv1 Leading dimension of V1 (ldv1 >= 1, >= n if JOBV='V')
 * @param[out] v2 (2,1) block of orthogonal symplectic V, dimension (ldv2,n)
 * @param[in] ldv2 Leading dimension of V2 (ldv2 >= 1, >= n if JOBV='V')
 * @param[out] wr Real parts of eigenvalues with nonnegative imaginary part, dimension (n)
 * @param[out] wi Imaginary parts of eigenvalues, dimension (n)
 * @param[out] ilo Index from balancing, balanced A(i,j)=0 for i>j, j<ilo
 * @param[out] scale Scaling factors from balancing, dimension (n)
 * @param[out] dwork Workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size (depends on job/jobu/jobv, minimum 2)
 * @param[out] info 0=success, <0=-i invalid arg, >0=i failed to converge
 */
void mb03xd(const char *balanc, const char *job, const char *jobu,
            const char *jobv, i32 n, f64 *a, i32 lda, f64 *qg, i32 ldqg,
            f64 *t, i32 ldt, f64 *u1, i32 ldu1, f64 *u2, i32 ldu2,
            f64 *v1, i32 ldv1, f64 *v2, i32 ldv2,
            f64 *wr, f64 *wi, i32 *ilo, f64 *scale,
            f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Compute periodic Schur decomposition of A*B product.
 *
 * Computes the periodic Schur decomposition and eigenvalues of a matrix
 * product H = A*B, with A upper Hessenberg and B upper triangular, without
 * evaluating the product. Specifically computes orthogonal Q and Z such that:
 *   Q' * A * Z = S (real Schur form)
 *   Z' * B * Q = T (upper triangular)
 *
 * @param[in] job 'E' = eigenvalues only, 'S' = Schur form
 * @param[in] compq 'N' = Q not needed, 'I' = Q initialized to I, 'V' = accumulate
 * @param[in] compz 'N' = Z not needed, 'I' = Z initialized to I, 'V' = accumulate
 * @param[in] n Order of matrices A and B (N >= 0)
 * @param[in] ilo Index for active block start (1 <= ILO <= max(1,N+1))
 * @param[in] ihi Index for active block end (min(ILO,N) <= IHI <= N)
 * @param[in,out] a Upper Hessenberg matrix, dimension (LDA,N)
 * @param[in] lda Leading dimension of A (LDA >= max(1,N))
 * @param[in,out] b Upper triangular matrix, dimension (LDB,N)
 * @param[in] ldb Leading dimension of B (LDB >= max(1,N))
 * @param[in,out] q Orthogonal factor Q, dimension (LDQ,N)
 * @param[in] ldq Leading dimension of Q (LDQ >= 1, or N if COMPQ != 'N')
 * @param[in,out] z Orthogonal factor Z, dimension (LDZ,N)
 * @param[in] ldz Leading dimension of Z (LDZ >= 1, or N if COMPZ != 'N')
 * @param[out] alphar Real parts of eigenvalue numerators, dimension (N)
 * @param[out] alphai Imaginary parts of eigenvalue numerators, dimension (N)
 * @param[out] beta Eigenvalue denominators, dimension (N)
 * @param[out] dwork Workspace, dimension (LDWORK)
 * @param[in] ldwork Workspace size (LDWORK >= max(1,N))
 * @param[out] info 0=success, <0=-i invalid arg, >0=failed to converge
 */
void mb03xp(const char *job, const char *compq, const char *compz,
            i32 n, i32 ilo, i32 ihi, f64 *a, i32 lda, f64 *b, i32 ldb,
            f64 *q, i32 ldq, f64 *z, i32 ldz,
            f64 *alphar, f64 *alphai, f64 *beta,
            f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Panel reduction for blocked Hamiltonian matrix.
 *
 * Reduces 2*NB columns and rows of a (K+2N)-by-(K+2N) Hamiltonian matrix H
 * using orthogonal symplectic transformations. H has structure:
 *   H = [op(A)   G  ]
 *       [  Q   op(B)]
 *
 * This is an auxiliary routine called by MB04TB.
 *
 * @param[in] ltra Form of op(A): false=A, true=A^T
 * @param[in] ltrb Form of op(B): false=B, true=B^T
 * @param[in] n Order of matrix Q (n >= 0)
 * @param[in] k Offset of reduction (k >= 0)
 * @param[in] nb Number of columns/rows to reduce (n > nb >= 0)
 * @param[in,out] a Matrix A, dimension (lda, n or k+n depending on ltra)
 * @param[in] lda Leading dimension of A
 * @param[in,out] b Matrix B, dimension (ldb, k+n or n depending on ltrb)
 * @param[in] ldb Leading dimension of B
 * @param[in,out] g Matrix G, dimension (ldg, k+n)
 * @param[in] ldg Leading dimension of G
 * @param[in,out] q Matrix Q, dimension (ldq, n)
 * @param[in] ldq Leading dimension of Q
 * @param[out] xa Update matrix XA, dimension (ldxa, 2*nb)
 * @param[in] ldxa Leading dimension of XA
 * @param[out] xb Update matrix XB, dimension (ldxb, 2*nb)
 * @param[in] ldxb Leading dimension of XB
 * @param[out] xg Update matrix XG, dimension (ldxg, 2*nb)
 * @param[in] ldxg Leading dimension of XG
 * @param[out] xq Update matrix XQ, dimension (ldxq, 2*nb)
 * @param[in] ldxq Leading dimension of XQ
 * @param[out] ya Update matrix YA, dimension (ldya, 2*nb)
 * @param[in] ldya Leading dimension of YA
 * @param[out] yb Update matrix YB, dimension (ldyb, 2*nb)
 * @param[in] ldyb Leading dimension of YB
 * @param[out] yg Update matrix YG, dimension (ldyg, 2*nb)
 * @param[in] ldyg Leading dimension of YG
 * @param[out] yq Update matrix YQ, dimension (ldyq, 2*nb)
 * @param[in] ldyq Leading dimension of YQ
 * @param[out] csl Cosines/sines of left Givens rotations, dimension (2*nb)
 * @param[out] csr Cosines/sines of right Givens rotations, dimension (2*nb)
 * @param[out] taul Reflector factors from left, dimension (nb)
 * @param[out] taur Reflector factors from right, dimension (nb)
 * @param[out] dwork Workspace, dimension (5*nb)
 */
void mb03xu(bool ltra, bool ltrb, i32 n, i32 k, i32 nb,
            f64 *a, i32 lda, f64 *b, i32 ldb, f64 *g, i32 ldg,
            f64 *q, i32 ldq, f64 *xa, i32 ldxa, f64 *xb, i32 ldxb,
            f64 *xg, i32 ldxg, f64 *xq, i32 ldxq, f64 *ya, i32 ldya,
            f64 *yb, i32 ldyb, f64 *yg, i32 ldyg, f64 *yq, i32 ldyq,
            f64 *csl, f64 *csr, f64 *taul, f64 *taur, f64 *dwork);

/**
 * @brief Annihilate subdiagonal entries of Hessenberg matrix.
 *
 * Annihilates one or two entries on the subdiagonal of the Hessenberg
 * matrix A for dealing with zero elements on the diagonal of the
 * triangular matrix B. Auxiliary routine for MB03XP and MB03YD.
 *
 * @param[in] wantt Compute full Schur form (true) or eigenvalues only (false)
 * @param[in] wantq Accumulate matrix Q (true) or not (false)
 * @param[in] wantz Accumulate matrix Z (true) or not (false)
 * @param[in] n Order of matrices A and B (n >= 0)
 * @param[in] ilo Lower bound of active block (1-based)
 * @param[in] ihi Upper bound of active block (1-based)
 * @param[in] iloq Lower row of Q/Z to transform (1-based)
 * @param[in] ihiq Upper row of Q/Z to transform (1-based)
 * @param[in] pos Position of zero diagonal in B (1-based)
 * @param[in,out] a Hessenberg matrix, dimension (lda,n)
 * @param[in] lda Leading dimension of a (lda >= max(1,n))
 * @param[in,out] b Upper triangular matrix with B(pos,pos)=0
 * @param[in] ldb Leading dimension of b (ldb >= max(1,n))
 * @param[in,out] q Transformation matrix Q, dimension (ldq,n)
 * @param[in] ldq Leading dimension of q (ldq >= 1, or >= n if wantq)
 * @param[in,out] z Transformation matrix Z, dimension (ldz,n)
 * @param[in] ldz Leading dimension of z (ldz >= 1, or >= n if wantz)
 * @param[out] info 0=success, <0=-i means i-th argument invalid
 */
void mb03ya(bool wantt, bool wantq, bool wantz, i32 n, i32 ilo, i32 ihi,
            i32 iloq, i32 ihiq, i32 pos, f64 *a, i32 lda, f64 *b, i32 ldb,
            f64 *q, i32 ldq, f64 *z, i32 ldz, i32 *info);

/**
 * @brief Periodic QR iteration for eigenvalues of periodic Hessenberg matrix.
 *
 * Computes eigenvalues of a product of two matrices A*inv(B), where A is
 * upper Hessenberg and B is upper triangular, using periodic QR iteration.
 * Optionally computes the full periodic Schur form and transformation matrices.
 *
 * @param[in] wantt True to compute full Schur form, false for eigenvalues only
 * @param[in] wantq True to update matrix Q
 * @param[in] wantz True to update matrix Z
 * @param[in] n Order of matrices A and B (N >= 0)
 * @param[in] ilo Lower bound of active block (1-based)
 * @param[in] ihi Upper bound of active block (1-based)
 * @param[in] iloq Lower row bound for Q/Z updates (1-based)
 * @param[in] ihiq Upper row bound for Q/Z updates (1-based)
 * @param[in,out] a Upper Hessenberg matrix, dimension (LDA,N)
 * @param[in] lda Leading dimension of A (LDA >= max(1,N))
 * @param[in,out] b Upper triangular matrix, dimension (LDB,N)
 * @param[in] ldb Leading dimension of B (LDB >= max(1,N))
 * @param[in,out] q Orthogonal transformation matrix, dimension (LDQ,N)
 * @param[in] ldq Leading dimension of Q (LDQ >= 1, or N if WANTQ)
 * @param[in,out] z Orthogonal transformation matrix, dimension (LDZ,N)
 * @param[in] ldz Leading dimension of Z (LDZ >= 1, or N if WANTZ)
 * @param[out] alphar Real parts of eigenvalue numerators, dimension (N)
 * @param[out] alphai Imaginary parts of eigenvalue numerators, dimension (N)
 * @param[out] beta Eigenvalue denominators, dimension (N)
 * @param[out] dwork Workspace, dimension (LDWORK)
 * @param[in] ldwork Workspace size (LDWORK >= max(1,N))
 * @param[out] info 0=success, <0=-i invalid arg, >0=i failed to converge
 */
void mb03yd(bool wantt, bool wantq, bool wantz, i32 n, i32 ilo, i32 ihi,
            i32 iloq, i32 ihiq, f64 *a, i32 lda, f64 *b, i32 ldb,
            f64 *q, i32 ldq, f64 *z, i32 ldz,
            f64 *alphar, f64 *alphai, f64 *beta,
            f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Periodic Schur factorization of 2x2 matrix pair.
 *
 * Computes the periodic Schur factorization of a real 2x2 matrix pair (A,B)
 * where B is upper triangular. Returns rotation matrices (csl,snl) and
 * (csr,snr) that transform the pair to periodic Schur form.
 *
 * If eigenvalues are real, A becomes upper triangular with eigenvalues
 * on the diagonal. If eigenvalues are complex conjugate, B becomes diagonal.
 *
 * @param[in,out] a Matrix A, dimension (lda,2). On exit, in periodic Schur form.
 * @param[in] lda Leading dimension of a (lda >= 2)
 * @param[in,out] b Upper triangular matrix B, dimension (ldb,2). On exit, transformed.
 * @param[in] ldb Leading dimension of b (ldb >= 2)
 * @param[out] alphar Real parts of eigenvalues, dimension (2)
 * @param[out] alphai Imaginary parts of eigenvalues, dimension (2)
 * @param[out] beta Scaling factors for eigenvalues, dimension (2)
 * @param[out] csl Cosine of left rotation
 * @param[out] snl Sine of left rotation
 * @param[out] csr Cosine of right rotation
 * @param[out] snr Sine of right rotation
 */
void mb03yt(f64 *a, i32 lda, f64 *b, i32 ldb,
            f64 *alphar, f64 *alphai, f64 *beta,
            f64 *csl, f64 *snl, f64 *csr, f64 *snr);

/**
 * @brief MB3OYZ - Complex rank-revealing QR factorization with column pivoting
 *
 * Computes a rank-revealing QR factorization of a complex general M-by-N matrix A,
 * which may be rank-deficient, and estimates its effective rank using incremental
 * condition estimation.
 *
 * The routine uses a truncated QR factorization with column pivoting:
 *     A * P = Q * R, where R = [ R11 R12 ]
 *                              [  0  R22 ]
 * with R11 defined as the largest leading upper triangular submatrix whose
 * estimated condition number is less than 1/RCOND.
 *
 * @param[in] m Number of rows of matrix A (m >= 0)
 * @param[in] n Number of columns of matrix A (n >= 0)
 * @param[in,out] a On entry, the M-by-N matrix A.
 *                  On exit, contains the QR factorization with column pivoting.
 * @param[in] lda Leading dimension of array a (lda >= max(1,m))
 * @param[in] rcond Used to determine effective rank (0 <= rcond <= 1)
 * @param[in] svlmax Estimate of largest singular value of parent matrix, or 0
 * @param[out] rank The effective (estimated) rank of A
 * @param[out] sval Array of size 3 with singular value estimates:
 *                  sval[0]: largest SV of R(1:rank,1:rank)
 *                  sval[1]: smallest SV of R(1:rank,1:rank)
 *                  sval[2]: smallest SV of R(1:rank+1,1:rank+1)
 * @param[out] jpvt Column permutation (jpvt[i]=k means i-th col of A*P was k-th col of A)
 * @param[out] tau Scalar factors of elementary reflectors (size min(m,n))
 * @param[out] dwork Real workspace (size 2*n)
 * @param[out] zwork Complex workspace (size 3*n-1)
 * @return info Error indicator (0=success, <0=-i means i-th argument invalid)
 */
i32 slicot_mb3oyz(i32 m, i32 n, c128* a, i32 lda, f64 rcond, f64 svlmax,
                  i32* rank, f64* sval, i32* jpvt, c128* tau, f64* dwork,
                  c128* zwork);

/**
 * @brief MB3PYZ - Complex rank-revealing RQ factorization with row pivoting
 *
 * Computes a rank-revealing RQ factorization of a complex general M-by-N matrix A,
 * which may be rank-deficient, and estimates its effective rank using incremental
 * condition estimation.
 *
 * The routine uses a truncated RQ factorization with row pivoting:
 *     P * A = R * Q, where R = [ R11 R12 ]
 *                              [  0  R22 ]
 * with R22 defined as the largest trailing upper triangular submatrix whose
 * estimated condition number is less than 1/RCOND.
 *
 * @param[in] m Number of rows of matrix A (m >= 0)
 * @param[in] n Number of columns of matrix A (n >= 0)
 * @param[in,out] a On entry, the M-by-N matrix A.
 *                  On exit, contains the RQ factorization with row pivoting.
 * @param[in] lda Leading dimension of array a (lda >= max(1,m))
 * @param[in] rcond Used to determine effective rank (0 <= rcond <= 1)
 * @param[in] svlmax Estimate of largest singular value of parent matrix, or 0
 * @param[out] rank The effective (estimated) rank of A
 * @param[out] sval Array of size 3 with singular value estimates:
 *                  sval[0]: largest SV of R(m-rank+1:m,n-rank+1:n)
 *                  sval[1]: smallest SV of R(m-rank+1:m,n-rank+1:n)
 *                  sval[2]: smallest SV of R(m-rank:m,n-rank:n)
 * @param[out] jpvt Row permutation (jpvt[i]=k means i-th row of P*A was k-th row of A)
 * @param[out] tau Scalar factors of elementary reflectors (size min(m,n))
 * @param[out] dwork Real workspace (size 2*m)
 * @param[out] zwork Complex workspace (size 3*m-1)
 * @return info Error indicator (0=success, <0=-i means i-th argument invalid)
 */
i32 slicot_mb3pyz(i32 m, i32 n, c128* a, i32 lda, f64 rcond, f64 svlmax,
                  i32* rank, f64* sval, i32* jpvt, c128* tau, f64* dwork,
                  c128* zwork);

/**
 * @brief Compute maps for Hessenberg index and signature array.
 *
 * Computes suitable maps (AMAP, QMAP) for periodic QZ algorithms based on
 * Hessenberg index H and signature array S. Auxiliary routine for periodic
 * QZ algorithms.
 *
 * @param[in] k Number of factors (K >= 1)
 * @param[in] h Index corresponding to A_1 (1-based)
 * @param[in] s Signature array, dimension (K). Each entry must be 1 or -1.
 * @param[out] smult Signature multiplier. Entries of S are virtually
 *                   multiplied by SMULT.
 * @param[out] amap Map for accessing factors, dimension (K). If AMAP[I-1] = J,
 *                  then factor A_I is stored at position J.
 * @param[out] qmap Map for accessing orthogonal transformations, dimension (K).
 *                  If QMAP[I-1] = J, then Q_I is stored at position J.
 */
void mb03ba(i32 k, i32 h, const i32 *s, i32 *smult, i32 *amap, i32 *qmap);

/**
 * @brief Compute Givens rotations for Wilkinson shift polynomial.
 *
 * Computes two Givens rotations (C1,S1) and (C2,S2) such that the orthogonal
 * matrix Z makes the first column of the real Wilkinson shift polynomial of
 * the product of matrices in periodic upper Hessenberg form parallel to the
 * first unit vector. Uses explicit shift values from eigenvalues.
 *
 * @param[in] shft Shift type: 'C'=complex conjugate, 'D'=real identical,
 *                 'R'=two real, 'S'=single real
 * @param[in] k Number of factors (K >= 1)
 * @param[in] n Order of factors (N >= 2 for single shift, N >= 3 for double)
 * @param[in] amap Map for accessing factors, dimension (K). AMAP[0] points
 *                 to Hessenberg matrix.
 * @param[in] s Signature array, dimension (K). Each entry 1 or -1.
 * @param[in] sinv Signature multiplier
 * @param[in] a 3D array (LDA1,LDA2,K) containing periodic Hessenberg product
 * @param[in] lda1 First leading dimension of A (>= N)
 * @param[in] lda2 Second leading dimension of A (>= N)
 * @param[in] w1 Real part of first eigenvalue (not used if SHFT='S')
 * @param[in] w2 Second eigenvalue (real) or imaginary part (complex pair)
 * @param[out] c1 Cosine of first Givens rotation
 * @param[out] s1 Sine of first Givens rotation
 * @param[out] c2 Cosine of second Givens rotation (1 if SHFT='S')
 * @param[out] s2 Sine of second Givens rotation (0 if SHFT='S')
 */
void mb03ab(const char *shft, i32 k, i32 n, const i32 *amap, const i32 *s,
            i32 sinv, const f64 *a, i32 lda1, i32 lda2, f64 w1, f64 w2,
            f64 *c1, f64 *s1, f64 *c2, f64 *s2);

/**
 * @brief Compute Givens rotations for Wilkinson shift polynomial (implicit).
 *
 * Similar to MB03AB but uses implicit shift computation. The Hessenberg matrix
 * is the last factor (AMAP[K-1]).
 *
 * @param[in] shft Shift type: 'D'=double shift, 'S'=single shift
 * @param[in] k Number of factors (K >= 1)
 * @param[in] n Order of factors (N >= 2 for single, N >= 3 for double)
 * @param[in] amap Map for factors. AMAP[K-1] points to Hessenberg matrix.
 * @param[in] s Signature array, dimension (K). S[K-1] not used (assumed 1).
 * @param[in] sinv Signature multiplier
 * @param[in] a 3D array (LDA1,LDA2,K) containing periodic Hessenberg product
 * @param[in] lda1 First leading dimension of A (>= N)
 * @param[in] lda2 Second leading dimension of A (>= N)
 * @param[out] c1 Cosine of first Givens rotation
 * @param[out] s1 Sine of first Givens rotation
 * @param[out] c2 Cosine of second rotation (1 if SHFT='S')
 * @param[out] s2 Sine of second rotation (0 if SHFT='S')
 */
void mb03af(const char *shft, i32 k, i32 n, const i32 *amap, const i32 *s,
            i32 sinv, const f64 *a, i32 lda1, i32 lda2, f64 *c1, f64 *s1,
            f64 *c2, f64 *s2);

/**
 * @brief Compute Givens rotations for real Wilkinson shift polynomial (partial evaluation).
 *
 * Computes two Givens rotations (C1,S1) and (C2,S2) such that the orthogonal
 * matrix Q makes the first column of the real Wilkinson double/single shift
 * polynomial of the periodic Hessenberg product parallel to the first unit
 * vector. The Hessenberg matrix is the last factor (AMAP[K-1]).
 *
 * Uses partial evaluation of the matrix product (trailing 2x2 and first two
 * columns). Called when convergence difficulties are encountered.
 *
 * For double shift with two real eigenvalues, both shifts equal the eigenvalue
 * with minimum modulus. For single shift, uses the (N,N) element of the product.
 *
 * @param[in] shft 'D'=double shift (assumes N>2), 'S'=single real shift
 * @param[in] k Number of factors (K >= 1)
 * @param[in] n Order of factors (N >= 2)
 * @param[in] amap Index map array, dimension (K). AMAP[K-1] points to Hessenberg matrix.
 * @param[in] s Signature array, dimension (K). Each entry +1 or -1.
 * @param[in] sinv Signature multiplier. Entries of S virtually multiplied by SINV.
 * @param[in] a 3D array (LDA1,LDA2,K) with periodic upper Hessenberg product
 * @param[in] lda1 First leading dimension (>= N)
 * @param[in] lda2 Second leading dimension (>= N)
 * @param[out] c1 Cosine of first Givens rotation
 * @param[out] s1 Sine of first Givens rotation
 * @param[out] c2 Cosine of second rotation (1 if SHFT='S' or N=2)
 * @param[out] s2 Sine of second rotation (0 if SHFT='S' or N=2)
 */
void mb03ah(const char *shft, i32 k, i32 n, const i32 *amap, const i32 *s,
            i32 sinv, const f64 *a, i32 lda1, i32 lda2, f64 *c1, f64 *s1,
            f64 *c2, f64 *s2);

/**
 * @brief Compute Givens rotations for Wilkinson shift polynomial (full evaluation).
 *
 * Evaluates the full matrix product and computes eigenvalues using DLAHQR to
 * find Givens rotations that make the first column of the Wilkinson shift
 * polynomial parallel to the first unit vector. The Hessenberg matrix is the
 * first factor (AMAP[0]).
 *
 * More robust but slower than implicit methods. Used when convergence
 * difficulties are encountered for small order matrices (N, K <= 6).
 *
 * @param[in] shft 'D'=double shift (N>2), 'S'=single real shift
 * @param[in] k Number of factors (K >= 1)
 * @param[in] n Order of factors (N >= 2)
 * @param[in] amap Index map array, dimension (K). AMAP[0] points to Hessenberg matrix.
 * @param[in] s Signature array, dimension (K). Each entry +1 or -1.
 * @param[in] sinv Signature multiplier
 * @param[in] a 3D array (LDA1,LDA2,K) with periodic upper Hessenberg product
 * @param[in] lda1 First leading dimension (>= N)
 * @param[in] lda2 Second leading dimension (>= N)
 * @param[out] c1 Cosine of first Givens rotation
 * @param[out] s1 Sine of first Givens rotation
 * @param[out] c2 Cosine of second rotation (1 if SHFT='S')
 * @param[out] s2 Sine of second rotation (0 if SHFT='S')
 * @param[out] iwork Integer workspace, dimension (2*N)
 * @param[out] dwork Double workspace, dimension (2*N*N). On exit, DWORK[N*N:N*N+N]
 *             and DWORK[N*N+N:N*N+2*N] contain eigenvalue real/imaginary parts.
 */
void mb03ag(const char *shft, i32 k, i32 n, const i32 *amap, const i32 *s,
            i32 sinv, const f64 *a, i32 lda1, i32 lda2, f64 *c1, f64 *s1,
            f64 *c2, f64 *s2, i32 *iwork, f64 *dwork);

/**
 * @brief Compute eigenvalues of 2x2 matrix product via complex periodic QZ.
 *
 * Computes eigenvalues of a 2x2 product of K matrices using a complex single
 * shifted periodic QZ algorithm. Eigenvalue i is:
 *   (ALPHAR[i] + ALPHAI[i]*sqrt(-1)) * BASE^SCAL[i]  if BETA[i] != 0
 *   infinite if BETA[i] = 0
 *
 * @param[in] base Machine base (DLAMCH('B'))
 * @param[in] lgbas Logarithm of BASE
 * @param[in] ulp Machine precision (DLAMCH('E'))
 * @param[in] k Number of factors (K >= 1)
 * @param[in] amap Map for accessing factors, dimension (K)
 * @param[in] s Signature array, dimension (K)
 * @param[in] sinv Signature multiplier
 * @param[in,out] a 3D array (LDA1,LDA2,K) with 2x2 Hessenberg-triangular product
 * @param[in] lda1 First leading dimension (>= 2)
 * @param[in] lda2 Second leading dimension (>= 2)
 * @param[out] alphar Real parts of scaled eigenvalues, dimension (2)
 * @param[out] alphai Imaginary parts (ALPHAI[0] >= 0), dimension (2)
 * @param[out] beta 1.0 if finite, 0.0 if infinite, dimension (2)
 * @param[out] scal Scaling exponents, dimension (2)
 * @param[out] dwork Workspace, dimension (8*K)
 * @param[out] info 0=success, 1=QZ didn't converge, 2=accuracy may be poor
 */
void mb03bb(f64 base, f64 lgbas, f64 ulp, i32 k, const i32 *amap,
            const i32 *s, i32 sinv, f64 *a, i32 lda1, i32 lda2,
            f64 *alphar, f64 *alphai, f64 *beta, i32 *scal,
            f64 *dwork, i32 *info);

/**
 * @brief Product SVD of K-1 triangular factors from 2x2 Hessenberg product.
 *
 * Computes Givens rotations so that the product of transformed 2x2 triangular
 * matrices is diagonal. This is used for deflation in the periodic QZ algorithm.
 *
 * @param[in] k Number of factors (K >= 1)
 * @param[in] amap Map for accessing factors, dimension (K)
 * @param[in] s Signature array, dimension (K)
 * @param[in] sinv Signature multiplier
 * @param[in,out] a 3D array (LDA1,LDA2,K). On exit, modified triangular factors.
 * @param[in] lda1 First leading dimension (>= 2)
 * @param[in] lda2 Second leading dimension (>= 2)
 * @param[in] macpar Machine parameters array, dimension (5):
 *                   [0]=overflow, [1]=underflow, [2]=safmin, [3]=eps, [4]=base
 * @param[out] cv Cosines of Givens rotations, dimension (K)
 * @param[out] sv Sines of Givens rotations, dimension (K)
 * @param[out] dwork Workspace, dimension (3*(K-1))
 */
void mb03bc(i32 k, const i32 *amap, const i32 *s, i32 sinv, f64 *a,
            i32 lda1, i32 lda2, const f64 *macpar, f64 *cv, f64 *sv,
            f64 *dwork);

/**
 * @brief Apply real single-shifted periodic QZ iteration to 2x2 product.
 *
 * Applies up to 20 iterations of a real single shifted periodic QZ algorithm
 * to deflate the 2x2 matrix product. The Hessenberg matrix is the last factor.
 *
 * @param[in] k Number of factors (K >= 1)
 * @param[in] amap Map for accessing factors, dimension (K)
 * @param[in] s Signature array, dimension (K)
 * @param[in] sinv Signature multiplier
 * @param[in,out] a 3D array (LDA1,LDA2,K). On exit, deflated product.
 * @param[in] lda1 First leading dimension (>= 2)
 * @param[in] lda2 Second leading dimension (>= 2)
 * @param[in] ulp Machine precision (DLAMCH('E'))
 */
void mb03bf(i32 k, const i32 *amap, const i32 *s, i32 sinv, f64 *a,
            i32 lda1, i32 lda2, f64 ulp);

/**
 * @brief Compute eigenvalues of periodic Hessenberg matrix product.
 *
 * Computes the eigenvalues of a product of K matrices:
 *   A = A_1^S(1) * A_2^S(2) * ... * A_K^S(K)
 * where S(i) = +1 or -1 (signature), and A^(-1) means inverse.
 *
 * The product is in periodic Hessenberg form: A_H is upper Hessenberg,
 * all other A_i are upper triangular.
 *
 * Optionally computes the periodic Schur form and transformation matrices.
 *
 * @param[in] job 'E'=eigenvalues only, 'S'=Schur form
 * @param[in] defl Deflation strategy: 'C'=classical, 'A'=aggressive
 * @param[in] compq 'N'=no Q, 'I'=initialize to I, 'V'=accumulate, 'P'=partial
 * @param[in] qind Integer array for partial Q (dimension K if COMPQ='P')
 * @param[in] k Number of factors (K >= 1)
 * @param[in] n Order of matrices (N >= 0)
 * @param[in] h Index of Hessenberg factor (1 <= H <= K)
 * @param[in] ilo,ihi Active submatrix bounds (1 <= ILO <= IHI <= N)
 * @param[in] s Signature array, dimension (K)
 * @param[in,out] a 3D array (LDA1,LDA2,K). On exit, Schur form if JOB='S'.
 * @param[in] lda1 First leading dimension (>= max(1,N))
 * @param[in] lda2 Second leading dimension (>= max(1,N))
 * @param[in,out] q 3D array for transformations. Updated if COMPQ!='N'.
 * @param[in] ldq1 First leading dimension of Q
 * @param[in] ldq2 Second leading dimension of Q
 * @param[out] alphar Real parts of eigenvalue numerators, dimension (N)
 * @param[out] alphai Imaginary parts, dimension (N)
 * @param[out] beta Eigenvalue denominators, dimension (N)
 * @param[out] scal Scaling exponents, dimension (N)
 * @param[out] iwork Integer workspace, dimension (LIWORK)
 * @param[in] liwork Integer workspace size (>= 2*K + N)
 * @param[out] dwork Real workspace, dimension (LDWORK)
 * @param[in] ldwork Real workspace size (depends on K, N, JOB)
 * @param[out] info 0=success, <0=-i invalid arg, >0=failed at eigenvalue info
 */
void mb03bd(const char *job, const char *defl, const char *compq,
            const i32 *qind, i32 k, i32 n, i32 h, i32 ilo, i32 ihi,
            const i32 *s, f64 *a, i32 lda1, i32 lda2,
            f64 *q, i32 ldq1, i32 ldq2,
            f64 *alphar, f64 *alphai, f64 *beta, i32 *scal,
            i32 *iwork, i32 liwork, f64 *dwork, i32 ldwork, i32 *iwarn, i32 *info);

/**
 * @brief Compute Givens rotations for Wilkinson shift polynomial.
 *
 * Computes two Givens rotations (C1,S1) and (C2,S2) such that the
 * orthogonal matrix
 *
 *           [ Q  0 ]        [  C1  S1  0 ]   [ 1  0   0  ]
 *       Z = [      ],  Q := [ -S1  C1  0 ] * [ 0  C2  S2 ],
 *           [ 0  I ]        [  0   0   1 ]   [ 0 -S2  C2 ]
 *
 * makes the first column of the real Wilkinson double/single shift
 * polynomial of the product of matrices in periodic upper Hessenberg
 * form, stored in the array A, parallel to the first unit vector.
 *
 * @param[in] shft 'D'=double shift (N>2), 'S'=single real shift
 * @param[in] k Number of factors (K >= 1)
 * @param[in] n Order of factors. N>=2 for single, N>=3 for double shift
 * @param[in] amap Index map array, dimension (K). AMAP(I)=J means A_I at pos J
 * @param[in] s Signature array, dimension (K). Each entry is +1 or -1
 * @param[in] sinv Signature multiplier (+1 or -1)
 * @param[in] a 3D array (LDA1,LDA2,K) periodic upper Hessenberg form
 * @param[in] lda1 First leading dimension (>= N)
 * @param[in] lda2 Second leading dimension (>= N)
 * @param[out] c1 Cosine of first Givens rotation
 * @param[out] s1 Sine of first Givens rotation
 * @param[out] c2 Cosine of second Givens rotation (1 if SHFT='S')
 * @param[out] s2 Sine of second Givens rotation (0 if SHFT='S')
 */
void mb03ad(const char *shft, i32 k, i32 n, const i32 *amap, const i32 *s,
            i32 sinv, const f64 *a, i32 lda1, i32 lda2, f64 *c1, f64 *s1,
            f64 *c2, f64 *s2);

/**
 * @brief Compute Givens rotations for Wilkinson shift polynomial (variant).
 *
 * Computes two Givens rotations (C1,S1) and (C2,S2) such that the
 * orthogonal matrix
 *
 *           [ Q  0 ]        [  C1  S1  0 ]   [ 1  0   0  ]
 *       Z = [      ],  Q := [ -S1  C1  0 ] * [ 0  C2  S2 ],
 *           [ 0  I ]        [  0   0   1 ]   [ 0 -S2  C2 ]
 *
 * makes the first column of the real Wilkinson double/single shift
 * polynomial of the product of matrices in periodic upper Hessenberg
 * form, stored in the array A, parallel to the first unit vector.
 *
 * Unlike MB03AB, this routine computes shifts from the trailing 2x2
 * submatrix rather than using explicit shift values.
 *
 * @param[in] shft 'D'=double shift (assumes N>2), 'S'=single real shift
 * @param[in] k Number of factors (K >= 1)
 * @param[in] n Order of factors (N >= 2)
 * @param[in] amap Index map array, dimension (K). AMAP(I)=J means A_I at pos J
 * @param[in] s Signature array, dimension (K). Each entry is +1 or -1
 * @param[in] sinv Signature multiplier (+1 or -1)
 * @param[in] a 3D array (LDA1,LDA2,K) periodic upper Hessenberg form
 * @param[in] lda1 First leading dimension (>= N)
 * @param[in] lda2 Second leading dimension (>= N)
 * @param[out] c1 Cosine of first Givens rotation
 * @param[out] s1 Sine of first Givens rotation
 * @param[out] c2 Cosine of second rotation (1 if SHFT='S' or N==2)
 * @param[out] s2 Sine of second rotation (0 if SHFT='S' or N==2)
 */
void mb03ae(const char *shft, i32 k, i32 n, const i32 *amap, const i32 *s,
            i32 sinv, const f64 *a, i32 lda1, i32 lda2, f64 *c1, f64 *s1,
            f64 *c2, f64 *s2);

/**
 * @brief Compute Givens rotations for Wilkinson shift polynomial (full evaluation).
 *
 * Evaluates the full matrix product and computes eigenvalues using DLAHQR.
 * More robust but slower. Used when convergence difficulties are encountered.
 *
 * @param[in] shft 'D'=double shift, 'S'=single real shift
 * @param[in] k Number of factors
 * @param[in] n Order of factors
 * @param[in] amap Index map array
 * @param[in] s Signature array
 * @param[in] sinv Signature multiplier
 * @param[in] a 3D array (LDA1,LDA2,K)
 * @param[in] lda1 First leading dimension
 * @param[in] lda2 Second leading dimension
 * @param[out] c1 Cosine of first rotation
 * @param[out] s1 Sine of first rotation
 * @param[out] c2 Cosine of second rotation
 * @param[out] s2 Sine of second rotation
 * @param[out] dwork Workspace, dimension (N*(N+2))
 */
void mb03ai(const char *shft, i32 k, i32 n, const i32 *amap, const i32 *s,
            i32 sinv, const f64 *a, i32 lda1, i32 lda2, f64 *c1, f64 *s1,
            f64 *c2, f64 *s2, f64 *dwork);

/**
 * @brief Apply periodic QZ iterations to 2x2 matrix product.
 *
 * Applies at most 20 iterations of a real single shifted periodic QZ algorithm
 * to the 2-by-2 product of matrices stored in the array A.
 *
 * @param[in] k Number of factors (K >= 1)
 * @param[in] amap Index map array, dimension (K)
 * @param[in] s Signature array, dimension (K), each entry +1 or -1
 * @param[in] sinv Signature multiplier
 * @param[in,out] a 3D array (LDA1,LDA2,K) containing 2x2 factors
 * @param[in] lda1 First leading dimension (>= 2)
 * @param[in] lda2 Second leading dimension (>= 2)
 */
void mb03be(i32 k, const i32 *amap, const i32 *s, i32 sinv, f64 *a, i32 lda1,
            i32 lda2);

/**
 * @brief Compute eigenvalues of 2x2 trailing submatrix of matrix product.
 *
 * Computes the eigenvalues of the 2-by-2 trailing submatrix of the product
 * A(:,:,1)^S(1) * A(:,:,2)^S(2) * ... * A(:,:,K)^S(K), where A(:,:,AMAP(K))
 * is upper Hessenberg and A(:,:,AMAP(i)) for i < K are upper triangular.
 *
 * @param[in] k Number of factors (K >= 1)
 * @param[in] n Order of factors (N >= 2)
 * @param[in] amap Index map array, dimension (K)
 * @param[in] s Signature array, dimension (K), each entry +1 or -1
 * @param[in] sinv Signature multiplier
 * @param[in] a 3D array (LDA1,LDA2,K) containing factors
 * @param[in] lda1 First leading dimension (>= N)
 * @param[in] lda2 Second leading dimension (>= N)
 * @param[out] wr Real parts of eigenvalues, dimension (2)
 * @param[out] wi Imaginary parts of eigenvalues, dimension (2)
 */
void mb03bg(i32 k, i32 n, const i32 *amap, const i32 *s, i32 sinv,
            const f64 *a, i32 lda1, i32 lda2, f64 *wr, f64 *wi);

/**
 * @brief Complex periodic QZ algorithm for eigenvalues of generalized matrix products.
 *
 * Computes eigenvalues of the complex generalized matrix product:
 *   A(:,:,1)^S(1) * A(:,:,2)^S(2) * ... * A(:,:,K)^S(K),  S(1) = 1
 *
 * where A(:,:,1) is upper Hessenberg and A(:,:,i), i=2,...,K are upper triangular.
 * Can optionally reduce A to periodic Schur form.
 *
 * @param[in] job 'E' eigenvalues only, 'S' Schur form + eigenvalues
 * @param[in] compq 'N' no Q, 'V' update Q, 'I' initialize Q to identity
 * @param[in] k Number of factors (K >= 1)
 * @param[in] n Order of factors (N >= 0)
 * @param[in] ilo Lower bound of balanced part
 * @param[in] ihi Upper bound of balanced part
 * @param[in] s Signature array, dimension (K), S(1)=1, others +/-1
 * @param[in,out] a 3D array (LDA1,LDA2,K), Hessenberg-triangular on entry
 * @param[in] lda1 First leading dimension of A
 * @param[in] lda2 Second leading dimension of A
 * @param[in,out] q 3D array (LDQ1,LDQ2,K) for unitary transformations
 * @param[in] ldq1 First leading dimension of Q
 * @param[in] ldq2 Second leading dimension of Q
 * @param[out] alpha Scaled eigenvalues, dimension (N)
 * @param[out] beta Infinite eigenvalue indicators, dimension (N)
 * @param[out] scal Scaling exponents, dimension (N)
 * @param[out] dwork Real workspace, dimension (LDWORK)
 * @param[in] ldwork Length of DWORK (>= MAX(1,N))
 * @param[out] zwork Complex workspace, dimension (LZWORK)
 * @param[in] lzwork Length of ZWORK (>= MAX(1,N))
 * @param[out] info 0=success, <0 param error, >0 convergence failure
 */
void mb03bz(const char *job, const char *compq, i32 k, i32 n, i32 ilo, i32 ihi,
            const i32 *s, c128 *a, i32 lda1, i32 lda2, c128 *q, i32 ldq1,
            i32 ldq2, c128 *alpha, c128 *beta, i32 *scal, f64 *dwork,
            i32 ldwork, c128 *zwork, i32 lzwork, i32 *info);

/**
 * @brief Exchange eigenvalues in 2x2, 3x3, or 4x4 block triangular pencils.
 *
 * Computes orthogonal matrices Q1, Q2, Q3 for a real 2-by-2, 3-by-3, or 4-by-4
 * regular block upper triangular pencil aAB - bD such that the eigenvalues in
 * Spec(A11*B11, D11) and Spec(A22*B22, D22) are exchanged.
 *
 * For UPLO='U': Upper block triangular, eigenvalues are exchanged on exit.
 * For UPLO='L': Lower block triangular, eigenvalues are NOT exchanged.
 *
 * @param[in] uplo 'U' upper block triangular, 'L' lower block triangular
 * @param[in,out] n1 Size of upper left block (N1 <= 2). If UPLO='U' and INFO=0,
 *                   N1 and N2 are exchanged on exit.
 * @param[in,out] n2 Size of lower right block (N2 <= 2). If UPLO='U' and INFO=0,
 *                   N1 and N2 are exchanged on exit.
 * @param[in] prec Machine precision (from DLAMCH)
 * @param[in,out] a Matrix A of pencil, dimension (LDA, N1+N2)
 * @param[in] lda Leading dimension of A (>= N1+N2)
 * @param[in,out] b Matrix B of pencil, dimension (LDB, N1+N2)
 * @param[in] ldb Leading dimension of B (>= N1+N2)
 * @param[in,out] d Matrix D of pencil, dimension (LDD, N1+N2). On exit contains
 *                  transformed D in real Schur form.
 * @param[in] ldd Leading dimension of D (>= N1+N2)
 * @param[out] q1 First orthogonal transformation, dimension (LDQ1, N1+N2)
 * @param[in] ldq1 Leading dimension of Q1 (>= N1+N2)
 * @param[out] q2 Second orthogonal transformation, dimension (LDQ2, N1+N2)
 * @param[in] ldq2 Leading dimension of Q2 (>= N1+N2)
 * @param[out] q3 Third orthogonal transformation, dimension (LDQ3, N1+N2)
 * @param[in] ldq3 Leading dimension of Q3 (>= N1+N2)
 * @param[out] dwork Workspace, dimension (LDWORK). Not referenced if N1+N2=2.
 * @param[in] ldwork Workspace size. LDWORK >= 16*N1+10*N2+23 for UPLO='U',
 *                   LDWORK >= 10*N1+16*N2+23 for UPLO='L'. LDWORK=0 if N1+N2=2.
 * @param[out] info 0=success, 1=QZ failed in DGGEV, 2=DGGEV error,
 *                  3=QZ failed in DGGES, 4=DGGES error, 5=DTGSEN reorder failed
 */
void mb03cd(const char *uplo, i32 *n1, i32 *n2, f64 prec, f64 *a, i32 lda,
            f64 *b, i32 ldb, f64 *d, i32 ldd, f64 *q1, i32 ldq1, f64 *q2,
            i32 ldq2, f64 *q3, i32 ldq3, f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Exchange eigenvalues of complex 2x2 upper triangular pencil.
 *
 * Computes unitary matrices Q1, Q2, Q3 for a complex 2-by-2 regular pencil
 * aAB - bD, with A, B, D upper triangular, such that Q3' A Q2, Q2' B Q1,
 * Q3' D Q1 are still upper triangular, but the eigenvalues are in reversed
 * order. The matrices Q1, Q2, Q3 are represented by:
 *
 *      (  CO1  SI1  )       (  CO2  SI2  )       (  CO3  SI3  )
 * Q1 = (            ), Q2 = (            ), Q3 = (            ).
 *      ( -SI1' CO1  )       ( -SI2' CO2  )       ( -SI3' CO3  )
 *
 * The notation M' denotes the conjugate transpose of the matrix M.
 *
 * @param[in] a Complex 2-by-2 upper triangular matrix A of the pencil
 * @param[in] lda Leading dimension of A (lda >= 2)
 * @param[in] b Complex 2-by-2 upper triangular matrix B of the pencil
 * @param[in] ldb Leading dimension of B (ldb >= 2)
 * @param[in] d Complex 2-by-2 upper triangular matrix D of the pencil
 * @param[in] ldd Leading dimension of D (ldd >= 2)
 * @param[out] co1 Cosine (real) of unitary matrix Q1
 * @param[out] si1 Sine (complex) of unitary matrix Q1
 * @param[out] co2 Cosine (real) of unitary matrix Q2
 * @param[out] si2 Sine (complex) of unitary matrix Q2
 * @param[out] co3 Cosine (real) of unitary matrix Q3
 * @param[out] si3 Sine (complex) of unitary matrix Q3
 */
void mb03cz(const c128 *a, i32 lda, const c128 *b, i32 ldb, const c128 *d,
            i32 ldd, f64 *co1, c128 *si1, f64 *co2, c128 *si2, f64 *co3,
            c128 *si3);

/**
 * @brief Exchange eigenvalues of real 2-by-2, 3-by-3 or 4-by-4 block upper
 * triangular pencil.
 *
 * Computes orthogonal matrices Q1 and Q2 for a real 2-by-2, 3-by-3, or 4-by-4
 * regular block upper triangular pencil such that the pencil
 * a(Q2' A Q1) - b(Q2' B Q1) is still in block upper triangular form, but the
 * eigenvalues in Spec(A11, B11), Spec(A22, B22) are exchanged.
 *
 * Optionally, to upper triangularize the real regular pencil in block lower
 * triangular form while keeping the eigenvalues in the same diagonal position.
 *
 * @param[in] uplo 'U': Upper block triangular, eigenvalues exchanged;
 *                 'T': Upper block triangular, B triangular, eigenvalues exchanged;
 *                 'L': Lower block triangular, eigenvalues not exchanged.
 * @param[in,out] n1 Size of upper left block (N1 <= 2). Exchanged with N2 on
 *                   exit if UPLO='U'/'T' and INFO=0, or UPLO='L' and INFO<>0.
 * @param[in,out] n2 Size of lower right block (N2 <= 2). Exchanged with N1 on
 *                   exit if UPLO='U'/'T' and INFO=0, or UPLO='L' and INFO<>0.
 * @param[in] prec Machine precision (e.g., DLAMCH('E')*DLAMCH('B'))
 * @param[in,out] a Matrix A of pencil aA - bB, dimension (LDA, N1+N2).
 *                  On exit, contains transformed quasi-triangular matrix.
 * @param[in] lda Leading dimension of A (LDA >= N1+N2)
 * @param[in,out] b Matrix B of pencil aA - bB, dimension (LDB, N1+N2).
 *                  On exit, contains transformed upper triangular matrix.
 * @param[in] ldb Leading dimension of B (LDB >= N1+N2)
 * @param[out] q1 First orthogonal transformation matrix, dimension (LDQ1, N1+N2)
 * @param[in] ldq1 Leading dimension of Q1 (LDQ1 >= N1+N2)
 * @param[out] q2 Second orthogonal transformation matrix, dimension (LDQ2, N1+N2)
 * @param[in] ldq2 Leading dimension of Q2 (LDQ2 >= N1+N2)
 * @param[out] dwork Workspace, dimension (LDWORK). Not referenced if N1+N2=2.
 * @param[in] ldwork Workspace size. LDWORK >= 16*N1+10*N2+23 for UPLO='U',
 *                   LDWORK >= 7*N1+7*N2+16 for UPLO='T',
 *                   LDWORK >= 10*N1+16*N2+23 for UPLO='L'. LDWORK=0 if N1+N2=2.
 * @param[out] info 0=success, 3=QZ failed in DGGES/DHGEQZ,
 *                  4=other error in DHGEQZ, 5=DTGSEN reorder failed
 */
void mb03dd(const char *uplo, i32 *n1, i32 *n2, f64 prec, f64 *a, i32 lda,
            f64 *b, i32 ldb, f64 *q1, i32 ldq1, f64 *q2, i32 ldq2,
            f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Exchange eigenvalues of complex 2x2 upper triangular pencil.
 *
 * Computes unitary matrices Q1 and Q2 for a complex 2-by-2 regular pencil
 * aA - bB with A, B upper triangular, such that Q2' (aA - bB) Q1 is still
 * upper triangular but the eigenvalues are in reversed order.
 *
 * The matrices Q1 and Q2 are represented by:
 *      (  CO1  SI1  )       (  CO2  SI2  )
 * Q1 = (            ), Q2 = (            ).
 *      ( -SI1' CO1  )       ( -SI2' CO2  )
 *
 * The notation M' denotes the conjugate transpose of the matrix M.
 *
 * @param[in] a Complex 2-by-2 upper triangular matrix A of the pencil
 * @param[in] lda Leading dimension of A (lda >= 2)
 * @param[in] b Complex 2-by-2 upper triangular matrix B of the pencil
 * @param[in] ldb Leading dimension of B (ldb >= 2)
 * @param[out] co1 Cosine (real) of unitary matrix Q1
 * @param[out] si1 Sine (complex) of unitary matrix Q1
 * @param[out] co2 Cosine (real) of unitary matrix Q2
 * @param[out] si2 Sine (complex) of unitary matrix Q2
 */
void mb03dz(const c128 *a, i32 lda, const c128 *b, i32 ldb, f64 *co1, c128 *si1,
            f64 *co2, c128 *si2);

/**
 * @brief Compute orthogonal matrices for 2x2 or 4x4 skew-Hamiltonian/Hamiltonian pencil.
 *
 * Computes orthogonal matrices Q1, Q2, Q3 for a real 2-by-2 or 4-by-4 regular
 * pencil aAB - bD, such that Q3' A Q2 and Q2' B Q1 are upper triangular,
 * Q3' D Q1 is upper quasi-triangular, and eigenvalues with negative real parts
 * (if any) are allocated on top.
 *
 * The pencil has the form:
 *     ( A11  0  ) ( B11  0  )     (  0  D12 )
 * a * (         ) (         ) - b (         )
 *     (  0  A22 ) (  0  B22 )     ( D21  0  )
 *
 * where A11, A22, B11, B22, D12 are upper triangular.
 *
 * @param[in] n Order of the input pencil, N = 2 or N = 4
 * @param[in] prec Machine precision (relative machine precision * base)
 * @param[in] a N-by-N upper triangular matrix A of pencil aAB - bD (LDA,N)
 * @param[in] lda Leading dimension of array A (lda >= n)
 * @param[in] b N-by-N upper triangular matrix B of pencil aAB - bD (LDB,N)
 * @param[in] ldb Leading dimension of array B (ldb >= n)
 * @param[in,out] d N-by-N matrix D of pencil. On exit if N=4, contains
 *                  transformed D in real Schur form. Unchanged if N=2. (LDD,N)
 * @param[in] ldd Leading dimension of array D (ldd >= n)
 * @param[out] q1 N-by-N first orthogonal transformation matrix (LDQ1,N)
 * @param[in] ldq1 Leading dimension of array Q1 (ldq1 >= n)
 * @param[out] q2 N-by-N second orthogonal transformation matrix (LDQ2,N)
 * @param[in] ldq2 Leading dimension of array Q2 (ldq2 >= n)
 * @param[out] q3 N-by-N third orthogonal transformation matrix (LDQ3,N)
 * @param[in] ldq3 Leading dimension of array Q3 (ldq3 >= n)
 * @param[out] dwork Workspace array (LDWORK). Not referenced if N=2.
 * @param[in] ldwork Workspace size. >= 79 if N=4, >= 0 if N=2.
 * @param[out] info 0=success, 1=QZ iteration failed in DGGES, 2=other DGGES error
 */
void mb03ed(i32 n, f64 prec, const f64 *a, i32 lda, const f64 *b, i32 ldb,
            f64 *d, i32 ldd, f64 *q1, i32 ldq1, f64 *q2, i32 ldq2,
            f64 *q3, i32 ldq3, f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Compute unitary and unitary symplectic matrices for 2x2 skew-Hamiltonian/Hamiltonian pencil.
 *
 * Computes a unitary matrix Q and a unitary symplectic matrix U for a complex
 * regular 2-by-2 skew-Hamiltonian/Hamiltonian pencil aS - bH with S = J Z' J' Z,
 * where Z and H are upper triangular:
 *
 *         (  Z11  Z12  )         (  H11  H12  )
 *     Z = (            ) and H = (            ),
 *         (   0   Z22  )         (   0  -H11' )
 *
 * such that U' Z Q and (J Q J')' H Q are both upper triangular, but the
 * eigenvalues are in reversed order.
 *
 * The matrices Q and U are represented by:
 *         (  CO1  SI1  )         (  CO2  SI2  )
 *     Q = (            ) and U = (            ), respectively.
 *         ( -SI1' CO1  )         ( -SI2' CO2  )
 *
 * @param[in] z11 Upper left element of Z
 * @param[in] z12 Upper right element of Z
 * @param[in] z22 Lower right element of Z
 * @param[in] h11 Upper left element of H
 * @param[in] h12 Upper right element of H
 * @param[out] co1 Cosine element of Q (real)
 * @param[out] si1 Sine element of Q (complex)
 * @param[out] co2 Cosine element of U (real)
 * @param[out] si2 Sine element of U (complex)
 */
void mb03gz(c128 z11, c128 z12, c128 z22, c128 h11, c128 h12,
            f64 *co1, c128 *si1, f64 *co2, c128 *si2);

/**
 * @brief Exchange eigenvalues of 2x2 or 4x4 skew-Hamiltonian/Hamiltonian pencil (factored version).
 *
 * Computes orthogonal matrix Q and orthogonal symplectic matrix U for a real
 * regular 2-by-2 or 4-by-4 skew-Hamiltonian/Hamiltonian pencil a J B' J' B - b D
 * with:
 *
 *         ( B11  B12 )      (  D11  D12  )      (  0  I  )
 *     B = (          ), D = (            ), J = (        )
 *         (  0   B22 )      (   0  -D11' )      ( -I  0  )
 *
 * such that J Q' J' D Q and U' B Q keep block triangular form, but eigenvalues
 * are reordered.
 *
 * @param[in] n Order of the pencil, N = 2 or N = 4
 * @param[in] b N-by-N upper block triangular matrix B. The (2,1) block is not referenced.
 * @param[in] ldb Leading dimension of B (ldb >= N)
 * @param[in] d (N/2)-by-N array. First block row of Hamiltonian matrix D.
 *              Strict lower triangle of (1,2) block not referenced.
 * @param[in] ldd Leading dimension of D (ldd >= N/2)
 * @param[in] macpar Machine parameters array (2): DLAMCH('P'), DLAMCH('S'). Not used if N=2.
 * @param[out] q N-by-N orthogonal transformation matrix Q
 * @param[in] ldq Leading dimension of Q (ldq >= N)
 * @param[out] u N-by-N orthogonal symplectic transformation matrix U
 * @param[in] ldu Leading dimension of U (ldu >= N)
 * @param[out] dwork Workspace array. If N=4 then LDWORK >= 12; if N=2 not used.
 * @param[in] ldwork Workspace size (>= 12 if N=4, >= 0 if N=2)
 * @param[out] info 0=success, 1=B11 or B22 is numerically singular
 */
void mb03gd(i32 n, const f64 *b, i32 ldb, const f64 *d, i32 ldd,
            const f64 *macpar, f64 *q, i32 ldq, f64 *u, i32 ldu,
            f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Exchange eigenvalues of 2x2 or 4x4 skew-Hamiltonian/Hamiltonian pencil.
 *
 * Computes an orthogonal matrix Q for a real regular 2-by-2 or 4-by-4
 * skew-Hamiltonian/Hamiltonian pencil in structured Schur form:
 *
 *                 ( A11 A12  )     ( B11  B12  )
 *     aA - bB = a (          ) - b (           )
 *                 (  0  A11' )     (  0  -B11' )
 *
 * such that J Q' J' (aA - bB) Q is still in structured Schur form but the
 * eigenvalues are exchanged.
 *
 * @param[in] n Order of the pencil, N = 2 or N = 4
 * @param[in] a N/2-by-N array. If N=4, first block row of skew-Hamiltonian A.
 *              Only (1,1), (1,2), (1,4), (2,2) referenced. Not used if N=2.
 * @param[in] lda Leading dimension of A (lda >= N/2)
 * @param[in] b N/2-by-N array. First block row of Hamiltonian matrix B.
 *              Entry (2,3) not referenced.
 * @param[in] ldb Leading dimension of B (ldb >= N/2)
 * @param[in] macpar Machine parameters array (2): DLAMCH('P'), DLAMCH('S').
 *                   Not used if N=2.
 * @param[out] q N-by-N orthogonal transformation matrix Q
 * @param[in] ldq Leading dimension of Q (ldq >= N)
 * @param[out] dwork Workspace array (24). Not used if N=2.
 * @param[out] info 0=success, 1=B11 nearly singular (perturbed values used)
 */
void mb03hd(i32 n, const f64 *a, i32 lda, const f64 *b, i32 ldb,
            const f64 *macpar, f64 *q, i32 ldq, f64 *dwork, i32 *info);

/**
 * @brief Exchange eigenvalues of complex 2x2 skew-Hamiltonian/Hamiltonian pencil.
 *
 * Computes a unitary matrix Q for a complex regular 2-by-2
 * skew-Hamiltonian/Hamiltonian pencil aS - bH with
 *
 *     (  S11  S12  )        (  H11  H12  )
 * S = (            ),   H = (            ),
 *     (   0   S11' )        (   0  -H11' )
 *
 * such that J Q' J' (aS - bH) Q is upper triangular but the eigenvalues
 * are in reversed order. The matrix Q is represented by
 *
 *     (  CO  SI  )
 * Q = (          ).
 *     ( -SI' CO  )
 *
 * The notation M' denotes the conjugate transpose of the matrix M.
 *
 * @param[in] s11 Upper left element of skew-Hamiltonian matrix S
 * @param[in] s12 Upper right element of skew-Hamiltonian matrix S
 * @param[in] h11 Upper left element of Hamiltonian matrix H
 * @param[in] h12 Upper right element of Hamiltonian matrix H
 * @param[out] co Upper left element of Q (real)
 * @param[out] si Upper right element of Q (complex)
 */
void mb03hz(c128 s11, c128 s12, c128 h11, c128 h12, f64 *co, c128 *si);

/**
 * @brief Reorder eigenvalues of real skew-Hamiltonian/Hamiltonian pencil.
 *
 * Moves eigenvalues with strictly negative real parts of an N-by-N real
 * skew-Hamiltonian/Hamiltonian pencil aS - bH in structured Schur form to the
 * leading principal subpencil, while keeping the triangular form.
 *
 * On entry:
 *     S = J Z' J' Z, J = [[0, I], [-I, 0]], Z = [[A, D], [0, C]], H = [[B, F], [0, -B']]
 *
 * where A is upper triangular, B is upper quasi-triangular, C is lower triangular.
 *
 * @param[in] compq 'N'=no Q, 'I'=init Q to identity, 'U'=update Q
 * @param[in] compu 'N'=no U, 'I'=init U to identity, 'U'=update U
 * @param[in] n Order of the pencil (n >= 0, even)
 * @param[in,out] a Upper triangular matrix A, dimension (lda, n/2)
 * @param[in] lda Leading dimension of A (lda >= max(1, n/2))
 * @param[in,out] c Lower triangular matrix C, dimension (ldc, n/2)
 * @param[in] ldc Leading dimension of C (ldc >= max(1, n/2))
 * @param[in,out] d Matrix D, dimension (ldd, n/2)
 * @param[in] ldd Leading dimension of D (ldd >= max(1, n/2))
 * @param[in,out] b Upper quasi-triangular matrix B, dimension (ldb, n/2)
 * @param[in] ldb Leading dimension of B (ldb >= max(1, n/2))
 * @param[in,out] f Symmetric matrix F (upper triangular part), dimension (ldf, n/2)
 * @param[in] ldf Leading dimension of F (ldf >= max(1, n/2))
 * @param[in,out] q Orthogonal matrix Q, dimension (ldq, n)
 * @param[in] ldq Leading dimension of Q (ldq >= 1 if compq='N', ldq >= n otherwise)
 * @param[in,out] u1 Upper left block of orthogonal symplectic U, dimension (ldu1, n/2)
 * @param[in] ldu1 Leading dimension of U1 (ldu1 >= 1 if compu='N', ldu1 >= n/2 otherwise)
 * @param[in,out] u2 Upper right block of orthogonal symplectic U, dimension (ldu2, n/2)
 * @param[in] ldu2 Leading dimension of U2 (ldu2 >= 1 if compu='N', ldu2 >= n/2 otherwise)
 * @param[out] neig Number of eigenvalues with strictly negative real part
 * @param[out] iwork Integer workspace, dimension (liwork)
 * @param[in] liwork Integer workspace size (liwork >= n+1)
 * @param[out] dwork Real workspace, dimension (ldwork)
 * @param[in] ldwork Real workspace size (>= max(2*n+48,171) if compq='N', >= max(4*n+48,171) otherwise)
 * @param[out] info 0=success, <0=-i means i-th argument invalid, 1=QZ failed, 2=MB03CD error, 3=MB03GD error
 */
void mb03id(const char *compq, const char *compu, i32 n, f64 *a, i32 lda,
            f64 *c, i32 ldc, f64 *d, i32 ldd, f64 *b, i32 ldb, f64 *f, i32 ldf,
            f64 *q, i32 ldq, f64 *u1, i32 ldu1, f64 *u2, i32 ldu2, i32 *neig,
            i32 *iwork, i32 liwork, f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Reduce 2x2 formal matrix product to periodic Hessenberg-triangular form.
 *
 * Reduces a 2-by-2 general, formal matrix product A of length K,
 *
 *    A_K^s(K) * A_K-1^s(K-1) * ... * A_1^s(1),
 *
 * to periodic Hessenberg-triangular form using K-periodic sequence of
 * elementary reflectors (Householder matrices). The matrices A_k are
 * stored in N-by-N-by-K array starting in the R-th row and column.
 *
 * Each elementary reflector H_k = I - tau_k * v_k * v_k' is constructed
 * such that all T_k are upper triangular except T_khess which is full.
 *
 * @param[in] k Number of matrices in the sequence (k >= 2)
 * @param[in] khess Index for which A_khess is Hessenberg (1 <= khess <= k)
 * @param[in] n Order of extended matrices (n = 3 or n = 4)
 * @param[in] r Starting row/column index for 2x2 submatrices (r = 1 or r = n-1)
 * @param[in] s Signature array of length k (each element +1 or -1)
 * @param[in,out] a Array of k N-by-N matrices stored with stride n*lda.
 *                  On entry: matrices Ae_k. On exit: transformed Te_k.
 * @param[in] lda Leading dimension of each matrix (lda >= n)
 * @param[out] v Array of length 2*k containing K 2-vectors v_k
 * @param[out] tau Array of length k containing tau_k values
 */
void mb03kc(i32 k, i32 khess, i32 n, i32 r, const i32 *s, f64 *a, i32 lda,
            f64 *v, f64 *tau);

/**
 * @brief Solve small periodic Sylvester-like equations.
 *
 * Solves periodic Sylvester-like equations (PSLE):
 *   op(A(i))*X(i)   + isgn*X(i+1)*op(B(i)) = -scale*C(i), S(i) =  1
 *   op(A(i))*X(i+1) + isgn*X(i)  *op(B(i)) = -scale*C(i), S(i) = -1
 *
 * for i = 1, ..., K, where op(A) means A or A**T, for the K-periodic
 * matrix sequence X(i) = X(i+K), where A, B and C are K-periodic
 * matrix sequences. The matrices A(i) are M-by-M and B(i) are N-by-N,
 * with 1 <= M, N <= 2.
 *
 * @param[in] trana  If true, op(A) = A**T; otherwise op(A) = A
 * @param[in] tranb  If true, op(B) = B**T; otherwise op(B) = B
 * @param[in] isgn   Sign variant: 1 or -1
 * @param[in] k      Period of sequences (k >= 2)
 * @param[in] m      Order of A matrices and rows of C,X (1 <= m <= 2)
 * @param[in] n      Order of B matrices and cols of C,X (1 <= n <= 2)
 * @param[in] prec   Relative machine precision (from DLAMCH)
 * @param[in] smin   Machine safe minimum divided by prec
 * @param[in] s      Array of K signatures, each +1 or -1
 * @param[in] a      Array of K M-by-M matrices (length M*M*K)
 * @param[in] b      Array of K N-by-N matrices (length N*N*K)
 * @param[in,out] c  On entry, K M-by-N matrices C(i). On exit, solution X(i).
 * @param[out] scale Scale factor (<= 1) to avoid overflow
 * @param[out] dwork Workspace array of length ldwork
 * @param[in] ldwork Workspace size. Required: (4*K-3)*(M*N)^2 + K*M*N.
 *                   If -1, workspace query.
 * @param[out] info  0=success, -21=ldwork too small, 1=scaled to avoid overflow
 */
void mb03ke(bool trana, bool tranb, i32 isgn, i32 k, i32 m, i32 n,
            f64 prec, f64 smin, const i32 *s, const f64 *a, const f64 *b,
            f64 *c, f64 *scale, f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Compute absolute minimum value in an array.
 *
 * Computes the minimum of |x[i]| for i = 0, incx, 2*incx, ..., (nx-1)*incx.
 * Returns 0.0 if nx <= 0.
 *
 * @param[in] nx    Number of elements to examine
 * @param[in] x     Array of dimension nx*incx
 * @param[in] incx  Increment between elements (incx >= 1)
 * @return Minimum absolute value, or 0.0 if nx <= 0
 */
f64 mb03my(i32 nx, const f64 *x, i32 incx);

/**
 * @brief Count singular values of bidiagonal matrix <= bound.
 *
 * Finds the number of singular values of the bidiagonal matrix J that are
 * less than or equal to a given bound THETA. Uses Sturm sequences applied
 * to an associated symmetric tridiagonal matrix.
 *
 * @param[in] n       Order of bidiagonal matrix J (n >= 0)
 * @param[in] theta   Upper bound for singular values
 * @param[in] q2      Array(n) of squared diagonal elements
 * @param[in] e2      Array(n-1) of squared superdiagonal elements
 * @param[in] pivmin  Minimum pivot value for Sturm sequence
 * @param[out] info   0=success, -1=invalid n
 * @return Number of singular values <= theta
 */
i32 mb03nd(i32 n, f64 theta, const f64 *q2, const f64 *e2, f64 pivmin, i32 *info);

/**
 * @brief Compute upper bound for L singular values of bidiagonal matrix.
 *
 * Computes an upper bound THETA using bisection such that the bidiagonal
 * matrix J has precisely L singular values <= THETA + TOL.
 *
 * @param[in] n       Order of bidiagonal matrix J (n >= 0)
 * @param[in,out] l   On entry: number of singular values <= bound.
 *                    On exit: may be increased if multiplicity > 1.
 * @param[in,out] theta  On entry: initial estimate (negative for default).
 *                       On exit: computed upper bound.
 * @param[in] q       Array(n) of diagonal elements
 * @param[in] e       Array(n-1) of superdiagonal elements
 * @param[in] q2      Array(n) of squared diagonal elements
 * @param[in] e2      Array(n-1) of squared superdiagonal elements
 * @param[in] pivmin  Minimum pivot value
 * @param[in] tol     Tolerance for singular value coincidence
 * @param[in] reltol  Relative tolerance for bisection convergence
 * @param[out] iwarn  0=ok, 1=L increased due to coinciding singular values
 * @param[out] info   0=success, <0=parameter error
 */
void mb03md(i32 n, i32 *l, f64 *theta, const f64 *q, const f64 *e,
            const f64 *q2, const f64 *e2, f64 pivmin, f64 tol, f64 reltol,
            i32 *iwarn, i32 *info);

/**
 * @brief Compute eigenvalues of upper quasi-triangular matrix pencil.
 *
 * Computes the generalized eigenvalues of an upper quasi-triangular matrix
 * pencil (S, T) where S is upper quasi-triangular and T is upper triangular.
 * The eigenvalues are returned as (ALPHAR + i*ALPHAI) / BETA.
 *
 * For real eigenvalues, ALPHAI(j) = 0. For complex conjugate pairs,
 * ALPHAI(j) > 0 for the first eigenvalue and ALPHAI(j+1) = -ALPHAI(j).
 *
 * @param[in] n Order of matrices S and T (n >= 0)
 * @param[in] s Upper quasi-triangular matrix S, dimension (lds, n)
 * @param[in] lds Leading dimension of S (lds >= max(1, n))
 * @param[in] t Upper triangular matrix T, dimension (ldt, n)
 * @param[in] ldt Leading dimension of T (ldt >= max(1, n))
 * @param[out] alphar Real parts of eigenvalue numerators, dimension (n)
 * @param[out] alphai Imaginary parts of eigenvalue numerators, dimension (n)
 * @param[out] beta Eigenvalue denominators, dimension (n)
 * @param[out] info 0=success, <0=-i means i-th argument invalid
 */
void mb03qv(i32 n, const f64 *s, i32 lds, const f64 *t, i32 ldt,
            f64 *alphar, f64 *alphai, f64 *beta, i32 *info);

/**
 * @brief Reduce 2-by-2 diagonal block pair of quasi-triangular pencil.
 *
 * Computes eigenvalues of a selected 2-by-2 diagonal block pair of an upper
 * quasi-triangular pencil, reduces the block pair to standard form, and
 * splits it if eigenvalues are real. Uses orthogonal transformations UT and VT.
 *
 * @param[in] n Order of matrices A, E, U, V (n >= 2)
 * @param[in] l Position of the 2-by-2 block (1 <= l < n)
 * @param[in,out] a Upper quasi-triangular matrix A, dimension (lda, n)
 * @param[in] lda Leading dimension of A (lda >= n)
 * @param[in,out] e Upper triangular matrix E, dimension (lde, n)
 * @param[in] lde Leading dimension of E (lde >= n)
 * @param[in,out] u Transformation matrix U, dimension (ldu, n)
 * @param[in] ldu Leading dimension of U (ldu >= n)
 * @param[in,out] v Transformation matrix V, dimension (ldv, n)
 * @param[in] ldv Leading dimension of V (ldv >= n)
 * @param[out] alphar Real parts of eigenvalue numerators, dimension (2)
 * @param[out] alphai Imaginary parts of eigenvalue numerators, dimension (2)
 * @param[out] beta Eigenvalue denominators, dimension (2)
 * @param[out] info 0=success, <0=-i means i-th argument invalid
 */
void mb03qw(i32 n, i32 l, f64 *a, i32 lda, f64 *e, i32 lde,
            f64 *u, i32 ldu, f64 *v, i32 ldv,
            f64 *alphar, f64 *alphai, f64 *beta, i32 *info);

/**
 * @brief Reorder eigenvalues of complex skew-Hamiltonian/Hamiltonian pencil.
 *
 * Moves eigenvalues with strictly negative real parts of an N-by-N complex
 * skew-Hamiltonian/Hamiltonian pencil aS - bH in structured Schur form to the
 * leading principal subpencil, while keeping the triangular form.
 *
 * On entry:
 *       (  A  D  )      (  B  F  )
 *   Z = (        ), H = (        ),
 *       (  0  C  )      (  0 -B' )
 *
 * where A and B are upper triangular and C is lower triangular.
 *
 * @param[in] compq 'N'=no Q, 'I'=init Q to identity, 'U'=update Q
 * @param[in] compu 'N'=no U, 'I'=init U to identity, 'U'=update U
 * @param[in] n Order of the pencil (n >= 0, even)
 * @param[in,out] a Upper triangular matrix A, dimension (lda, n/2)
 * @param[in] lda Leading dimension of A (lda >= max(1, n/2))
 * @param[in,out] c Lower triangular matrix C, dimension (ldc, n/2)
 * @param[in] ldc Leading dimension of C (ldc >= max(1, n/2))
 * @param[in,out] d Matrix D, dimension (ldd, n/2)
 * @param[in] ldd Leading dimension of D (ldd >= max(1, n/2))
 * @param[in,out] b Upper triangular matrix B, dimension (ldb, n/2)
 * @param[in] ldb Leading dimension of B (ldb >= max(1, n/2))
 * @param[in,out] f Hermitian matrix F (upper triangular part), dimension (ldf, n/2)
 * @param[in] ldf Leading dimension of F (ldf >= max(1, n/2))
 * @param[in,out] q Unitary matrix Q, dimension (ldq, n)
 * @param[in] ldq Leading dimension of Q (ldq >= 1 if compq='N', ldq >= n otherwise)
 * @param[in,out] u1 Upper left block of unitary symplectic U, dimension (ldu1, n/2)
 * @param[in] ldu1 Leading dimension of U1 (ldu1 >= 1 if compu='N', ldu1 >= n/2 otherwise)
 * @param[in,out] u2 Upper right block of unitary symplectic U, dimension (ldu2, n/2)
 * @param[in] ldu2 Leading dimension of U2 (ldu2 >= 1 if compu='N', ldu2 >= n/2 otherwise)
 * @param[out] neig Number of eigenvalues with strictly negative real part
 * @param[in] tol Tolerance for eigenvalue sign (tol <= 0 uses default)
 * @param[out] info 0=success, <0=-i means i-th argument invalid
 */
void mb03iz(const char *compq, const char *compu, i32 n, c128 *a, i32 lda,
            c128 *c, i32 ldc, c128 *d, i32 ldd, c128 *b, i32 ldb, c128 *f,
            i32 ldf, c128 *q, i32 ldq, c128 *u1, i32 ldu1, c128 *u2, i32 ldu2,
            i32 *neig, f64 tol, i32 *info);

/**
 * @brief Move eigenvalues with negative real parts to leading subpencil
 *
 * Moves eigenvalues with strictly negative real parts of an N-by-N real
 * skew-Hamiltonian/Hamiltonian pencil aS - bH in structured Schur form to
 * the leading principal subpencil, while keeping the triangular form.
 *
 * On entry:
 *       (  A  D  )      (  B  F  )
 *   S = (        ), H = (        ),
 *       (  0  A' )      (  0 -B' )
 *
 * where A is upper triangular and B is upper quasi-triangular.
 *
 * @param[in] compq 'N'=no Q, 'I'=init Q to identity, 'U'=update Q
 * @param[in] n Order of the pencil (n >= 0, even)
 * @param[in,out] a Upper triangular matrix A, dimension (lda, n/2)
 * @param[in] lda Leading dimension of A (lda >= max(1, n/2))
 * @param[in,out] d Skew-symmetric matrix D (upper triangular part), dimension (ldd, n/2)
 * @param[in] ldd Leading dimension of D (ldd >= max(1, n/2))
 * @param[in,out] b Upper quasi-triangular matrix B, dimension (ldb, n/2)
 * @param[in] ldb Leading dimension of B (ldb >= max(1, n/2))
 * @param[in,out] f Symmetric matrix F (upper triangular part), dimension (ldf, n/2)
 * @param[in] ldf Leading dimension of F (ldf >= max(1, n/2))
 * @param[in,out] q Orthogonal matrix Q, dimension (ldq, n)
 * @param[in] ldq Leading dimension of Q (ldq >= 1 if compq='N', ldq >= n otherwise)
 * @param[out] neig Number of eigenvalues with strictly negative real part
 * @param[out] iwork Integer workspace, dimension (liwork)
 * @param[in] liwork Dimension of iwork (liwork >= n+1)
 * @param[out] dwork Double workspace, dimension (ldwork)
 * @param[in] ldwork Dimension of dwork (compq='N': max(2*n+32,108), otherwise max(4*n+32,108))
 * @param[out] info 0=success, <0=-i means i-th argument invalid, 1=MB03DD error, 2=MB03HD error
 */
void mb03jd(const char* compq, i32 n, f64* a, i32 lda,
            f64* d, i32 ldd, f64* b, i32 ldb,
            f64* f, i32 ldf, f64* q, i32 ldq,
            i32* neig, i32* iwork, i32 liwork,
            f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Move eigenvalues with negative real parts to leading subpencil (panel variant)
 *
 * Panel-based blocked variant of MB03JD for better performance on large matrices.
 * Moves eigenvalues with strictly negative real parts of an N-by-N real
 * skew-Hamiltonian/Hamiltonian pencil aS - bH in structured Schur form to
 * the leading principal subpencil, while keeping the triangular form.
 *
 * @param[in] compq 'N'=no Q, 'I'=init Q to identity, 'U'=update Q
 * @param[in] n Order of the pencil (n >= 0, even)
 * @param[in,out] a Upper triangular matrix A, dimension (lda, n/2)
 * @param[in] lda Leading dimension of A (lda >= max(1, n/2))
 * @param[in,out] d Skew-symmetric matrix D (upper triangular part)
 * @param[in] ldd Leading dimension of D (ldd >= max(1, n/2))
 * @param[in,out] b Upper quasi-triangular matrix B, dimension (ldb, n/2)
 * @param[in] ldb Leading dimension of B (ldb >= max(1, n/2))
 * @param[in,out] f Symmetric matrix F (upper triangular part)
 * @param[in] ldf Leading dimension of F (ldf >= max(1, n/2))
 * @param[in,out] q Orthogonal matrix Q, dimension (ldq, n)
 * @param[in] ldq Leading dimension of Q (ldq >= 1 if compq='N', ldq >= n otherwise)
 * @param[out] neig Number of eigenvalues with strictly negative real part
 * @param[out] iwork Integer workspace, dimension (liwork)
 * @param[in] liwork Dimension of iwork (liwork >= 3*n-3)
 * @param[out] dwork Double workspace, dimension (ldwork)
 * @param[in] ldwork Dimension of dwork (compq='N': max(2*n+32,108)+5*n/2, otherwise max(4*n+32,108)+5*n/2)
 * @param[out] info 0=success, <0=-i means i-th argument invalid
 */
void mb03jp(const char* compq, i32 n, f64* a, i32 lda,
            f64* d, i32 ldd, f64* b, i32 ldb,
            f64* f, i32 ldf, f64* q, i32 ldq,
            i32* neig, i32* iwork, i32 liwork,
            f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Move eigenvalues with negative real parts to leading subpencil (complex)
 *
 * Moves eigenvalues with strictly negative real parts of an N-by-N complex
 * skew-Hamiltonian/Hamiltonian pencil aS - bH in structured Schur form to
 * the leading principal subpencil, while keeping the triangular form.
 *
 * On entry:
 *       (  A  D  )      (  B  F  )
 *   S = (        ), H = (        ),
 *       (  0  A' )      (  0 -B' )
 *
 * where A and B are upper triangular.
 *
 * @param[in] compq 'N'=no Q, 'I'=init Q to identity, 'U'=update Q
 * @param[in] n Order of the pencil (n >= 0, even)
 * @param[in,out] a Upper triangular matrix A, dimension (lda, n/2)
 * @param[in] lda Leading dimension of A (lda >= max(1, n/2))
 * @param[in,out] d Skew-Hermitian matrix D (upper triangular part)
 * @param[in] ldd Leading dimension of D (ldd >= max(1, n/2))
 * @param[in,out] b Upper triangular matrix B, dimension (ldb, n/2)
 * @param[in] ldb Leading dimension of B (ldb >= max(1, n/2))
 * @param[in,out] f Hermitian matrix F (upper triangular part)
 * @param[in] ldf Leading dimension of F (ldf >= max(1, n/2))
 * @param[in,out] q Unitary matrix Q, dimension (ldq, n)
 * @param[in] ldq Leading dimension of Q (ldq >= 1 if compq='N', ldq >= n otherwise)
 * @param[out] neig Number of eigenvalues with strictly negative real part
 * @param[in] tol Tolerance for eigenvalue sign (tol <= 0 uses default MIN(N,10)*EPS)
 * @param[out] info 0=success, <0=-i means i-th argument invalid
 */
void mb03jz(const char* compq, i32 n, c128* a, i32 lda,
            c128* d, i32 ldd, c128* b, i32 ldb,
            c128* f, i32 ldf, c128* q, i32 ldq,
            i32* neig, f64 tol, i32* info);

/**
 * @brief MB03KA - Move diagonal blocks in generalized periodic Schur form
 *
 * Reorders the diagonal blocks of the formal matrix product
 * T22_K^S(K) * T22_K-1^S(K-1) * ... * T22_1^S(1) of length K
 * such that the block with starting row index IFST is moved to row index ILST.
 *
 * @param[in] compq 'N'=no Q, 'U'=update Q, 'W'=selective Q update via WHICHQ
 * @param[in] whichq Array(K) specifying Q_k computation when COMPQ='W'
 * @param[in] ws If true, perform strong stability tests
 * @param[in] k Period of the periodic matrix sequences (K >= 2)
 * @param[in] nc Number of core eigenvalues (0 <= NC <= min(N))
 * @param[in] kschur Index for which T22_kschur is upper quasi-triangular
 * @param[in,out] ifst Starting row index of block to move (1-based, adjusted on exit)
 * @param[in,out] ilst Target row index for block (1-based, adjusted on exit)
 * @param[in] n Array(K) of matrix dimensions
 * @param[in] ni Array(K) of T11_k dimensions
 * @param[in] s Array(K) of signatures (+1 or -1)
 * @param[in,out] t Flattened array containing K matrices T_k at positions IXT(k)
 * @param[in] ldt Array(K) of leading dimensions for T_k
 * @param[in] ixt Array(K) of start indices (1-based) for T_k in T
 * @param[in,out] q Flattened array containing K matrices Q_k at positions IXQ(k)
 * @param[in] ldq Array(K) of leading dimensions for Q_k
 * @param[in] ixq Array(K) of start indices (1-based) for Q_k in Q
 * @param[in] tol Array(3): [c, EPS, SMLNUM] tolerance parameters
 * @param[out] iwork Integer workspace array(4*K)
 * @param[out] dwork Double workspace array(LDWORK)
 * @param[in] ldwork Size of DWORK; -1 for workspace query
 * @param[out] info 0=success, -21=LDWORK too small, 1=reordering failed
 */
void mb03ka(const char* compq, const i32* whichq, bool ws,
            i32 k, i32 nc, i32 kschur, i32* ifst, i32* ilst,
            const i32* n, const i32* ni, const i32* s,
            f64* t, const i32* ldt, const i32* ixt,
            f64* q, const i32* ldq, const i32* ixq,
            const f64* tol, i32* iwork, f64* dwork,
            i32 ldwork, i32* info);

/**
 * @brief Swap pairs of adjacent diagonal blocks in generalized periodic Schur form
 *
 * Reorders the diagonal blocks of the formal matrix product
 * T22_K^S(K) * T22_K-1^S(K-1) * ... * T22_1^S(1) of length K
 * in generalized periodic Schur form such that pairs of adjacent
 * diagonal blocks of sizes 1 and/or 2 are swapped.
 *
 * @param[in] compq 'N'=no Q, 'U'=update Q, 'W'=selective Q update via WHICHQ
 * @param[in] whichq Array(K) specifying Q_k computation when COMPQ='W'
 * @param[in] ws If true, perform strong stability tests
 * @param[in] k Period of the periodic matrix sequences (K >= 2)
 * @param[in] nc Number of core eigenvalues (0 <= NC <= min(N))
 * @param[in] kschur Index for which T22_kschur is upper quasi-triangular
 * @param[in] j1 First row/column index of first block to swap (1-based)
 * @param[in] n1 Order of first block (0, 1, or 2)
 * @param[in] n2 Order of second block (0, 1, or 2)
 * @param[in] n Array(K) of matrix dimensions
 * @param[in] ni Array(K) of T11_k dimensions
 * @param[in] s Array(K) of signatures (+1 or -1)
 * @param[in,out] t Flattened array containing K matrices T_k at positions IXT(k)
 * @param[in] ldt Array(K) of leading dimensions for T_k
 * @param[in] ixt Array(K) of start indices (1-based) for T_k in T
 * @param[in,out] q Flattened array containing K matrices Q_k at positions IXQ(k)
 * @param[in] ldq Array(K) of leading dimensions for Q_k
 * @param[in] ixq Array(K) of start indices (1-based) for Q_k in Q
 * @param[in] tol Array(3): [c, EPS, SMLNUM] tolerance parameters
 * @param[out] iwork Integer workspace array(4*K)
 * @param[out] dwork Double workspace array(LDWORK)
 * @param[in] ldwork Size of DWORK; -1 for workspace query
 * @param[out] info 0=success, -22=LDWORK too small, 1=swap rejected
 */
void mb03kb(const char* compq, const i32* whichq, bool ws,
            i32 k, i32 nc, i32 kschur, i32 j1, i32 n1, i32 n2,
            const i32* n, const i32* ni, const i32* s,
            f64* t, const i32* ldt, const i32* ixt,
            f64* q, const i32* ldq, const i32* ixq,
            const f64* tol, i32* iwork, f64* dwork,
            i32 ldwork, i32* info);

/**
 * @brief MB03QG - Reorder diagonal blocks of an upper quasi-triangular
 *                 matrix pencil A-lambda*E
 *
 * Reorders the diagonal blocks of a principal subpencil of an upper
 * quasi-triangular matrix pencil A-lambda*E together with their
 * generalized eigenvalues, by constructing orthogonal similarity
 * transformations UT and VT.
 *
 * @param[in] dico 'C'=continuous-time, 'D'=discrete-time
 * @param[in] stdom 'S'=stability domain, 'U'=instability domain
 * @param[in] jobu 'I'=init U to identity, 'U'=update given U
 * @param[in] jobv 'I'=init V to identity, 'U'=update given V
 * @param[in] n Order of matrices A, E, U, V (N >= 0)
 * @param[in] nlow Lower boundary index for subpencil (0 <= NLOW <= NSUP)
 * @param[in] nsup Upper boundary index for subpencil (NLOW <= NSUP <= N)
 * @param[in] alpha Boundary of domain of interest (ALPHA >= 0 for DICO='D')
 * @param[in,out] a N-by-N Schur form matrix, reordered on exit
 * @param[in] lda Leading dimension of A (LDA >= max(1,N))
 * @param[in,out] e N-by-N upper triangular matrix, updated on exit
 * @param[in] lde Leading dimension of E (LDE >= max(1,N))
 * @param[in,out] u N-by-N transformation matrix (accumulated)
 * @param[in] ldu Leading dimension of U (LDU >= max(1,N))
 * @param[in,out] v N-by-N transformation matrix (accumulated)
 * @param[in] ldv Leading dimension of V (LDV >= max(1,N))
 * @param[out] ndim Number of eigenvalues in domain of interest
 * @param[out] dwork Workspace array(LDWORK)
 * @param[in] ldwork Size of DWORK (>= 1, >= 4*N+16 if N > 1; -1 for query)
 * @param[out] info 0=success, <0=param error, 1=block boundary error, 2=swap failed
 */
void mb03qg(const char *dico, const char *stdom, const char *jobu, const char *jobv,
            i32 n, i32 nlow, i32 nsup, f64 alpha,
            f64 *a, i32 lda, f64 *e, i32 lde,
            f64 *u, i32 ldu, f64 *v, i32 ldv,
            i32 *ndim, f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief MB03RW - Solve complex Sylvester equation -AX + XB = C
 *
 * Solves the Sylvester equation -AX + XB = C where A (M-by-M) and B (N-by-N)
 * are complex upper triangular matrices in Schur form. Aborts if any element
 * of X exceeds PMAX in absolute value.
 *
 * @param[in] m Order of A, number of rows of C and X (M >= 0)
 * @param[in] n Order of B, number of columns of C and X (N >= 0)
 * @param[in] pmax Upper bound for absolute value of X elements
 * @param[in] a M-by-M complex upper triangular matrix A
 * @param[in] lda Leading dimension of A (LDA >= max(1,M))
 * @param[in] b N-by-N complex upper triangular matrix B
 * @param[in] ldb Leading dimension of B (LDB >= max(1,N))
 * @param[in,out] c On entry: M-by-N RHS matrix C. On exit: solution X
 * @param[in] ldc Leading dimension of C (LDC >= max(1,M))
 * @param[out] info 0=success, 1=element of X exceeds PMAX, 2=perturbed values used
 */
void mb03rw(i32 m, i32 n, f64 pmax, const c128 *a, i32 lda, const c128 *b,
            i32 ldb, c128 *c, i32 ldc, i32 *info);

/**
 * @brief MB03SD - Eigenvalues of a square-reduced Hamiltonian matrix
 *
 * Computes the eigenvalues of an N-by-N square-reduced Hamiltonian matrix
 *
 *              ( A'   G'  )
 *       H'  =  (        T ).
 *              ( Q'  -A'  )
 *
 * where A' is N-by-N, and G' and Q' are symmetric N-by-N matrices. The matrix
 * H' is assumed to be square-reduced (from MB04ZD), meaning H'^2 has the form:
 *
 *         2    ( A''   G'' )
 *       H'  =  (         T )    with A'' upper Hessenberg.
 *              ( 0    A''  )
 *
 * The eigenvalues of H' are computed as square roots of eigenvalues of A''.
 *
 * @param[in] jobscl Specifies whether to use balancing:
 *                   - 'N': Do not use balancing
 *                   - 'S': Do scaling to equilibrate rows/columns of A''
 * @param[in] n Order of matrices A, G, and Q (N >= 0)
 * @param[in] a N-by-N upper left block A' of H'
 * @param[in] lda Leading dimension of A (LDA >= max(1,N))
 * @param[in] qg N-by-(N+1) array containing Q' and G':
 *               - Q'(i,j) stored in QG(i,j) for i >= j (lower triangular)
 *               - G'(i,j) stored in QG(j,i+1) for i >= j (upper triangular in cols 2:N+1)
 * @param[in] ldqg Leading dimension of QG (LDQG >= max(1,N))
 * @param[out] wr Real parts of N eigenvalues with non-negative real part
 * @param[out] wi Imaginary parts of N eigenvalues with non-negative real part
 * @param[out] dwork Workspace array of size LDWORK
 * @param[in] ldwork Workspace size (LDWORK >= max(1, N*(N+1)))
 * @param[out] info Exit status:
 *                  - 0: Success
 *                  - <0: -i means i-th argument is invalid
 *                  - >0: DHSEQR failed to converge at i-th eigenvalue
 */
void mb03sd(const char *jobscl, i32 n, f64 *a, i32 lda, f64 *qg, i32 ldqg,
            f64 *wr, f64 *wi, f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief MB03KD - Reorder diagonal blocks in generalized periodic Schur form
 *
 * Reorders the diagonal blocks of a formal matrix product
 * T22_K^S(K) * T22_K-1^S(K-1) * ... * T22_1^S(1) of length K,
 * such that the M selected eigenvalues pointed to by SELECT
 * end up in the leading part of the matrix sequence T22_k.
 *
 * @param[in] compq Specifies computation of Q:
 *                  - 'N': do not compute Q
 *                  - 'I': initialize Q to identity, compute Q
 *                  - 'U': update existing Q
 *                  - 'W': selective computation via whichq
 * @param[in] whichq If compq='W', specifies Q computation per factor (array of K)
 *                   0=none, 1=identity init, 2=update
 * @param[in] strong 'N' no strong stability tests, 'S' perform them
 * @param[in] k Period of periodic matrix sequences (K >= 2)
 * @param[in] nc Number of core eigenvalues (0 <= NC <= min(N))
 * @param[in] kschur Index for quasi-triangular T22_kschur (1 <= KSCHUR <= K)
 * @param[in] n Array of K dimensions for factors
 * @param[in] ni Array of K dimensions for T11_k factors
 * @param[in] s Array of K signatures (+1 or -1)
 * @param[in] select Logical array of NC specifying eigenvalue selection
 * @param[in,out] t Packed array containing K matrices T_k
 * @param[in] ldt Array of K leading dimensions for T_k
 * @param[in] ixt Array of K start indices in t (1-based)
 * @param[in,out] q Packed array containing K matrices Q_k
 * @param[in] ldq Array of K leading dimensions for Q_k
 * @param[in] ixq Array of K start indices in q (1-based)
 * @param[out] m Number of selected eigenvalues reordered
 * @param[in] tol Tolerance parameter (should be >= 10)
 * @param[out] iwork Integer workspace (4*K)
 * @param[out] dwork Double workspace (LDWORK)
 * @param[in] ldwork Workspace size, or -1 for query
 * @param[out] info Exit status: 0=success, <0=param error, 1=reordering failed
 */
void mb03kd(const char* compq, const i32* whichq, const char* strong,
            i32 k, i32 nc, i32 kschur,
            const i32* n, const i32* ni, const i32* s,
            const bool* select, f64* t, const i32* ldt, const i32* ixt,
            f64* q, const i32* ldq, const i32* ixq,
            i32* m, f64 tol, i32* iwork, f64* dwork,
            i32 ldwork, i32* info);

/**
 * @brief MB03RZ - Reduce complex Schur form to block-diagonal form
 *
 * Reduces an upper triangular complex matrix A (Schur form) to a
 * block-diagonal form using well-conditioned non-unitary similarity
 * transformations. The condition numbers of the transformations used
 * for reduction are roughly bounded by PMAX. The transformations are
 * optionally postmultiplied in a given matrix X. The Schur form is
 * optionally ordered, so that clustered eigenvalues are grouped in the
 * same block.
 *
 * @param[in] jobx 'N': transformations not accumulated;
 *                 'U': transformations accumulated in X (X is updated)
 * @param[in] sort 'N': no reordering;
 *                 'S': reorder before each step for clustered eigenvalues;
 *                 'C': closest-neighbour strategy;
 *                 'B': reordering + closest-neighbour
 * @param[in] n Order of matrices A and X (N >= 0)
 * @param[in] pmax Upper bound for transformation elements (PMAX >= 1.0)
 * @param[in,out] a N-by-N upper triangular matrix A; on exit, block-diagonal
 * @param[in] lda Leading dimension of A (LDA >= max(1,N))
 * @param[in,out] x If JOBX='U', N-by-N input matrix; on exit, X * transformation
 * @param[in] ldx Leading dimension of X (LDX >= 1, or LDX >= N if JOBX='U')
 * @param[out] nblcks Number of diagonal blocks
 * @param[out] blsize Array of NBLCKS block sizes
 * @param[out] w Array of N eigenvalues
 * @param[in] tol Tolerance for eigenvalue clustering (see documentation)
 * @param[out] info 0: success; <0: parameter error
 */
void mb03rz(const char* jobx, const char* sort, i32 n, f64 pmax,
            c128* a, i32 lda, c128* x, i32 ldx, i32* nblcks,
            i32* blsize, c128* w, f64 tol, i32* info);

/**
 * @brief MB03TS - Swap diagonal blocks in (skew-)Hamiltonian Schur form
 *
 * Swaps diagonal blocks A11 and A22 of order 1 or 2 in the upper
 * quasi-triangular matrix A contained in a skew-Hamiltonian matrix
 *
 *           [  A   G  ]          T
 *     X  =  [       T ],   G = -G,
 *           [  0   A  ]
 *
 * or in a Hamiltonian matrix
 *
 *           [  A   G  ]          T
 *     X  =  [       T ],   G =  G.
 *           [  0  -A  ]
 *
 * This is a modified version of LAPACK subroutine DLAEX2.
 *
 * @param[in] isham true=Hamiltonian, false=skew-Hamiltonian matrix
 * @param[in] wantu true=update U1 and U2, false=do not update
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in,out] a N-by-N upper quasi-triangular matrix in Schur form
 * @param[in] lda Leading dimension of a (lda >= max(1,n))
 * @param[in,out] g N-by-N symmetric (ISHAM) or skew-symmetric (!ISHAM) matrix
 * @param[in] ldg Leading dimension of g (ldg >= max(1,n))
 * @param[in,out] u1 N-by-N matrix U1 (if wantu)
 * @param[in] ldu1 Leading dimension of u1 (ldu1 >= max(1,n) if wantu)
 * @param[in,out] u2 N-by-N matrix U2 (if wantu)
 * @param[in] ldu2 Leading dimension of u2 (ldu2 >= max(1,n) if wantu)
 * @param[in] j1 Index of first row of first block A11 (1-based)
 * @param[in] n1 Order of first block A11 (0, 1, or 2)
 * @param[in] n2 Order of second block A22 (0, 1, or 2)
 * @param[out] dwork Workspace array of dimension (n)
 * @param[out] info 0=success, 1=swap rejected (result would be far from Schur form)
 */
void mb03ts(bool isham, bool wantu, i32 n, f64 *a, i32 lda, f64 *g, i32 ldg,
            f64 *u1, i32 ldu1, f64 *u2, i32 ldu2, i32 j1, i32 n1, i32 n2,
            f64 *dwork, i32 *info);

/**
 * @brief Reorder diagonal blocks of (skew-)Hamiltonian Schur form.
 *
 * Reorders a matrix X in skew-Hamiltonian Schur form:
 *     X = [A  G; 0  A^T], G = -G^T (skew-symmetric)
 * or Hamiltonian Schur form:
 *     X = [A  G; 0  -A^T], G = G^T (symmetric)
 *
 * so that selected eigenvalues appear in leading diagonal blocks of A.
 *
 * @param[in] typ 'S'=skew-Hamiltonian, 'H'=Hamiltonian
 * @param[in] compu 'U'=update U1,U2, 'N'=do not update
 * @param[in,out] select Array of dimension n, eigenvalue selection
 * @param[in,out] lower Array of dimension n, controls which eigenvalue copy
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in,out] a N-by-N upper quasi-triangular matrix in Schur form
 * @param[in] lda Leading dimension of a (lda >= max(1,n))
 * @param[in,out] g N-by-N symmetric (H) or skew-symmetric (S) matrix
 * @param[in] ldg Leading dimension of g (ldg >= max(1,n))
 * @param[in,out] u1 N-by-N matrix U1 (if compu='U')
 * @param[in] ldu1 Leading dimension of u1
 * @param[in,out] u2 N-by-N matrix U2 (if compu='U')
 * @param[in] ldu2 Leading dimension of u2
 * @param[out] wr Real parts of reordered eigenvalues
 * @param[out] wi Imaginary parts of reordered eigenvalues
 * @param[out] m Dimension of specified invariant subspace
 * @param[out] dwork Workspace array of dimension ldwork
 * @param[in] ldwork Length of dwork (ldwork >= max(1,n))
 * @param[out] info 0=success, <0=param error, 1=reordering failed
 */
void mb03td(const char *typ, const char *compu, bool *select, bool *lower,
            i32 n, f64 *a, i32 lda, f64 *g, i32 ldg,
            f64 *u1, i32 ldu1, f64 *u2, i32 ldu2,
            f64 *wr, f64 *wi, i32 *m, f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Compute eigenvalues of a product of matrices in periodic Schur form.
 *
 * Computes eigenvalues of T = T_1*T_2*...*T_p where T_1 is upper
 * quasi-triangular (real Schur form) and T_2, ..., T_p are upper triangular.
 *
 * @param[in] n Order of matrix T (n >= 0)
 * @param[in] p Number of matrices in product (p >= 1)
 * @param[in] t 3D array (ldt1 x ldt2 x p). T(*,*,1) is upper quasi-triangular,
 *              T(*,*,j) for j > 1 are upper triangular.
 * @param[in] ldt1 First leading dimension of t (ldt1 >= max(1,n))
 * @param[in] ldt2 Second leading dimension of t (ldt2 >= max(1,n))
 * @param[out] wr Real parts of eigenvalues, dimension (n)
 * @param[out] wi Imaginary parts of eigenvalues, dimension (n)
 * @param[out] info 0=success, <0=param error
 */
void mb03wx(i32 n, i32 p, const f64 *t, i32 ldt1, i32 ldt2,
            f64 *wr, f64 *wi, i32 *info);

/**
 * @brief Reduce 2x2 or 4x4 block diagonal skew-Hamiltonian/Hamiltonian pencil
 *        to generalized Schur form.
 *
 * Computes orthogonal matrices Q1 and Q2 for a real 2-by-2 or 4-by-4 regular
 * pencil aA - bB where:
 *   A = [A11  0 ; 0  A22]  (block diagonal, upper triangular blocks)
 *   B = [0  B12; B21  0]   (anti-block-diagonal)
 *
 * such that Q2' A Q1 is upper triangular, Q2' B Q1 is upper quasi-triangular,
 * and eigenvalues with negative real parts are allocated on top.
 *
 * @param[in] n Order of pencil, must be 2 or 4
 * @param[in] prec Machine precision (relative machine precision * base)
 * @param[in,out] a N-by-N matrix A. On exit (N=4), transformed upper triangular.
 * @param[in] lda Leading dimension of a (lda >= n)
 * @param[in,out] b N-by-N matrix B. On exit (N=4), transformed quasi-triangular.
 * @param[in] ldb Leading dimension of b (ldb >= n)
 * @param[out] q1 N-by-N first orthogonal transformation matrix
 * @param[in] ldq1 Leading dimension of q1 (ldq1 >= n)
 * @param[out] q2 N-by-N second orthogonal transformation matrix
 * @param[in] ldq2 Leading dimension of q2 (ldq2 >= n)
 * @param[out] dwork Workspace (N=4: dimension >= 63; N=2: not referenced)
 * @param[in] ldwork Workspace size (N=4: >= 63; N=2: >= 0)
 * @param[out] info 0=success, 1=QZ iteration failed in DGGES, 2=other DGGES error
 */
void mb03fd(i32 n, f64 prec, f64 *a, i32 lda, f64 *b, i32 ldb,
            f64 *q1, i32 ldq1, f64 *q2, i32 ldq2,
            f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Eigenvalues and deflating subspace of complex skew-Hamiltonian/Hamiltonian pencil
 *
 * Computes eigenvalues of a complex N-by-N skew-Hamiltonian/Hamiltonian pencil
 * aS - bH with S = J Z' J' Z and H = [B F; G -B'] where J = [0 I; -I 0].
 *
 * Optionally computes orthonormal basis of right deflating subspace (Q) and
 * companion subspace (U) corresponding to eigenvalues with strictly negative
 * real part.
 *
 * @param[in] compq 'N'=no Q, 'C'=compute Q
 * @param[in] compu 'N'=no U, 'C'=compute U
 * @param[in] orth Orthogonalization method: 'P'=QR with pivoting, 'S'=SVD
 * @param[in] n Order of pencil (n >= 0, even)
 * @param[in,out] z N-by-N complex matrix Z in S = J Z' J' Z
 * @param[in] ldz Leading dimension of Z (ldz >= max(1,n))
 * @param[in,out] b N/2-by-N/2 complex matrix B of Hamiltonian H
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in,out] fg N/2-by-(N/2+1) packed F and G: lower triangle = G, upper = F
 * @param[in] ldfg Leading dimension of FG (ldfg >= max(1,n))
 * @param[out] neig Number of eigenvalues with strictly negative real part
 * @param[out] d N-by-N complex output matrix BD (if compq='C' or compu='C')
 * @param[in] ldd Leading dimension of D
 * @param[out] c N-by-N complex output matrix BC (lower triangular, if compq='C' or compu='C')
 * @param[in] ldc Leading dimension of C
 * @param[out] q N-by-NEIG right deflating subspace (if compq='C')
 * @param[in] ldq Leading dimension of Q (ldq >= 1 if compq='N', ldq >= 2*n otherwise)
 * @param[out] u N-by-NEIG companion subspace (if compu='C')
 * @param[in] ldu Leading dimension of U (ldu >= 1 if compu='N', ldu >= n otherwise)
 * @param[out] alphar Real parts of eigenvalues, dimension (n)
 * @param[out] alphai Imaginary parts of eigenvalues, dimension (n)
 * @param[out] beta Eigenvalue denominators, dimension (n). Eigenvalue = alpha/beta
 * @param[out] iwork Integer workspace, dimension (liwork)
 * @param[in] liwork Dimension of iwork (liwork >= 2*n+9)
 * @param[out] dwork Double workspace, dimension (ldwork)
 * @param[in] ldwork Dimension of dwork (see documentation for formula)
 * @param[out] zwork Complex workspace, dimension (lzwork)
 * @param[in] lzwork Dimension of zwork (see documentation for formula)
 * @param[out] bwork Logical workspace, dimension (n) if compq='C' or compu='C'
 * @param[out] info 0=success, 1=MB03BD error, 2=QZ iteration failed, 3=SVD failed
 */
void mb03fz(const char *compq, const char *compu, const char *orth, i32 n,
            c128 *z, i32 ldz, c128 *b, i32 ldb, c128 *fg, i32 ldfg, i32 *neig,
            c128 *d, i32 ldd, c128 *c, i32 ldc, c128 *q, i32 ldq, c128 *u,
            i32 ldu, f64 *alphar, f64 *alphai, f64 *beta, i32 *iwork,
            i32 liwork, f64 *dwork, i32 ldwork, c128 *zwork, i32 lzwork,
            bool *bwork, i32 *info);

/**
 * @brief Eigenvalues and right deflating subspace of complex skew-Hamiltonian/Hamiltonian pencil
 *
 * Computes eigenvalues of a complex N-by-N skew-Hamiltonian/Hamiltonian pencil
 * aS - bH with:
 *   S = [[A, D], [E, A']]  where D, E are skew-Hermitian
 *   H = [[B, F], [G, -B']] where F, G are Hermitian
 *
 * Optionally computes orthonormal basis of right deflating subspace
 * corresponding to eigenvalues with strictly negative real part.
 *
 * @param[in] compq 'N'=eigenvalues only, 'C'=compute deflating subspace
 * @param[in] orth Orthogonalization: 'P'=QR with pivoting, 'S'=SVD (only if compq='C')
 * @param[in] n Order of pencil (n >= 0, even)
 * @param[in,out] a N/2-by-N/2 complex matrix A. If compq='C', on exit contains N-by-N BA.
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] de N/2-by-(N/2+1) packed D and E. Lower triangle = E, cols 2:N/2+1 upper = D.
 * @param[in] ldde Leading dimension of DE (ldde >= max(1,n))
 * @param[in,out] b N/2-by-N/2 complex matrix B. If compq='C', on exit contains N-by-N BB.
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in,out] fg N/2-by-(N/2+1) packed F and G. Lower triangle = G, cols 2:N/2+1 upper = F.
 * @param[in] ldfg Leading dimension of FG (ldfg >= max(1,n))
 * @param[out] neig Number of eigenvalues with strictly negative real part (if compq='C')
 * @param[out] q N-by-NEIG right deflating subspace (if compq='C'), dimension (ldq, 2*n)
 * @param[in] ldq Leading dimension of Q (ldq >= 1 if compq='N', ldq >= 2*n otherwise)
 * @param[out] alphar Real parts of eigenvalues, dimension (n)
 * @param[out] alphai Imaginary parts of eigenvalues, dimension (n)
 * @param[out] beta Eigenvalue denominators, dimension (n). Eigenvalue = alpha/beta
 * @param[out] iwork Integer workspace, dimension (n+1)
 * @param[out] dwork Double workspace, dimension (ldwork)
 * @param[in] ldwork Dimension of dwork. Use -1 for workspace query.
 * @param[out] zwork Complex workspace, dimension (lzwork)
 * @param[in] lzwork Dimension of zwork. Use -1 for workspace query.
 * @param[out] bwork Logical workspace, dimension (n-1) if compq='C'
 * @param[out] info 0=success, 1=MB04FD QZ failed, 2=ZHGEQZ failed, 3=SVD failed, 4=pencil singular
 */
void mb03lz(const char *compq, const char *orth, i32 n,
            c128 *a, i32 lda, c128 *de, i32 ldde, c128 *b, i32 ldb,
            c128 *fg, i32 ldfg, i32 *neig, c128 *q, i32 ldq,
            f64 *alphar, f64 *alphai, f64 *beta,
            i32 *iwork, f64 *dwork, i32 ldwork,
            c128 *zwork, i32 lzwork, bool *bwork, i32 *info);

/**
 * @brief Eigenvalues and right deflating subspace of real skew-Hamiltonian/Hamiltonian pencil
 *
 * Computes relevant eigenvalues of a real N-by-N skew-Hamiltonian/Hamiltonian pencil
 * aS - bH with:
 *   S = [[A, D], [E, A']]  where D, E are skew-symmetric
 *   H = [[B, F], [G, -B']] where F, G are symmetric
 *
 * Optionally computes orthonormal basis of right deflating subspace
 * corresponding to eigenvalues with strictly negative real part.
 *
 * @param[in] compq 'N'=eigenvalues only, 'C'=compute deflating subspace
 * @param[in] orth Orthogonalization: 'P'=QR with pivoting, 'S'=SVD (only if compq='C')
 * @param[in] n Order of pencil (n >= 0, even)
 * @param[in,out] a N/2-by-N/2 matrix A. On exit contains Aout (upper triangular)
 * @param[in] lda Leading dimension of A (lda >= max(1,n/2))
 * @param[in,out] de N/2-by-(N/2+1) packed D and E. Lower = E, cols 2:N/2+1 upper = D.
 * @param[in] ldde Leading dimension of DE (ldde >= max(1,n/2))
 * @param[in,out] b N/2-by-N/2 matrix B. On exit contains C1out (upper triangular)
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n/2))
 * @param[in,out] fg N/2-by-(N/2+1) packed F and G. Lower = G, cols 2:N/2+1 upper = F.
 * @param[in] ldfg Leading dimension of FG (ldfg >= max(1,n/2))
 * @param[out] neig Number of eigenvalues with strictly negative real part (if compq='C')
 * @param[out] q N-by-NEIG right deflating subspace (if compq='C'), dimension (ldq, 2*n)
 * @param[in] ldq Leading dimension of Q (ldq >= 1 if compq='N', ldq >= 2*n otherwise)
 * @param[out] alphar Real parts of eigenvalues, dimension (n/2)
 * @param[out] alphai Imaginary parts of eigenvalues, dimension (n/2)
 * @param[out] beta Eigenvalue denominators, dimension (n/2). Eigenvalue = alpha/beta
 * @param[out] iwork Integer workspace, dimension (liwork)
 * @param[in] liwork Dimension of iwork
 * @param[out] dwork Double workspace, dimension (ldwork). dwork[0] returns optimal size.
 * @param[in] ldwork Dimension of dwork. Use -1 for workspace query.
 * @param[out] bwork Logical workspace, dimension (n/2)
 * @param[out] info 0=success, 1=MB04BD/MB04HD QZ failed, 2=MB04HD/MB03DD failed,
 *                  3=MB03HD singular, 4=DGESVD failed, 5=eigenvalues may be inaccurate (warning)
 */
void mb03ld(const char *compq, const char *orth, i32 n,
            f64 *a, i32 lda, f64 *de, i32 ldde,
            f64 *b, i32 ldb, f64 *fg, i32 ldfg,
            i32 *neig, f64 *q, i32 ldq,
            f64 *alphar, f64 *alphai, f64 *beta,
            i32 *iwork, i32 liwork, f64 *dwork, i32 ldwork,
            i32 *bwork, i32 *info);

/**
 * @brief Eigenvalues and right deflating subspace of real skew-Hamiltonian/Hamiltonian pencil (panel-based)
 *
 * Same as MB03LD but applies transformations on panels of columns for better
 * performance on large matrices. Uses MB04BP instead of MB04BD internally.
 *
 * Computes relevant eigenvalues of a real N-by-N skew-Hamiltonian/Hamiltonian pencil
 * aS - bH with:
 *   S = [[A, D], [E, A']]  where D, E are skew-symmetric
 *   H = [[B, F], [G, -B']] where F, G are symmetric
 *
 * Optionally computes orthonormal basis of right deflating subspace
 * corresponding to eigenvalues with strictly negative real part.
 *
 * @param[in] compq 'N'=eigenvalues only, 'C'=compute deflating subspace
 * @param[in] orth Orthogonalization: 'P'=QR with pivoting, 'S'=SVD (only if compq='C')
 * @param[in] n Order of pencil (n >= 0, even)
 * @param[in,out] a N/2-by-N/2 matrix A. On exit contains Aout (upper triangular)
 * @param[in] lda Leading dimension of A (lda >= max(1,n/2))
 * @param[in,out] de N/2-by-(N/2+1) packed D and E. Lower = E, cols 2:N/2+1 upper = D.
 * @param[in] ldde Leading dimension of DE (ldde >= max(1,n/2))
 * @param[in,out] b N/2-by-N/2 matrix B. On exit contains C1out (upper triangular)
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n/2))
 * @param[in,out] fg N/2-by-(N/2+1) packed F and G. Lower = G, cols 2:N/2+1 upper = F.
 * @param[in] ldfg Leading dimension of FG (ldfg >= max(1,n/2))
 * @param[out] neig Number of eigenvalues with strictly negative real part (if compq='C')
 * @param[out] q N-by-NEIG right deflating subspace (if compq='C'), dimension (ldq, 2*n)
 * @param[in] ldq Leading dimension of Q (ldq >= 1 if compq='N', ldq >= 2*n otherwise)
 * @param[out] alphar Real parts of eigenvalues, dimension (n/2)
 * @param[out] alphai Imaginary parts of eigenvalues, dimension (n/2)
 * @param[out] beta Eigenvalue denominators, dimension (n/2). Eigenvalue = alpha/beta
 * @param[out] iwork Integer workspace, dimension (liwork)
 * @param[in] liwork Dimension of iwork
 * @param[out] dwork Double workspace, dimension (ldwork). dwork[0] returns optimal size.
 * @param[in] ldwork Dimension of dwork. Use -1 for workspace query.
 * @param[out] bwork Logical workspace, dimension (n/2)
 * @param[out] info 0=success, 1=MB04BP/MB04HD QZ failed, 2=MB04HD/MB03DD failed,
 *                  3=MB03HD singular, 4=DGESVD failed, 5=eigenvalues may be inaccurate (warning)
 */
void mb03lp(const char *compq, const char *orth, i32 n,
            f64 *a, i32 lda, f64 *de, i32 ldde,
            f64 *b, i32 ldb, f64 *fg, i32 ldfg,
            i32 *neig, f64 *q, i32 ldq,
            f64 *alphar, f64 *alphai, f64 *beta,
            i32 *iwork, i32 liwork, f64 *dwork, i32 ldwork,
            i32 *bwork, i32 *info);

/**
 * @brief Eigenvalues and real skew-Hamiltonian Schur form of a skew-Hamiltonian matrix.
 *
 * Computes eigenvalues and real skew-Hamiltonian Schur form of W = [[A, G], [Q, A^T]]
 * where G, Q are N-by-N skew-symmetric matrices. Computes orthogonal symplectic U s.t.
 * U^T W U = [[Aout, Gout], [0, Aout^T]] with Aout in Schur canonical form.
 *
 * Optionally returns U in terms of U1, U2 where U = [[U1, U2], [-U2, U1]].
 *
 * @param[in] jobu 'N' = don't compute U, 'U' = compute U1 and U2
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in,out] a On entry: N-by-N matrix A. On exit: Aout in Schur form.
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] qg On entry: N-by-(N+1), cols 0:N = strictly lower tri Q, cols 1:N+1 = strictly upper tri G.
 *                   On exit: cols 1:N+1 contain strictly upper tri Gout. Q part zeroed.
 * @param[in] ldqg Leading dimension of QG (ldqg >= max(1,n))
 * @param[out] u1 N-by-N matrix U1 (if jobu='U'), not referenced if jobu='N'
 * @param[in] ldu1 Leading dimension of U1 (ldu1 >= max(1,n) if jobu='U', ldu1 >= 1 otherwise)
 * @param[out] u2 N-by-N matrix U2 (if jobu='U'), not referenced if jobu='N'
 * @param[in] ldu2 Leading dimension of U2 (ldu2 >= max(1,n) if jobu='U', ldu2 >= 1 otherwise)
 * @param[out] wr Real parts of eigenvalues of Aout, dimension (n)
 * @param[out] wi Imaginary parts of eigenvalues of Aout, dimension (n)
 * @param[out] dwork Workspace, dimension (ldwork). dwork[0] returns optimal size.
 * @param[in] ldwork Workspace size. max(1,(n+5)*n) if jobu='U', max(1,5*n,(n+1)*n) if jobu='N'.
 *                   Use -1 for workspace query.
 * @param[out] info 0 = success, < 0 = -i-th arg invalid, > 0 = DHSEQR failed at eigenvalue i.
 */
void mb03xs(const char *jobu, i32 n, f64 *a, i32 lda,
            f64 *qg, i32 ldqg,
            f64 *u1, i32 ldu1, f64 *u2, i32 ldu2,
            f64 *wr, f64 *wi,
            f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Eigenvalues of a complex Hamiltonian matrix.
 *
 * Computes eigenvalues of a complex Hamiltonian matrix:
 *     H = [A    G  ]    where G = G^H, Q = Q^H
 *         [Q   -A^H]
 *
 * Uses embedding to real skew-Hamiltonian matrix and structured Schur form.
 * Due to Hamiltonian structure, if lambda is an eigenvalue, -conj(lambda) is also.
 *
 * Optionally computes Schur form Sc and matrix Gc of the decomposition:
 *     U^H (i*He) U = [Sc   Gc ]    Gc = Gc^H
 *                    [0   -Sc^H]
 *
 * @param[in] balanc 'N'=no balancing, 'P'=permute, 'S'=scale, 'B'=both
 * @param[in] job 'E'=eigenvalues only, 'S'=compute Sc, 'G'=compute Sc and Gc
 * @param[in] jobu 'N'=don't compute U, 'U'=compute U
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in,out] a On entry: N-by-N complex matrix A.
 *                  On exit if job='E': balanced A (if balanc!='N').
 *                  On exit if job!='E': 2N-by-2N upper triangular Sc.
 * @param[in] lda Leading dimension of A (lda >= max(1,k), k=n if job='E', k=2n otherwise)
 * @param[in,out] qg On entry: N-by-(N+1), lower tri Q in cols 0:N, upper tri G in cols 1:N+1.
 *                   On exit if job='G': 2N-by-2N upper triangular Gc.
 * @param[in] ldqg Leading dimension of QG (ldqg >= max(1,k))
 * @param[out] u1 2N-by-2N matrix U1 (if job!='E' and jobu='U'), block (1,1) of U
 * @param[in] ldu1 Leading dimension of U1 (ldu1 >= 2n if jobu='U', ldu1 >= 1 otherwise)
 * @param[out] u2 2N-by-2N matrix U2 (if job!='E' and jobu='U'), block (1,2) of U
 * @param[in] ldu2 Leading dimension of U2 (ldu2 >= 2n if jobu='U', ldu2 >= 1 otherwise)
 * @param[out] wr Real parts of 2N eigenvalues, dimension (2*n)
 * @param[out] wi Imaginary parts of 2N eigenvalues, dimension (2*n)
 * @param[out] ilo Index from balancing (1 if balanc='N')
 * @param[out] scale Scaling factors from balancing, dimension (n)
 * @param[out] dwork Real workspace. dwork[0]=optimal size, dwork[1]=1-norm of H.
 * @param[in] ldwork Real workspace size. Use -1 for query.
 * @param[out] zwork Complex workspace. zwork[0]=optimal size.
 * @param[in] lzwork Complex workspace size. Use -1 for query.
 * @param[out] bwork Logical workspace, dimension (2*n-1) if job!='E', unused otherwise.
 * @param[out] info 0=success, <0=-i-th arg invalid, >0=QR algorithm failed, 2*n+1=2x2 block failed
 */
void mb03xz(const char *balanc, const char *job, const char *jobu, i32 n,
            c128 *a, i32 lda, c128 *qg, i32 ldqg,
            c128 *u1, i32 ldu1, c128 *u2, i32 ldu2,
            f64 *wr, f64 *wi, i32 *ilo, f64 *scale,
            f64 *dwork, i32 ldwork, c128 *zwork, i32 lzwork,
            bool *bwork, i32 *info);

/**
 * @brief Reorder eigenvalues of complex skew-Hamiltonian/Hamiltonian pencil (panel version)
 *
 * Moves eigenvalues with strictly negative real parts to the leading subpencil,
 * while keeping the triangular form. Panel version with blocked updates for large N.
 *
 * On entry:
 *     S = [A  D]    H = [B  F]
 *         [0  A']       [0 -B']
 *
 * where A and B are upper triangular.
 *
 * @param[in] compq 'N'=don't compute Q, 'I'=initialize Q to identity, 'U'=update Q
 * @param[in] n Order of the pencil (n >= 0, must be even)
 * @param[in,out] a N/2-by-N/2 upper triangular matrix A
 * @param[in] lda Leading dimension of A (lda >= max(1, n/2))
 * @param[in,out] d N/2-by-N/2 upper triangular skew-Hermitian matrix D
 * @param[in] ldd Leading dimension of D (ldd >= max(1, n/2))
 * @param[in,out] b N/2-by-N/2 upper triangular matrix B
 * @param[in] ldb Leading dimension of B (ldb >= max(1, n/2))
 * @param[in,out] f N/2-by-N/2 upper triangular Hermitian matrix F
 * @param[in] ldf Leading dimension of F (ldf >= max(1, n/2))
 * @param[in,out] q N-by-N unitary transformation matrix
 * @param[in] ldq Leading dimension of Q (ldq >= 1 if compq='N', ldq >= n otherwise)
 * @param[out] neig Number of eigenvalues with strictly negative real part
 * @param[in] tol Tolerance for eigenvalue sign (<=0 for default)
 * @param[out] dwork Real workspace, dimension (n/2)
 * @param[out] zwork Complex workspace, dimension (n/2)
 * @param[in,out] info On entry: <=0 for auto block size, >0 for user block size.
 *                     On exit: 0=success, <0=-i-th argument invalid
 */
void mb3jzp(const char *compq, i32 n, c128 *a, i32 lda, c128 *d, i32 ldd,
            c128 *b, i32 ldb, c128 *f, i32 ldf, c128 *q, i32 ldq, i32 *neig,
            f64 tol, f64 *dwork, c128 *zwork, i32 *info);

/**
 * @brief Compute eigenvalues and deflating subspace of complex skew-Hamiltonian/Hamiltonian pencil
 *
 * Computes eigenvalues of a complex N-by-N skew-Hamiltonian/Hamiltonian pencil aS - bH, with
 *
 *       (  A  D  )         (  B  F  )
 *   S = (        ) and H = (        )
 *       (  E  A' )         (  G -B' )
 *
 * where A' denotes conjugate transpose. The routine:
 * 1. Embeds the complex pencil into a real skew-Hamiltonian/skew-Hamiltonian pencil
 * 2. Applies MB04FP to compute the structured Schur form
 * 3. Applies MB3JZP to reorder eigenvalues with negative real parts to top
 * 4. Optionally computes the right deflating subspace via QR or SVD
 *
 * @param[in] compq 'N': compute eigenvalues only, 'C': also compute deflating subspace
 * @param[in] orth Orthogonalization method (if compq='C'): 'P'=QR with pivoting, 'S'=SVD
 * @param[in] n Order of the pencil (n >= 0, must be even)
 * @param[in,out] a Complex array (lda, n). On entry: N/2-by-N/2 matrix A.
 *                  On exit if compq='C': upper triangular BA in (lda, n)
 * @param[in] lda Leading dimension of A (lda >= max(1, n))
 * @param[in,out] de Complex array (ldde, n). On entry: E in lower triangular part,
 *                   D in upper triangular part of columns 2 to n/2+1.
 *                   On exit if compq='C': skew-Hermitian BD
 * @param[in] ldde Leading dimension of DE (ldde >= max(1, n))
 * @param[in,out] b Complex array (ldb, n). On entry: N/2-by-N/2 matrix B.
 *                  On exit if compq='C': upper triangular BB
 * @param[in] ldb Leading dimension of B (ldb >= max(1, n))
 * @param[in,out] fg Complex array (ldfg, n). On entry: G in lower triangular part,
 *                   F in upper triangular part of columns 2 to n/2+1.
 *                   On exit if compq='C': Hermitian BF
 * @param[in] ldfg Leading dimension of FG (ldfg >= max(1, n))
 * @param[out] neig Number of eigenvalues with strictly negative real part (if compq='C')
 * @param[out] q Complex array (ldq, 2*n). If compq='C': leading N-by-NEIG part contains
 *               orthonormal basis of right deflating subspace
 * @param[in] ldq Leading dimension of Q (ldq >= 1 if compq='N', ldq >= 2*n if compq='C')
 * @param[out] alphar Real parts of eigenvalue numerators, dimension (n)
 * @param[out] alphai Imaginary parts of eigenvalue numerators, dimension (n)
 * @param[out] beta Eigenvalue denominators, dimension (n). lambda = alpha/beta
 * @param[out] iwork Integer workspace, dimension (n+1)
 * @param[out] dwork Real workspace. dwork[0]=optimal ldwork on exit
 * @param[in] ldwork Real workspace size. ldwork >= 4*n*n+2*n+max(3,n) if compq='N',
 *                   ldwork >= 11*n*n+2*n if compq='C'. Use -1 for query.
 * @param[out] zwork Complex workspace. zwork[0]=optimal lzwork on exit
 * @param[in] lzwork Complex workspace size. lzwork >= 1 if compq='N',
 *                   lzwork >= 8*n+4 if compq='C'. Use -1 for query.
 * @param[out] bwork Logical workspace, dimension (n-1) if compq='C', unused otherwise
 * @param[in,out] info On entry: <=0 for auto block size, >0 for user block size.
 *                     On exit: 0=success, <0=argument error, 1=MB04FP QZ failed,
 *                     2=ZHGEQZ failed, 3=ZGESVD failed, 4=numerically singular pencil
 */
void mb3lzp(const char *compq, const char *orth, i32 n,
            c128 *a, i32 lda, c128 *de, i32 ldde, c128 *b, i32 ldb,
            c128 *fg, i32 ldfg, i32 *neig, c128 *q, i32 ldq,
            f64 *alphar, f64 *alphai, f64 *beta,
            i32 *iwork, f64 *dwork, i32 ldwork, c128 *zwork, i32 lzwork,
            bool *bwork, i32 *info);

/**
 * @brief MB03LF - Eigenvalues and deflating subspace of skew-Hamiltonian/Hamiltonian pencil
 *
 * Computes the relevant eigenvalues of a real N-by-N skew-Hamiltonian/Hamiltonian
 * pencil aS - bH, with
 *     S = T Z = J Z' J' Z  and  H = [B  F; G -B'], J = [0 I; -I 0]
 *
 * Optionally computes orthogonal basis of the right deflating subspace
 * (COMPQ='C') and companion subspace (COMPU='C') corresponding to eigenvalues
 * with strictly negative real part.
 *
 * @param[in] compq 'N': no deflating subspace, 'C': compute and store in Q
 * @param[in] compu 'N': no companion subspace, 'C': compute and store in U
 * @param[in] orth Orthogonalization method: 'P'=QR with pivoting, 'S'=SVD
 * @param[in] n Order of the pencil (n >= 0, must be even)
 * @param[in,out] z Array (ldz, n). On entry: factor Z in S = J Z' J' Z.
 *                  On exit: transformed triangular matrix Z11
 * @param[in] ldz Leading dimension of Z (ldz >= max(1, n))
 * @param[in] b Array (ldb, n/2). The N/2-by-N/2 matrix B
 * @param[in] ldb Leading dimension of B (ldb >= max(1, n/2))
 * @param[in] fg Array (ldfg, n/2+1). Lower triangular contains G,
 *               upper triangular of columns 2:n/2+1 contains F
 * @param[in] ldfg Leading dimension of FG (ldfg >= max(1, n/2))
 * @param[out] neig Number of eigenvalues with strictly negative real part
 * @param[out] q Array (ldq, 2*n). If compq='C': orthogonal basis of right
 *               deflating subspace in first N-by-NEIG part
 * @param[in] ldq Leading dimension of Q (ldq >= 1 if compq='N', ldq >= 2*n if compq='C')
 * @param[out] u Array (ldu, 2*n). If compu='C': orthogonal basis of companion
 *               subspace in first N-by-NEIG part
 * @param[in] ldu Leading dimension of U (ldu >= 1 if compu='N', ldu >= n if compu='C')
 * @param[out] alphar Real parts of eigenvalue numerators, dimension (n/2)
 * @param[out] alphai Imaginary parts of eigenvalue numerators, dimension (n/2)
 * @param[out] beta Eigenvalue denominators, dimension (n/2). lambda = alpha/beta
 * @param[out] iwork Integer workspace, dimension (liwork)
 * @param[in] liwork Integer workspace size. liwork >= n+18 if no subspace,
 *                   liwork >= max(2*n+1, 48) otherwise
 * @param[out] dwork Real workspace. dwork[0]=optimal ldwork on exit
 * @param[in] ldwork Real workspace size. Use -1 for query.
 * @param[out] bwork Logical workspace, dimension (n/2)
 * @param[out] iwarn 0=no warning, 1=some eigenvalues might be unreliable
 * @param[out] info 0=success, <0=argument error, 1=QZ failed, 2=QZ iteration failed,
 *                  3=numerically singular matrix, 4=SVD failed
 */
void mb03lf(const char *compq, const char *compu, const char *orth,
            i32 n, f64 *z, i32 ldz, const f64 *b, i32 ldb,
            const f64 *fg, i32 ldfg, i32 *neig, f64 *q, i32 ldq,
            f64 *u, i32 ldu, f64 *alphar, f64 *alphai, f64 *beta,
            i32 *iwork, i32 liwork, f64 *dwork, i32 ldwork,
            bool *bwork, i32 *iwarn, i32 *info);

/**
 * @brief MB03WA - Swap adjacent diagonal blocks in periodic real Schur form
 *
 * Swaps adjacent diagonal blocks A11*B11 and A22*B22 of size 1-by-1 or 2-by-2
 * in an upper (quasi) triangular matrix product A*B by an orthogonal
 * equivalence transformation.
 *
 * (A, B) must be in periodic real Schur canonical form, i.e., A is block upper
 * triangular with 1-by-1 and 2-by-2 diagonal blocks, and B is upper triangular.
 *
 * Optionally, the matrices Q and Z of generalized Schur vectors are updated:
 *     Q(in) * A(in) * Z(in)' = Q(out) * A(out) * Z(out)'
 *     Z(in) * B(in) * Q(in)' = Z(out) * B(out) * Q(out)'
 *
 * @param[in] wantq true = update Q, false = Q not required
 * @param[in] wantz true = update Z, false = Z not required
 * @param[in] n1 Order of first block A11*B11 (0, 1, or 2)
 * @param[in] n2 Order of second block A22*B22 (0, 1, or 2)
 * @param[in,out] a Array (lda, n1+n2). On entry: matrix A. On exit: reordered A
 * @param[in] lda Leading dimension of a (lda >= max(1, n1+n2))
 * @param[in,out] b Array (ldb, n1+n2). On entry: matrix B. On exit: reordered B
 * @param[in] ldb Leading dimension of b (ldb >= max(1, n1+n2))
 * @param[in,out] q Array (ldq, n1+n2). If wantq: updated orthogonal matrix Q
 * @param[in] ldq Leading dimension of q (ldq >= 1, or >= n1+n2 if wantq)
 * @param[in,out] z Array (ldz, n1+n2). If wantz: updated orthogonal matrix Z
 * @param[in] ldz Leading dimension of z (ldz >= 1, or >= n1+n2 if wantz)
 * @param[out] info 0=success, 1=swap rejected (would be too far from Schur form)
 */
void mb03wa(bool wantq, bool wantz, i32 n1, i32 n2,
            f64 *a, i32 lda, f64 *b, i32 ldb,
            f64 *q, i32 ldq, f64 *z, i32 ldz, i32 *info);

/**
 * @brief MB03ZA - Reorder eigenvalues in periodic Schur form and compute stable subspace
 *
 * Computes orthogonal matrices Ur and Vr so that:
 *     Vr' * A * Ur = [A11  A12]    Ur' * B * Vr = [B11  B12]
 *                    [0    A22]                   [0    B22]
 * is in periodic Schur form with eigenvalues of A11*B11 forming selected cluster.
 *
 * Also computes orthogonal W transforming [0, -A11; B11, 0] to block triangular
 * form with eigenvalues of R11 having positive real part.
 *
 * @param[in] compc 'U'=update C, 'N'=don't update
 * @param[in] compu 'U'=update U1,U2, 'N'=don't update
 * @param[in] compv 'U'=update V1,V2, 'N'=don't update
 * @param[in] compw 'N'=W not needed, 'I'=init W to identity, 'V'=accumulate
 * @param[in] which 'A'=all eigenvalues, 'S'=selected by SELECT
 * @param[in] select Array (n). If WHICH='S', specifies selected eigenvalues
 * @param[in] n Order of A and B
 * @param[in,out] a Array (lda,n). Upper quasi-triangular A. Exit: R22 in M-by-M
 * @param[in] lda Leading dimension of a
 * @param[in,out] b Array (ldb,n). Upper triangular B. Exit: overwritten
 * @param[in] ldb Leading dimension of b
 * @param[in,out] c Array (ldc,n). General matrix C. Exit: Ur'*C*Vr if compc='U'
 * @param[in] ldc Leading dimension of c
 * @param[in,out] u1 Array (ldu1,n). Block of orthogonal symplectic U
 * @param[in] ldu1 Leading dimension of u1
 * @param[in,out] u2 Array (ldu2,n). Block of orthogonal symplectic U
 * @param[in] ldu2 Leading dimension of u2
 * @param[in,out] v1 Array (ldv1,n). Block of orthogonal symplectic V
 * @param[in] ldv1 Leading dimension of v1
 * @param[in,out] v2 Array (ldv2,n). Block of orthogonal symplectic V
 * @param[in] ldv2 Leading dimension of v2
 * @param[in,out] w Array (ldw,2*m). Orthogonal transformation matrix
 * @param[in] ldw Leading dimension of w
 * @param[out] wr Array (m). Real parts of eigenvalues of R11
 * @param[out] wi Array (m). Imaginary parts of eigenvalues of R11
 * @param[out] m Number of selected eigenvalues
 * @param[out] dwork Workspace array
 * @param[in] ldwork Workspace size >= max(1, 4*n, 8*m)
 * @param[out] info 0=success, 1=reorder A*B failed, 2=reorder submatrix failed,
 *                  3=QR failed, 4=eigenvalue condition violated
 */
void mb03za(const char *compc, const char *compu, const char *compv,
            const char *compw, const char *which, const i32 *select,
            i32 n, f64 *a, i32 lda, f64 *b, i32 ldb,
            f64 *c, i32 ldc, f64 *u1, i32 ldu1, f64 *u2, i32 ldu2,
            f64 *v1, i32 ldv1, f64 *v2, i32 ldv2, f64 *w, i32 ldw,
            f64 *wr, f64 *wi, i32 *m, f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief MB03ZD - Compute stable/unstable invariant subspaces for Hamiltonian matrix.
 *
 * Computes the stable and unstable invariant subspaces for a Hamiltonian matrix
 * with no eigenvalues on the imaginary axis, using the output of MB03XD.
 *
 * @param[in] which 'A'=select all n eigenvalues, 'S'=select cluster by SELECT
 * @param[in] meth Method if WHICH='A': 'S'=n vectors, 'L'=2*n vectors, 'Q'/'R'=quick
 * @param[in] stab 'S'=stable subspace, 'U'=unstable subspace, 'B'=both
 * @param[in] balanc 'N'=none, 'P'=permutation, 'S'=scaling, 'B'=both
 * @param[in] ortbal 'B'=balance before orthogonalization, 'A'=after
 * @param[in] select Array (n). If WHICH='S', specifies selected eigenvalues
 * @param[in] n Order of matrices S, T, G
 * @param[in] mm Number of columns in US and/or UU
 * @param[in] ilo Lower index from MB03XD if BALANC!='N'
 * @param[in] scale Array (n). Permutation/scaling factors from MB03XD
 * @param[in,out] s Array (lds,n). Matrix S in real Schur form. Overwritten.
 * @param[in] lds Leading dimension of s
 * @param[in,out] t Array (ldt,n). Upper triangular T. Overwritten.
 * @param[in] ldt Leading dimension of t
 * @param[in,out] g Array (ldg,n). General matrix G if METH='L'/'R'. Overwritten.
 * @param[in] ldg Leading dimension of g
 * @param[in,out] u1 Array (ldu1,n). (1,1) block of orthogonal symplectic U
 * @param[in] ldu1 Leading dimension of u1
 * @param[in,out] u2 Array (ldu2,n). (2,1) block of orthogonal symplectic U
 * @param[in] ldu2 Leading dimension of u2
 * @param[in,out] v1 Array (ldv1,n). (1,1) block of orthogonal symplectic V
 * @param[in] ldv1 Leading dimension of v1
 * @param[in,out] v2 Array (ldv2,n). (2,1) block of orthogonal symplectic V
 * @param[in] ldv2 Leading dimension of v2
 * @param[out] m Number of selected eigenvalues
 * @param[out] wr Array (m). Real parts of selected eigenvalues
 * @param[out] wi Array (m). Imaginary parts of selected eigenvalues
 * @param[out] us Array (ldus,mm). Stable invariant subspace basis if STAB='S'/'B'
 * @param[in] ldus Leading dimension of us
 * @param[out] uu Array (lduu,mm). Unstable invariant subspace basis if STAB='U'/'B'
 * @param[in] lduu Leading dimension of uu
 * @param[out] lwork Logical workspace (2*n) if WHICH='A' and METH='L'/'R'
 * @param[out] iwork Integer workspace
 * @param[out] dwork Double workspace
 * @param[in] ldwork Workspace size. -1 for query.
 * @param[out] info 0=success, 1=eigenvalues on imaginary axis, 2=reorder failed,
 *                  3=QR failed, 4=MB03TD failed, 5/6=subspace inaccurate
 */
void mb03zd(const char *which, const char *meth, const char *stab,
            const char *balanc, const char *ortbal, const i32 *select,
            i32 n, i32 mm, i32 ilo, const f64 *scale,
            f64 *s, i32 lds, f64 *t, i32 ldt, f64 *g, i32 ldg,
            f64 *u1, i32 ldu1, f64 *u2, i32 ldu2,
            f64 *v1, i32 ldv1, f64 *v2, i32 ldv2,
            i32 *m, f64 *wr, f64 *wi,
            f64 *us, i32 ldus, f64 *uu, i32 lduu,
            bool *lwork, i32 *iwork, f64 *dwork, i32 ldwork, i32 *info);

#ifdef __cplusplus
}
#endif

#endif /* SLICOT_MB03_H */
