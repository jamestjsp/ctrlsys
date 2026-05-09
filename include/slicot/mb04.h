/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#ifndef SLICOT_MB04_H
#define SLICOT_MB04_H

#include "../slicot_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief RQ factorization of special structured block matrix.
 *
 * Calculates an RQ factorization of the first block row and applies
 * the orthogonal transformations (from the right) to the second block row:
 *
 *     [ A   R ]        [ 0   R_new ]
 *     [       ] * Q' = [           ]
 *     [ C   B ]        [ C_new B_new ]
 *
 * where R and R_new are upper triangular. Matrix A can be full or
 * upper trapezoidal/triangular.
 *
 * @param[in] uplo 'U' = A is upper trapezoidal, 'F' = A is full
 * @param[in] n Order of matrices R and R_new, n >= 0
 * @param[in] m Number of rows of matrices B and C, m >= 0
 * @param[in] p Number of columns of matrices A and C, p >= 0
 * @param[in,out] r On entry: n-by-n upper triangular matrix R
 *                  On exit: n-by-n upper triangular matrix R_new
 * @param[in] ldr Leading dimension of R, ldr >= max(1,n)
 * @param[in,out] a On entry: n-by-p matrix A (full or upper trapezoidal)
 *                  On exit: Householder vectors in corresponding positions
 * @param[in] lda Leading dimension of A, lda >= max(1,n)
 * @param[in,out] b On entry: m-by-n matrix B
 *                  On exit: transformed matrix B_new
 * @param[in] ldb Leading dimension of B, ldb >= max(1,m)
 * @param[in,out] c On entry: m-by-p matrix C
 *                  On exit: transformed matrix C_new
 * @param[in] ldc Leading dimension of C, ldc >= max(1,m)
 * @param[out] tau Scalar factors of elementary reflectors, dimension (n)
 * @param[out] dwork Workspace, dimension max(n-1, m)
 */
void SLC_MB04ND(const char* uplo, i32 n, i32 m, i32 p,
                f64* r, i32 ldr, f64* a, i32 lda,
                f64* b, i32 ldb, f64* c, i32 ldc,
                f64* tau, f64* dwork);

/**
 * @brief Apply Householder reflector H to matrix [A B] from the right.
 *
 * Applies elementary reflector H to m-by-(n+1) matrix C = [A B],
 * where A has one column:
 *
 *     H = I - tau * u * u',  u = [1; v]
 *
 * Computes C := C * H.
 *
 * Uses inline code for order < 11, BLAS for larger orders.
 *
 * @param[in] m Number of rows of matrices A and B, m >= 0
 * @param[in] n Number of columns of matrix B, n >= 0
 * @param[in] v Householder vector v, dimension (1+(n-1)*abs(incv))
 * @param[in] incv Increment between elements of v, incv != 0
 * @param[in] tau Scalar factor tau (if tau=0, H is identity)
 * @param[in,out] a On entry: m-by-1 matrix A
 *                  On exit: updated first column of C*H
 * @param[in] lda Leading dimension of A, lda >= max(1,m)
 * @param[in,out] b On entry: m-by-n matrix B
 *                  On exit: updated last n columns of C*H
 * @param[in] ldb Leading dimension of B, ldb >= max(1,m)
 * @param[out] dwork Workspace of dimension m (not referenced if n+1 < 11)
 *
 * @note Based on LAPACK's DLARFX and DLATZM with special structure optimization.
 */
void SLC_MB04NY(i32 m, i32 n, const f64* v, i32 incv, f64 tau,
                f64* a, i32 lda, f64* b, i32 ldb, f64* dwork);

/**
 * @brief Apply Householder reflector H to matrix [A; B] from the left.
 *
 * Applies elementary reflector H to (m+1)-by-n matrix C = [A; B],
 * where A has one row:
 *
 *     H = I - tau * u * u',  u = [1; v]
 *
 * Computes C := H * C.
 *
 * Uses inline code for order < 11, BLAS for larger orders.
 *
 * @param[in] m Number of rows of matrix B, m >= 0
 * @param[in] n Number of columns, n >= 0
 * @param[in] v Householder vector v of dimension m
 * @param[in] tau Scalar factor tau (if tau=0, H is identity)
 * @param[in,out] a On entry: 1-by-n matrix A
 *                  On exit: updated first row of H*C
 * @param[in] lda Leading dimension of A, lda >= 1
 * @param[in,out] b On entry: m-by-n matrix B
 *                  On exit: updated last m rows of H*C
 * @param[in] ldb Leading dimension of B, ldb >= max(1,m)
 * @param[out] dwork Workspace of dimension n (not referenced if m+1 < 11)
 *
 * @note Based on LAPACK's DLARFX and DLATZM with special structure optimization.
 */
void SLC_MB04OY(i32 m, i32 n, const f64* v, f64 tau,
                f64* a, i32 lda, f64* b, i32 ldb, f64* dwork);

/**
 * @brief Apply elementary reflector to matrix from left or right.
 *
 * Applies real elementary reflector H to m-by-n matrix C, from left or right:
 *
 *     H * C  (side='L')
 *     C * H  (side='R')
 *
 * where:
 *     H = I - tau * u * u',  u = [1; v]
 *
 * tau is scalar, v is (m-1)-vector if side='L', (n-1)-vector if side='R'.
 * If tau=0, H is identity matrix.
 *
 * Uses inline code if H has order < 11 for efficiency.
 *
 * @param[in] side 'L' = apply H from left (H*C), 'R' = from right (C*H)
 * @param[in] m Number of rows of matrix C, m >= 0
 * @param[in] n Number of columns of matrix C, n >= 0
 * @param[in] v Vector in reflector representation, dimension (m-1) if side='L',
 *              (n-1) if side='R'
 * @param[in] tau Scalar factor of elementary reflector H
 * @param[in,out] c On entry: m-by-n matrix C
 *                  On exit: H*C if side='L', C*H if side='R'
 * @param[in] ldc Leading dimension of C, ldc >= max(1,m)
 * @param[out] dwork Workspace, dimension (n) if side='L', (m) if side='R'.
 *                   Not referenced if H has order < 11.
 *
 * @note Based on LAPACK's DLARFX with special structure optimization.
 */
void SLC_MB04PY(char side, i32 m, i32 n, const f64* v, f64 tau,
                f64* c, i32 ldc, f64* dwork);

/**
 * @brief Balance a real Hamiltonian matrix.
 *
 * Balances a real Hamiltonian matrix H = [A, G; Q, -A'] where A is NxN
 * and G, Q are symmetric NxN matrices. Involves permuting to isolate
 * eigenvalues and diagonal similarity transformations.
 *
 * @param[in] job 'N' = none, 'P' = permute only, 'S' = scale only, 'B' = both
 * @param[in] n Order of matrix A (N >= 0)
 * @param[in,out] a N-by-N matrix A, balanced on exit
 * @param[in] lda Leading dimension of A
 * @param[in,out] qg N-by-(N+1) matrix with lower tri of Q and upper tri of G
 * @param[in] ldqg Leading dimension of QG
 * @param[out] ilo Number of deflated eigenvalues + 1
 * @param[out] scale Permutation and scaling factors, dimension N
 * @param[out] info Exit code (0 = success, <0 = invalid parameter -i)
 */
void mb04dd(const char *job, i32 n, f64 *a, i32 lda,
            f64 *qg, i32 ldqg, i32 *ilo, f64 *scale, i32 *info);

/**
 * @brief QR factorization of matrix with lower-left zero triangle
 *
 * Computes QR factorization A = Q*R of n-by-m matrix A having p-by-min(p,m)
 * zero triangle in lower left corner. Optionally applies Q' to n-by-l matrix B.
 * Exploits structure for efficiency (useful in Kalman filtering).
 *
 * Example structure (n=8, m=7, p=2):
 *     [ x x x x x x x ]
 *     [ x x x x x x x ]
 *     [ x x x x x x x ]
 *     [ x x x x x x x ]
 * A = [ x x x x x x x ]
 *     [ x x x x x x x ]
 *     [ 0 x x x x x x ]  <- p rows with
 *     [ 0 0 x x x x x ]  <- lower-left zeros
 *
 * @param[in] n Number of rows of A (n >= 0)
 * @param[in] m Number of columns of A (m >= 0)
 * @param[in] p Order of zero triangle (p >= 0)
 * @param[in] l Number of columns of B (l >= 0)
 * @param[in,out] a Matrix A, dimension (lda,m)
 *                  In: n-by-m matrix with p-by-min(p,m) zero lower-left triangle
 *                  Out: min(n,m)-by-m upper trapezoidal R, Householder vectors below diagonal
 * @param[in] lda Leading dimension of a (lda >= max(1,n))
 * @param[in,out] b Matrix B, dimension (ldb,l)
 *                  In: n-by-l matrix B
 *                  Out: Q'*B
 *                  Not referenced if l=0
 * @param[in] ldb Leading dimension of b (ldb >= max(1,n) if l>0, ldb >= 1 if l=0)
 * @param[out] tau Householder scalar factors, dimension (min(n,m))
 * @param[out] dwork Workspace, dimension (ldwork)
 *                   dwork[0] returns optimal ldwork
 * @param[in] ldwork Workspace size (ldwork >= max(1,m-1,m-p,l))
 *                   If ldwork=-1, workspace query
 * @param[out] info Exit code (0=success, <0=invalid parameter)
 */
void mb04id(i32 n, i32 m, i32 p, i32 l, f64 *a, i32 lda, f64 *b, i32 ldb,
            f64 *tau, f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Apply orthogonal transformations from MB04ID to matrix C.
 *
 * Overwrites real n-by-m matrix C with Q'*C, Q*C, C*Q', or C*Q, where
 * Q is orthogonal matrix defined as product of k elementary reflectors:
 *
 *     Q = H(1) H(2) ... H(k)
 *
 * as returned by MB04ID. Q is order n if SIDE='L', order m if SIDE='R'.
 * Reflectors stored in special format for lower-left zero triangle structure.
 *
 * @param[in] side 'L' = apply Q or Q' from left, 'R' = from right
 * @param[in] trans 'N' = apply Q (no transpose), 'T' = apply Q' (transpose)
 * @param[in] n Number of rows of matrix C, n >= 0
 * @param[in] m Number of columns of matrix C, m >= 0
 * @param[in] k Number of elementary reflectors, constraints:
 *              n >= k >= 0 if SIDE='L'
 *              m >= k >= 0 if SIDE='R'
 * @param[in] p Order of zero triangle (rows of zero trapezoid), p >= 0
 * @param[in,out] a Reflector storage, dimension (lda,k)
 *                  Row i+1:min(n,n-p-1+i) of column i contains H(i)
 *                  Modified but restored on exit
 * @param[in] lda Leading dimension: lda >= max(1,n) if SIDE='L',
 *                                   lda >= max(1,m) if SIDE='R'
 * @param[in] tau Reflector scalar factors, dimension (k)
 * @param[in,out] c On entry: n-by-m matrix C
 *                  On exit: transformed matrix
 * @param[in] ldc Leading dimension, ldc >= max(1,n)
 * @param[out] dwork Workspace, dimension (ldwork)
 *                   dwork[0] returns optimal ldwork on exit
 * @param[in] ldwork Workspace size:
 *                   ldwork >= max(1,m) if SIDE='L'
 *                   ldwork >= max(1,n) if SIDE='R'
 * @param[out] info Exit code: 0=success, <0=invalid parameter
 */
void mb04iy(
    const char* side,
    const char* trans,
    const i32 n,
    const i32 m,
    const i32 k,
    const i32 p,
    f64* a,
    const i32 lda,
    const f64* tau,
    f64* c,
    const i32 ldc,
    f64* dwork,
    const i32 ldwork,
    i32* info
);

/**
 * @brief QR factorization of special structured block matrix.
 *
 * Computes QR factorization of first block column and applies orthogonal
 * transformations to second block column:
 *         [[R], [A B]] -> Q' * [[R], [A B]] = [[R_bar C], [0 D]]
 * where R and R_bar are upper triangular.
 *
 * @param[in] uplo Indicates structure of A:
 *                 'U' = upper trapezoidal/triangular
 *                 'F' = full matrix
 * @param[in] n Order of matrices R and R_bar (n >= 0)
 * @param[in] m Number of columns of B, C, D (m >= 0)
 * @param[in] p Number of rows of A, B, D (p >= 0)
 * @param[in,out] r Matrix R, dimension (ldr,n)
 *                  In: n-by-n upper triangular R
 *                  Out: n-by-n upper triangular R_bar
 * @param[in] ldr Leading dimension of r (ldr >= max(1,n))
 * @param[in,out] a Matrix A, dimension (lda,n)
 *                  In: p-by-n matrix (full or upper trapezoidal)
 *                  Out: Householder vectors v_i
 * @param[in] lda Leading dimension of a (lda >= max(1,p))
 * @param[in,out] b Matrix B, dimension (ldb,m)
 *                  In: p-by-m matrix B
 *                  Out: p-by-m matrix D
 * @param[in] ldb Leading dimension of b (ldb >= max(1,p))
 * @param[out] c Matrix C, dimension (ldc,m)
 *               n-by-m matrix C
 * @param[in] ldc Leading dimension of c (ldc >= max(1,n))
 * @param[out] tau Scalar factors of elementary reflectors, dimension (n)
 * @param[out] dwork Workspace, dimension (n)
 */
void mb04kd(const char uplo, i32 n, i32 m, i32 p,
            f64 *r, i32 ldr,
            f64 *a, i32 lda,
            f64 *b, i32 ldb,
            f64 *c, i32 ldc,
            f64 *tau,
            f64 *dwork);

/**
 * @brief LQ factorization of structured block matrix.
 *
 * Computes LQ factorization of first block row and applies orthogonal
 * transformations to second block row:
 *         [[L A], [0 B]] * Q = [[L_bar 0], [C D]]
 * where L and L_bar are lower triangular.
 *
 * This computation is useful for Kalman filter square-root covariance updates.
 *
 * @param[in] uplo Indicates structure of A:
 *                 'L' = lower trapezoidal/triangular
 *                 'F' = full matrix
 * @param[in] n Order of matrices L and L_bar (n >= 0)
 * @param[in] m Number of columns of A, B, D (m >= 0)
 * @param[in] p Number of rows of B, C, D (p >= 0)
 * @param[in,out] l Matrix L, dimension (ldl,n)
 *                  In: n-by-n lower triangular L
 *                  Out: n-by-n lower triangular L_bar
 * @param[in] ldl Leading dimension of l (ldl >= max(1,n))
 * @param[in,out] a Matrix A, dimension (lda,m)
 *                  In: n-by-m matrix (full or lower trapezoidal)
 *                  Out: Householder vectors v_i (stored in rows)
 * @param[in] lda Leading dimension of a (lda >= max(1,n))
 * @param[in,out] b Matrix B, dimension (ldb,m)
 *                  In: p-by-m matrix B
 *                  Out: p-by-m matrix D
 * @param[in] ldb Leading dimension of b (ldb >= max(1,p))
 * @param[out] c Matrix C, dimension (ldc,n)
 *               p-by-n matrix C
 * @param[in] ldc Leading dimension of c (ldc >= max(1,p))
 * @param[out] tau Scalar factors of elementary reflectors, dimension (n)
 * @param[out] dwork Workspace, dimension (n)
 */
void mb04ld(const char uplo, i32 n, i32 m, i32 p,
            f64 *l, i32 ldl,
            f64 *a, i32 lda,
            f64 *b, i32 ldb,
            f64 *c, i32 ldc,
            f64 *tau,
            f64 *dwork);

/**
 * @brief QR factorization of structured block matrix.
 *
 * Calculates QR factorization of first block column and applies orthogonal
 * transformations to second block column:
 *         [ R   B ]   [ R_   B_ ]
 *    Q' * [       ] = [         ]
 *         [ A   C ]   [ 0    C_ ]
 *
 * where R and R_ are upper triangular, A can be full or upper trapezoidal.
 *
 * @param[in] uplo 'U' for A upper trapezoidal/triangular, 'F' for A full
 * @param[in] n Order of R (n >= 0)
 * @param[in] m Number of columns in B, C (m >= 0)
 * @param[in] p Number of rows in A, C (p >= 0)
 * @param[in,out] r n-by-n upper triangular matrix R, dimension (ldr,n)
 * @param[in] ldr Leading dimension of R (ldr >= max(1,n))
 * @param[in,out] a p-by-n matrix A, dimension (lda,n)
 *                  On exit: Householder vectors
 * @param[in] lda Leading dimension of A (lda >= max(1,p))
 * @param[in,out] b n-by-m matrix B, dimension (ldb,m)
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in,out] c p-by-m matrix C, dimension (ldc,m)
 * @param[in] ldc Leading dimension of C (ldc >= max(1,p))
 * @param[out] tau Householder scalars, dimension (n)
 * @param[out] dwork Workspace, dimension (max(n-1,m))
 *
 * @note Algorithm is backward stable
 */
void mb04od(const char* uplo, const i32 n, const i32 m, const i32 p,
            f64* r, const i32 ldr, f64* a, const i32 lda,
            f64* b, const i32 ldb, f64* c, const i32 ldc,
            f64* tau, f64* dwork);

/**
 * @brief Performs a QR factorization update.
 *
 * MB04OW performs the QR factorization
 *
 *      ( U  ) = Q*( R ),  where  U = ( U1  U2 ),  R = ( R1  R2 ),
 *      ( x' )     ( 0 )              ( 0   T  )       ( 0   R3 )
 *
 * where U and R are (m+n)-by-(m+n) upper triangular matrices, x is
 * an m+n element vector, U1 is m-by-m, T is n-by-n, stored
 * separately, and Q is an (m+n+1)-by-(m+n+1) orthogonal matrix.
 *
 * The transformations performed are also applied to the (m+n+1)-by-p
 * matrix ( B' C' d )' (' denotes transposition), where B, C, and d'
 * are m-by-p, n-by-p, and 1-by-p matrices, respectively.
 *
 * @param[in] m The number of rows of the matrix ( U1  U2 ). M >= 0.
 * @param[in] n The order of the matrix T. N >= 0.
 * @param[in] p The number of columns of the matrices B and C. P >= 0.
 * @param[in,out] a Array of dimension (LDA, N+M). On entry, the leading M-by-(M+N) 
 *                  upper trapezoidal part contains ( U1 U2 ). On exit, ( R1 R2 ).
 * @param[in] lda The leading dimension of the array A. LDA >= max(1,M).
 * @param[in,out] t Array of dimension (LDT, N). On entry, the leading N-by-N 
 *                  upper triangular part contains T. On exit, R3.
 * @param[in] ldt The leading dimension of the array T. LDT >= max(1,N).
 * @param[in,out] x Array of dimension (1+(M+N-1)*INCX). On entry, the vector x. 
 *                  On exit, the content is changed (destroyed).
 * @param[in] incx The increment for the elements of X. INCX > 0.
 * @param[in,out] b Array of dimension (LDB, P). On entry, B. On exit, transformed B.
 * @param[in] ldb The leading dimension of the array B. LDB >= max(1,M) if P > 0.
 * @param[in,out] c Array of dimension (LDC, P). On entry, C. On exit, transformed C.
 * @param[in] ldc The leading dimension of the array C. LDC >= max(1,N) if P > 0.
 * @param[in,out] d Array of dimension (1+(P-1)*INCD). On entry, the vector d. 
 *                  On exit, transformed d.
 * @param[in] incd The increment for the elements of D. INCD > 0.
 */
void mb04ow(i32 m, i32 n, i32 p, f64 *a, i32 lda, f64 *t, i32 ldt, 
            f64 *x, i32 incx, f64 *b, i32 ldb, f64 *c, i32 ldc, 
            f64 *d, i32 incd);

/**
 * @brief QR factorization update for rank-one modification.
 *
 * Performs QR factorization of augmented matrix:
 *     ( U  )     ( R )
 *     ( x' ) = Q ( 0 )
 *
 * where U and R are n-by-n upper triangular matrices, x is an n-element
 * vector, and Q is an (n+1)-by-(n+1) orthogonal matrix.
 *
 * U must be supplied in the n-by-n upper triangular part of A and is
 * overwritten by R.
 *
 * @param[in] n Order of the matrix A (N >= 0)
 * @param[in,out] a On entry: upper triangular matrix U.
 *                  On exit: upper triangular matrix R. Dimension (LDA,N)
 * @param[in] lda Leading dimension of A (LDA >= max(1,N))
 * @param[in,out] x On entry: vector x. On exit: modified. Dimension (1+(N-1)*INCX)
 * @param[in] incx Increment for elements of X (INCX > 0)
 */
void mb04ox(i32 n, f64* a, i32 lda, f64* x, i32 incx);

/**
 * @brief Apply Householder transformation to block matrix rows.
 *
 * Inline code for applying Householder transformation H = I - tau*u*u' where
 * u = [1; v] to block matrix rows, exploiting structure in MB04OD.
 *
 * @param[in] m Length of vector v (m >= 0)
 * @param[in] n Number of columns to update (n >= 0)
 * @param[in] v Householder vector v, dimension (m)
 * @param[in] tau Householder scalar
 * @param[in,out] c1 First row to update, dimension (n)
 * @param[in] ldc1 Leading dimension of C1
 * @param[in,out] c2 Remaining m rows to update, dimension (ldc2,n)
 * @param[in] ldc2 Leading dimension of C2
 * @param[out] dwork Workspace, dimension (n)
 */
void mb04oy(const i32* m, const i32* n, const f64* v, const f64* tau,
            f64* c1, const i32* ldc1, f64* c2, const i32* ldc2, f64* dwork);

/**
 * @brief Apply product of symplectic reflectors and Givens rotations (blocked).
 *
 * Overwrites m-by-n matrices C and D with Q*[op(C);op(D)] or Q^T*[op(C);op(D)]
 * where Q is defined as product of symplectic reflectors and Givens rotations:
 *   Q = diag(H(1),H(1)) G(1) diag(F(1),F(1)) ... diag(H(k),H(k)) G(k) diag(F(k),F(k))
 *
 * Blocked version.
 *
 * @param[in] tranc Form of op(C): 'N'=C, 'T'/'C'=C^T
 * @param[in] trand Form of op(D): 'N'=D, 'T'/'C'=D^T
 * @param[in] tranq 'N'=apply Q, 'T'=apply Q^T
 * @param[in] storev Storage of V reflectors: 'C'=columnwise, 'R'=rowwise
 * @param[in] storew Storage of W reflectors: 'C'=columnwise, 'R'=rowwise
 * @param[in] m Number of rows of op(C) and op(D) (m >= 0)
 * @param[in] n Number of columns of op(C) and op(D) (n >= 0)
 * @param[in] k Number of elementary reflectors (m >= k >= 0)
 * @param[in] v Reflector vectors F(i), dimension (ldv,k) or (ldv,m)
 * @param[in] ldv Leading dimension of V
 * @param[in] w Reflector vectors H(i), dimension (ldw,k) or (ldw,m)
 * @param[in] ldw Leading dimension of W
 * @param[in,out] c Matrix C, dimension (ldc,n) or (ldc,m)
 * @param[in] ldc Leading dimension of C
 * @param[in,out] d Matrix D, dimension (ldd,n) or (ldd,m)
 * @param[in] ldd Leading dimension of D
 * @param[in] cs Cosines and sines of Givens rotations G(i), dimension (2*k)
 * @param[in] tau Scalar factors of reflectors F(i), dimension (k)
 * @param[out] dwork Workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size (ldwork >= max(1,n))
 * @param[out] info Exit code: 0=success, <0=invalid parameter -info
 */
void mb04qb(const char *tranc, const char *trand, const char *tranq,
            const char *storev, const char *storew, i32 m, i32 n, i32 k,
            const f64 *v, i32 ldv, const f64 *w, i32 ldw,
            f64 *c, i32 ldc, f64 *d, i32 ldd,
            const f64 *cs, const f64 *tau,
            f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Apply symplectic block reflector to matrices.
 *
 * Applies the orthogonal symplectic block reflector Q or Q^T to a real
 * 2m-by-n matrix [op(A); op(B)] from the left:
 *
 *         [  I+V*T*V'  V*R*S*V'  ]
 *    Q =  [                      ]
 *         [ -V*R*S*V'  I+V*T*V'  ]
 *
 * The block factors R, S, T are computed by MB04QF.
 *
 * @param[in] strab Structure: 'Z'=leading K rows zero, 'N'=no structure
 * @param[in] trana Form of op(A): 'N'=A, 'T'/'C'=A^T
 * @param[in] tranb Form of op(B): 'N'=B, 'T'/'C'=B^T
 * @param[in] tranq Apply: 'N'=Q, 'T'=Q^T
 * @param[in] direct Reserved for future use
 * @param[in] storev Storage of V reflectors: 'C'=columnwise, 'R'=rowwise
 * @param[in] storew Storage of W reflectors: 'C'=columnwise, 'R'=rowwise
 * @param[in] m Rows of op(A) and op(B) (m >= 0)
 * @param[in] n Columns of op(A) and op(B) (n >= 0)
 * @param[in] k Order of block factors (m >= k >= 0)
 * @param[in] v V reflector vectors
 * @param[in] ldv Leading dimension of V
 * @param[in] w W reflector vectors
 * @param[in] ldw Leading dimension of W
 * @param[in] rs R,S block factors from MB04QF, dimension (ldrs,6*k)
 * @param[in] ldrs Leading dimension of RS (ldrs >= k)
 * @param[in] t T block factor from MB04QF, dimension (ldt,9*k)
 * @param[in] ldt Leading dimension of T (ldt >= k)
 * @param[in,out] a Matrix A (modified)
 * @param[in] lda Leading dimension of A
 * @param[in,out] b Matrix B (modified)
 * @param[in] ldb Leading dimension of B
 * @param[out] dwork Workspace, dimension 8*n*k (STRAB='Z') or 9*n*k (STRAB='N')
 */
void mb04qc(const char *strab, const char *trana, const char *tranb,
            const char *tranq, const char *direct, const char *storev,
            const char *storew, i32 m, i32 n, i32 k,
            const f64 *v, i32 ldv, const f64 *w, i32 ldw,
            const f64 *rs, i32 ldrs, const f64 *t, i32 ldt,
            f64 *a, i32 lda, f64 *b, i32 ldb,
            f64 *dwork);

/**
 * @brief Form triangular block factors of symplectic block reflector.
 *
 * Forms triangular block factors R, S, T of symplectic block reflector SH
 * defined as a product of 2k Householder reflectors and k Givens rotations:
 *   SH = diag(H(1),H(1)) G(1) diag(F(1),F(1)) ... diag(H(k),H(k)) G(k) diag(F(k),F(k))
 *
 * The upper triangular blocks of R=[R1 R2 R3], S=[S1;S2;S3], T=[T11 T12 T13;...]
 * are stored rowwise in RS and T arrays.
 *
 * @param[in] direct Reserved for future use (not referenced)
 * @param[in] storev Storage of F(i) reflectors: 'C'=columnwise, 'R'=rowwise
 * @param[in] storew Storage of H(i) reflectors: 'C'=columnwise, 'R'=rowwise
 * @param[in] n Order of reflectors F(i) and H(i) (n >= 0)
 * @param[in] k Number of Givens rotations (k >= 1)
 * @param[in] v Reflector vectors F(i), dimension (ldv,k) or (ldv,n)
 * @param[in] ldv Leading dimension of V
 * @param[in] w Reflector vectors H(i), dimension (ldw,k) or (ldw,n)
 * @param[in] ldw Leading dimension of W
 * @param[in] cs Cosines and sines of Givens rotations G(i), dimension (2*k)
 * @param[in] tau Scalar factors of reflectors F(i), dimension (k)
 * @param[out] rs Upper triangular R and S factors, dimension (ldrs,6*k)
 * @param[in] ldrs Leading dimension of RS (ldrs >= k)
 * @param[out] t Upper triangular T factor, dimension (ldt,9*k)
 * @param[in] ldt Leading dimension of T (ldt >= k)
 * @param[out] dwork Workspace, dimension (3*k)
 */
void mb04qf(const char *direct, const char *storev, const char *storew,
            i32 n, i32 k,
            f64 *v, i32 ldv, f64 *w, i32 ldw,
            const f64 *cs, const f64 *tau,
            f64 *rs, i32 ldrs, f64 *t, i32 ldt,
            f64 *dwork);

/**
 * @brief Apply product of symplectic reflectors and Givens rotations.
 *
 * Overwrites m-by-n matrices C and D with Q*[op(C);op(D)] or Q^T*[op(C);op(D)]
 * where Q is defined as product of symplectic reflectors and Givens rotations:
 *   Q = diag(H(1),H(1)) G(1) diag(F(1),F(1)) ... diag(H(k),H(k)) G(k) diag(F(k),F(k))
 *
 * Unblocked version.
 *
 * @param[in] tranc Form of op(C): 'N'=C, 'T'/'C'=C^T
 * @param[in] trand Form of op(D): 'N'=D, 'T'/'C'=D^T
 * @param[in] tranq 'N'=apply Q, 'T'=apply Q^T
 * @param[in] storev Storage of V reflectors: 'C'=columnwise, 'R'=rowwise
 * @param[in] storew Storage of W reflectors: 'C'=columnwise, 'R'=rowwise
 * @param[in] m Number of rows of op(C) and op(D) (m >= 0)
 * @param[in] n Number of columns of op(C) and op(D) (n >= 0)
 * @param[in] k Number of elementary reflectors (m >= k >= 0)
 * @param[in] v Reflector vectors F(i), dimension (ldv,k) or (ldv,m)
 * @param[in] ldv Leading dimension of V
 * @param[in] w Reflector vectors H(i), dimension (ldw,k) or (ldw,m)
 * @param[in] ldw Leading dimension of W
 * @param[in,out] c Matrix C, dimension (ldc,n) or (ldc,m)
 * @param[in] ldc Leading dimension of C
 * @param[in,out] d Matrix D, dimension (ldd,n) or (ldd,m)
 * @param[in] ldd Leading dimension of D
 * @param[in] cs Cosines and sines of Givens rotations G(i), dimension (2*k)
 * @param[in] tau Scalar factors of reflectors F(i), dimension (k)
 * @param[out] dwork Workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size (ldwork >= max(1,n))
 * @param[out] info Exit code: 0=success, <0=invalid parameter -info
 */
void mb04qu(const char *tranc, const char *trand, const char *tranq,
            const char *storev, const char *storew,
            i32 m, i32 n, i32 k,
            f64 *v, i32 ldv, f64 *w, i32 ldw,
            f64 *c, i32 ldc, f64 *d, i32 ldd,
            const f64 *cs, const f64 *tau,
            f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Symplectic URV decomposition (blocked version).
 *
 * Computes H = U * R * V^T where:
 * - H = [op(A) G; Q op(B)] is 2N-by-2N Hamiltonian matrix
 * - U, V are 2N-by-2N orthogonal symplectic matrices
 * - R = [op(R11) R12; 0 op(R22)] is block upper triangular
 * - op(R11) is upper triangular, op(R22) is lower Hessenberg
 *
 * @param[in] trana 'N' for op(A)=A, 'T'/'C' for op(A)=A^T
 * @param[in] tranb 'N' for op(B)=B, 'T'/'C' for op(B)=B^T
 * @param[in] n Order of matrices A, B, G, Q (n >= 0)
 * @param[in] ilo Starting index for reduction (1 <= ilo <= n+1 for n>0, ilo=1 for n=0)
 * @param[in,out] a N-by-N matrix A on entry, R11 and reflector info on exit
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] b N-by-N matrix B on entry, R22 (Hessenberg) and reflector info on exit
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in,out] g N-by-N matrix G on entry, R12 on exit
 * @param[in] ldg Leading dimension of G (ldg >= max(1,n))
 * @param[in,out] q N-by-N matrix Q on entry, reflector info on exit
 * @param[in] ldq Leading dimension of Q (ldq >= max(1,n))
 * @param[out] csl Cosines/sines of left Givens rotations, dimension (2*n)
 * @param[out] csr Cosines/sines of right Givens rotations, dimension (2*n-2)
 * @param[out] taul Scalar factors of left reflectors, dimension (n)
 * @param[out] taur Scalar factors of right reflectors, dimension (n-1)
 * @param[out] dwork Workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size (ldwork >= max(1,n))
 * @param[out] info Exit code: 0=success, <0=invalid parameter -info
 */
void mb04tb(const char *trana, const char *tranb, i32 n, i32 ilo,
            f64 *a, i32 lda, f64 *b, i32 ldb, f64 *g, i32 ldg,
            f64 *q, i32 ldq, f64 *csl, f64 *csr, f64 *taul, f64 *taur,
            f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Symplectic URV decomposition (unblocked version).
 *
 * Computes H = U * R * V^T where:
 * - H = [op(A) G; Q op(B)] is 2N-by-2N Hamiltonian matrix
 * - U, V are 2N-by-2N orthogonal symplectic matrices
 * - R = [op(R11) R12; 0 op(R22)] is block upper triangular
 * - op(R11) is upper triangular, op(R22) is lower Hessenberg
 *
 * @param[in] trana 'N' for op(A)=A, 'T'/'C' for op(A)=A^T
 * @param[in] tranb 'N' for op(B)=B, 'T'/'C' for op(B)=B^T
 * @param[in] n Order of matrices A, B, G, Q (n >= 0)
 * @param[in] ilo Starting index for reduction (1 <= ilo <= n for n>0, ilo=1 for n=0)
 * @param[in,out] a N-by-N matrix A on entry, R11 and reflector info on exit
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] b N-by-N matrix B on entry, R22 (Hessenberg) and reflector info on exit
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in,out] g N-by-N matrix G on entry, R12 on exit
 * @param[in] ldg Leading dimension of G (ldg >= max(1,n))
 * @param[in,out] q N-by-N matrix Q on entry, reflector info on exit
 * @param[in] ldq Leading dimension of Q (ldq >= max(1,n))
 * @param[out] csl Cosines/sines of left Givens rotations, dimension (2*n)
 * @param[out] csr Cosines/sines of right Givens rotations, dimension (2*n-2)
 * @param[out] taul Scalar factors of left reflectors, dimension (n)
 * @param[out] taur Scalar factors of right reflectors, dimension (n-1)
 * @param[out] dwork Workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size (ldwork >= max(1,n))
 * @param[out] info Exit code: 0=success, <0=invalid parameter -info
 */
void mb04ts(const char *trana, const char *tranb, i32 n, i32 ilo,
            f64 *a, i32 lda, f64 *b, i32 ldb, f64 *g, i32 ldg,
            f64 *q, i32 ldq, f64 *csl, f64 *csr, f64 *taul, f64 *taur,
            f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Hamiltonian matrix square-reduction.
 *
 * Transforms a Hamiltonian matrix H = [[A, G], [Q, -A^T]] into a
 * square-reduced Hamiltonian matrix H' = [[A', G'], [Q', -A'^T]]
 * by an orthogonal symplectic similarity transformation H' = U^T H U.
 *
 * The square-reduced form satisfies Q'A' - A'^T Q' = 0. The square
 * of H' has the form [[A'', G''], [0, A''^T]] where A'' is upper
 * Hessenberg and G'' is skew symmetric.
 *
 * Uses implicit Van Loan's method.
 *
 * @param[in] compu Indicates transformation matrix handling:
 *                  'N' - U is not required
 *                  'I'/'F' - On exit, U contains orthogonal symplectic matrix
 *                  'V'/'A' - On input, U contains orthogonal symplectic S;
 *                            On exit, U contains S*U
 * @param[in] n Order of matrices A, G, Q (n >= 0)
 * @param[in,out] a On input: N-by-N upper left block A
 *                  On output: N-by-N upper left block A' (square-reduced)
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] qg On input: N-by-(N+1) array storing Q and G in packed form:
 *                   - Q(i,j) stored in QG(i,j) for i >= j (lower triangular)
 *                   - G(i,j) stored in QG(j,i+1) for i >= j (upper triangular)
 *                   On output: Square-reduced Q' and G' in same format
 * @param[in] ldqg Leading dimension of QG (ldqg >= max(1,n))
 * @param[in,out] u N-by-2N array for transformation matrix:
 *                  - If COMPU='N': not referenced
 *                  - If COMPU='I'/'F': on exit, first N rows of U
 *                  - If COMPU='V'/'A': on input, first N rows of S;
 *                                      on exit, first N rows of S*U
 * @param[in] ldu Leading dimension of U (ldu >= max(1,n) if COMPU != 'N',
 *                ldu >= 1 otherwise)
 * @param[out] dwork Workspace, dimension (2*n)
 * @param[out] info Exit code:
 *                  0 = success
 *                  < 0 = invalid parameter -info
 */
void mb04zd(const char *compu, i32 n, f64 *a, i32 lda, f64 *qg, i32 ldqg,
            f64 *u, i32 ldu, f64 *dwork, i32 *info);

/**
 * @brief Eigenvalues of real skew-Hamiltonian/Hamiltonian pencil.
 *
 * Computes the eigenvalues of a real N-by-N skew-Hamiltonian/Hamiltonian
 * pencil aS - bH with S = T Z = J Z' J' Z via generalized symplectic URV
 * decomposition.
 *
 * @param[in] job Computation mode:
 *                'E' = eigenvalues only
 *                'T' = triangular form and eigenvalues
 * @param[in] compq1 Q1 computation: 'N'=none, 'I'=init, 'U'=update
 * @param[in] compq2 Q2 computation: 'N'=none, 'I'=init, 'U'=update
 * @param[in] compu1 U1 computation: 'N'=none, 'I'=init, 'U'=update
 * @param[in] compu2 U2 computation: 'N'=none, 'I'=init, 'U'=update
 * @param[in] n Order of pencil (n >= 0, must be even)
 * @param[in,out] z N-by-N matrix Z
 * @param[in] ldz Leading dimension of Z (ldz >= max(1,n))
 * @param[in,out] h N-by-N Hamiltonian matrix H
 * @param[in] ldh Leading dimension of H (ldh >= max(1,n))
 * @param[in,out] q1 N-by-N orthogonal transformation Q1
 * @param[in] ldq1 Leading dimension of Q1
 * @param[in,out] q2 N-by-N orthogonal transformation Q2
 * @param[in] ldq2 Leading dimension of Q2
 * @param[in,out] u11 (N/2)-by-(N/2) upper left block of U1
 * @param[in] ldu11 Leading dimension of U11
 * @param[in,out] u12 (N/2)-by-(N/2) upper right block of U1
 * @param[in] ldu12 Leading dimension of U12
 * @param[in,out] u21 (N/2)-by-(N/2) upper left block of U2
 * @param[in] ldu21 Leading dimension of U21
 * @param[in,out] u22 (N/2)-by-(N/2) upper right block of U2
 * @param[in] ldu22 Leading dimension of U22
 * @param[out] t N-by-N output matrix T
 * @param[in] ldt Leading dimension of T (ldt >= max(1,n))
 * @param[out] alphar Real parts of eigenvalues (N/2)
 * @param[out] alphai Imaginary parts of eigenvalues (N/2)
 * @param[out] beta Scaling factors for eigenvalues (N/2)
 * @param[out] iwork Integer workspace (N+18)
 * @param[in] liwork Size of iwork (>= N+18)
 * @param[out] dwork Double workspace
 * @param[in] ldwork Size of dwork (-1 for query)
 * @param[out] info Exit code: 0=success, <0=param error, 1-3=algorithm issues
 */
void mb04ad(const char *job, const char *compq1, const char *compq2,
            const char *compu1, const char *compu2, i32 n,
            f64 *z, i32 ldz, f64 *h, i32 ldh,
            f64 *q1, i32 ldq1, f64 *q2, i32 ldq2,
            f64 *u11, i32 ldu11, f64 *u12, i32 ldu12,
            f64 *u21, i32 ldu21, f64 *u22, i32 ldu22,
            f64 *t, i32 ldt,
            f64 *alphar, f64 *alphai, f64 *beta,
            i32 *iwork, i32 liwork,
            f64 *dwork, i32 ldwork,
            i32 *info);

/**
 * @brief Eigenvalues of skew-Hamiltonian/Hamiltonian pencil
 *
 * Computes the eigenvalues of a real N-by-N skew-Hamiltonian/Hamiltonian
 * pencil aS - bH with
 *
 *       (  A  D  )         (  C  V  )
 *   S = (        ) and H = (        )
 *       (  E  A' )         (  W -C' )
 *
 * Optionally computes decompositions via orthogonal transformations Q1, Q2.
 *
 * @param[in] job 'E'=eigenvalues only, 'T'=triangular form and eigenvalues
 * @param[in] compq1 Q1 computation: 'N'=none, 'I'=init, 'U'=update
 * @param[in] compq2 Q2 computation: 'N'=none, 'I'=init, 'U'=update
 * @param[in] n Order of pencil (n >= 0, must be even)
 * @param[in,out] a N/2-by-N/2 matrix A
 * @param[in] lda Leading dimension of A (lda >= max(1,n/2))
 * @param[in,out] de N/2-by-(N/2+1) matrix containing E (lower) and D (upper)
 * @param[in] ldde Leading dimension of DE (ldde >= max(1,n/2))
 * @param[in,out] c1 N/2-by-N/2 matrix C1=C
 * @param[in] ldc1 Leading dimension of C1 (ldc1 >= max(1,n/2))
 * @param[in,out] vw N/2-by-(N/2+1) matrix containing W (lower) and V (upper)
 * @param[in] ldvw Leading dimension of VW (ldvw >= max(1,n/2))
 * @param[in,out] q1 N-by-N orthogonal transformation Q1 (if compq1 != 'N')
 * @param[in] ldq1 Leading dimension of Q1
 * @param[out] q2 N-by-N orthogonal transformation Q2 (if compq2 != 'N')
 * @param[in] ldq2 Leading dimension of Q2
 * @param[out] b N/2-by-N/2 output matrix B
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n/2))
 * @param[out] f N/2-by-N/2 skew-symmetric matrix F
 * @param[in] ldf Leading dimension of F (ldf >= max(1,n/2))
 * @param[out] c2 N/2-by-N/2 output matrix C2
 * @param[in] ldc2 Leading dimension of C2 (ldc2 >= max(1,n/2))
 * @param[out] alphar Real parts of eigenvalues (N/2)
 * @param[out] alphai Imaginary parts of eigenvalues (N/2)
 * @param[out] beta Scaling factors for eigenvalues (N/2)
 * @param[out] iwork Integer workspace (N+12)
 * @param[in] liwork Size of iwork (>= N+12)
 * @param[out] dwork Double workspace
 * @param[in] ldwork Size of dwork
 * @param[out] info Exit code: 0=success, <0=param error, 1-3=algorithm issues
 */
void mb04bd(const char *job, const char *compq1, const char *compq2,
            i32 n, f64 *a, i32 lda, f64 *de, i32 ldde, f64 *c1, i32 ldc1,
            f64 *vw, i32 ldvw, f64 *q1, i32 ldq1, f64 *q2, i32 ldq2,
            f64 *b, i32 ldb, f64 *f, i32 ldf, f64 *c2, i32 ldc2,
            f64 *alphar, f64 *alphai, f64 *beta,
            i32 *iwork, i32 liwork, f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief MB04BP - Eigenvalues of skew-Hamiltonian/Hamiltonian pencil (block algorithm)
 *
 * Computes the eigenvalues of a real N-by-N skew-Hamiltonian/Hamiltonian
 * pencil aS - bH using a block algorithm for transformations on panels.
 * For small N (<=250), delegates to MB04BD directly.
 *
 * Same interface as MB04BD but uses blocking for better performance on large N.
 *
 * @param[in] job 'E'=eigenvalues only, 'T'=Schur form and eigenvalues
 * @param[in] compq1 Q1 computation: 'N'=none, 'I'=init, 'U'=update
 * @param[in] compq2 Q2 computation: 'N'=none, 'I'=init, 'U'=update
 * @param[in] n Order of pencil (n >= 0, must be even)
 * @param[in,out] a N/2-by-N/2 matrix A
 * @param[in] lda Leading dimension of A (lda >= max(1,n/2))
 * @param[in,out] de N/2-by-(N/2+1) array containing E (lower) and D (upper)
 * @param[in] ldde Leading dimension of DE (ldde >= max(1,n/2))
 * @param[in,out] c1 N/2-by-N/2 matrix C1=C
 * @param[in] ldc1 Leading dimension of C1 (ldc1 >= max(1,n/2))
 * @param[in,out] vw N/2-by-(N/2+1) array containing W (lower) and V (upper)
 * @param[in] ldvw Leading dimension of VW (ldvw >= max(1,n/2))
 * @param[in,out] q1 N-by-N orthogonal transformation Q1 (if compq1 != 'N')
 * @param[in] ldq1 Leading dimension of Q1
 * @param[out] q2 N-by-N orthogonal transformation Q2 (if compq2 != 'N')
 * @param[in] ldq2 Leading dimension of Q2
 * @param[out] b N/2-by-N/2 output matrix B
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n/2))
 * @param[out] f N/2-by-N/2 output skew-symmetric matrix F
 * @param[in] ldf Leading dimension of F (ldf >= max(1,n/2))
 * @param[out] c2 N/2-by-N/2 output matrix C2
 * @param[in] ldc2 Leading dimension of C2 (ldc2 >= max(1,n/2))
 * @param[out] alphar Real parts of eigenvalues
 * @param[out] alphai Imaginary parts of eigenvalues
 * @param[out] beta Scale factors for eigenvalues
 * @param[out] iwork Integer workspace
 * @param[in] liwork Size of iwork (>= n+12)
 * @param[out] dwork Double workspace
 * @param[in] ldwork Size of dwork
 * @param[in,out] info On entry: block size hint (<0=auto, 0=use MB04BD). On exit: status
 */
void mb04bp(const char *job, const char *compq1, const char *compq2,
            i32 n, f64 *a, i32 lda, f64 *de, i32 ldde, f64 *c1, i32 ldc1,
            f64 *vw, i32 ldvw, f64 *q1, i32 ldq1, f64 *q2, i32 ldq2,
            f64 *b, i32 ldb, f64 *f, i32 ldf, f64 *c2, i32 ldc2,
            f64 *alphar, f64 *alphai, f64 *beta,
            i32 *iwork, i32 liwork, f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief MB04CD - Reduce skew-Hamiltonian/Hamiltonian pencil to generalized Schur form
 *
 * Computes transformed matrices A, B and D using orthogonal matrices Q1, Q2, Q3
 * for a real N-by-N regular pencil
 *
 *                ( A11   0  ) ( B11   0  )     (  0   D12 )
 *   aA*B - bD = a(          )(          ) - b (          ),
 *                (  0   A22 ) (  0   B22 )     ( D21   0  )
 *
 * where A11, A22, B11, B22 and D12 are upper triangular, D21 is upper
 * quasi-triangular and the generalized matrix product
 *   A11^(-1) D12 B22^(-1) A22^(-1) D21 B11^(-1) is upper quasi-triangular.
 *
 * @param[in] compq1 Q1 computation: 'N'=none, 'I'=init, 'U'=update
 * @param[in] compq2 Q2 computation: 'N'=none, 'I'=init, 'U'=update
 * @param[in] compq3 Q3 computation: 'N'=none, 'I'=init, 'U'=update
 * @param[in] n Order of pencil (n >= 0, must be even)
 * @param[in,out] a N-by-N matrix A
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] b N-by-N matrix B
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in,out] d N-by-N matrix D
 * @param[in] ldd Leading dimension of D (ldd >= max(1,n))
 * @param[in,out] q1 N-by-N orthogonal transformation Q1 (if compq1 != 'N')
 * @param[in] ldq1 Leading dimension of Q1
 * @param[in,out] q2 N-by-N orthogonal transformation Q2 (if compq2 != 'N')
 * @param[in] ldq2 Leading dimension of Q2
 * @param[in,out] q3 N-by-N orthogonal transformation Q3 (if compq3 != 'N')
 * @param[in] ldq3 Leading dimension of Q3
 * @param[out] iwork Integer workspace
 * @param[in] liwork Size of iwork
 * @param[out] dwork Double workspace
 * @param[in] ldwork Size of dwork
 * @param[out] bwork Boolean workspace
 * @param[out] info Exit code: 0=success, <0=param error, 1-4=algorithm issues
 */
void mb04cd(const char *compq1, const char *compq2, const char *compq3,
            i32 n, f64 *a, i32 lda, f64 *b, i32 ldb, f64 *d, i32 ldd,
            f64 *q1, i32 ldq1, f64 *q2, i32 ldq2, f64 *q3, i32 ldq3,
            i32 *iwork, i32 liwork, f64 *dwork, i32 ldwork,
            bool *bwork, i32 *info);

/**
 * @brief Apply inverse balancing transformation to [V1; sgn*V2].
 *
 * Applies from the left the inverse of a balancing transformation,
 * computed by the SLICOT Library routine MB04DP, to the matrix
 * [V1; sgn*V2] where sgn is either +1 or -1.
 *
 * @param[in] job Type of inverse transformation:
 *                'N' = do nothing, return immediately
 *                'P' = inverse permutation only
 *                'S' = inverse scaling only
 *                'B' = both inverse permutation and scaling
 * @param[in] sgn Sign to use for V2: 'P' = +1, 'N' = -1
 * @param[in] n Number of rows in V1 and V2, n >= 0
 * @param[in] ilo ILO value from MB04DP, 1 <= ilo <= n+1
 * @param[in] lscale Left permutation/scaling factors from MB04DP, dim (n)
 * @param[in] rscale Right permutation/scaling factors from MB04DP, dim (n)
 * @param[in] m Number of columns in V1 and V2, m >= 0
 * @param[in,out] v1 N-by-M matrix V1
 * @param[in] ldv1 Leading dimension of V1, ldv1 >= max(1,n)
 * @param[in,out] v2 N-by-M matrix V2
 * @param[in] ldv2 Leading dimension of V2, ldv2 >= max(1,n)
 * @param[out] info 0 = success, < 0 = -i-th argument had illegal value
 */
void mb04db(const char *job, const char *sgn, i32 n, i32 ilo,
            const f64 *lscale, const f64 *rscale, i32 m,
            f64 *v1, i32 ldv1, f64 *v2, i32 ldv2, i32 *info);

/**
 * @brief Apply inverse balancing transformation to [V1; sgn*V2].
 *
 * Applies from the left the inverse of a balancing transformation,
 * computed by the SLICOT Library routine MB04DD or MB04DS, to the matrix
 * [V1; sgn*V2] where sgn is either +1 or -1.
 *
 * @param[in] job Type of inverse transformation:
 *                'N' = do nothing, return immediately
 *                'P' = inverse permutation only
 *                'S' = inverse scaling only
 *                'B' = both inverse permutation and scaling
 * @param[in] sgn Sign to use for V2: 'P' = +1, 'N' = -1
 * @param[in] n Number of rows in V1 and V2, n >= 0
 * @param[in] ilo ILO value from MB04DD or MB04DS, 1 <= ilo <= n+1
 * @param[in] scale Permutation/scaling factors from MB04DD or MB04DS, dim (n)
 * @param[in] m Number of columns in V1 and V2, m >= 0
 * @param[in,out] v1 N-by-M matrix V1
 * @param[in] ldv1 Leading dimension of V1, ldv1 >= max(1,n)
 * @param[in,out] v2 N-by-M matrix V2
 * @param[in] ldv2 Leading dimension of V2, ldv2 >= max(1,n)
 * @param[out] info 0 = success, < 0 = -i-th argument had illegal value
 */
void mb04di(const char *job, const char *sgn, i32 n, i32 ilo,
            const f64 *scale, i32 m,
            f64 *v1, i32 ldv1, f64 *v2, i32 ldv2, i32 *info);

/**
 * @brief Balance a real matrix pencil (A, B).
 *
 * Balances a pair of N-by-N real matrices (A,B) by:
 * 1. Permuting by equivalence transformations to isolate eigenvalues
 * 2. Applying diagonal equivalence transformation to equalize row/column 1-norms
 *
 * Optionally improves conditioning compared to LAPACK DGGBAL.
 *
 * @param[in] job Operations to perform:
 *                'N' = none, set ILO=1, scales to 1.0
 *                'P' = permute only
 *                'S' = scale only
 *                'B' = both permute and scale
 * @param[in] n Order of matrices A and B (n >= 0)
 * @param[in] thresh Threshold for scaling elements:
 *                   >= 0: elements with |val| <= thresh*max_norm are ignored
 *                   < 0: automatic threshold selection (-1 to -4, or <= -10)
 * @param[in,out] a N-by-N matrix A, balanced on exit
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] b N-by-N matrix B, balanced on exit
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[out] ilo Index such that A(i,j)=B(i,j)=0 for i>j and j=1..ilo-1
 * @param[out] ihi Index such that A(i,j)=B(i,j)=0 for i>j and i=ihi+1..n
 * @param[out] lscale Left permutation/scaling factors, dimension (n)
 * @param[out] rscale Right permutation/scaling factors, dimension (n)
 * @param[out] dwork Workspace, dimension (ldwork):
 *                   0 if job='N'/'P' or n=0
 *                   6*n if (job='S'/'B') and thresh >= 0
 *                   8*n if (job='S'/'B') and thresh < 0
 *                   On exit: dwork[0:1] initial norms, dwork[2:3] final norms,
 *                   dwork[4] threshold used
 * @param[out] iwarn Warning indicator:
 *                   0 = no warning
 *                   1 = scaling reset to 1 (for thresh=-2/-4)
 * @param[out] info Error indicator:
 *                  0 = success
 *                  < 0 = parameter -info had illegal value
 */
void mb04dl(const char *job, i32 n, f64 thresh, f64 *a, i32 lda,
            f64 *b, i32 ldb, i32 *ilo, i32 *ihi, f64 *lscale, f64 *rscale,
            f64 *dwork, i32 *iwarn, i32 *info);

/**
 * @brief Balance a real skew-Hamiltonian/Hamiltonian pencil.
 *
 * Balances the 2*N-by-2*N skew-Hamiltonian/Hamiltonian pencil aS - bH with:
 *     S = [[A, D], [E, A']] and H = [[C, V], [W, -C']]
 * where D, E are skew-symmetric and V, W are symmetric.
 *
 * Involves:
 * 1. Permuting to isolate eigenvalues in first ILO-1 elements
 * 2. Diagonal equivalence to equalize row/column 1-norms
 *
 * @param[in] job Operations to perform:
 *                'N' = none, set ILO=1, scales to 1.0
 *                'P' = permute only
 *                'S' = scale only
 *                'B' = both permute and scale
 * @param[in] n Order of matrices A, D, E, C, V, W (n >= 0)
 * @param[in] thresh Threshold for scaling:
 *                   >= 0: elements with |val| <= thresh*max_norm ignored
 *                   < 0: automatic threshold selection
 * @param[in,out] a N-by-N matrix A, balanced on exit
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] de N-by-(N+1) array: E (lower tri) and D (upper tri cols 2:N+1)
 * @param[in] ldde Leading dimension of DE (ldde >= max(1,n))
 * @param[in,out] c N-by-N matrix C, balanced on exit
 * @param[in] ldc Leading dimension of C (ldc >= max(1,n))
 * @param[in,out] vw N-by-(N+1) array: W (lower tri) and V (upper tri cols 2:N+1)
 * @param[in] ldvw Leading dimension of VW (ldvw >= max(1,n))
 * @param[out] ilo ILO-1 is number of deflated eigenvalues
 * @param[out] lscale Left permutation/scaling factors, dimension (n)
 * @param[out] rscale Right permutation/scaling factors, dimension (n)
 * @param[out] dwork Workspace, dimension (ldwork):
 *                   0 if job='N'/'P' or n=0
 *                   6*n if (job='S'/'B') and thresh >= 0
 *                   8*n if (job='S'/'B') and thresh < 0
 * @param[out] iwarn Warning indicator:
 *                   0 = no warning
 *                   1 = scaling reset to 1
 * @param[out] info Error indicator:
 *                  0 = success
 *                  < 0 = parameter -info had illegal value
 */
void mb04dp(const char *job, i32 n, f64 thresh, f64 *a, i32 lda,
            f64 *de, i32 ldde, f64 *c, i32 ldc, f64 *vw, i32 ldvw,
            i32 *ilo, f64 *lscale, f64 *rscale, f64 *dwork,
            i32 *iwarn, i32 *info);

/**
 * @brief MB04DS - Balance a real skew-Hamiltonian matrix
 *
 * Balances a real 2N-by-2N skew-Hamiltonian matrix:
 *     S = [  A   G  ]
 *         [  Q  A^T ]
 * where A is N-by-N and G, Q are N-by-N skew-symmetric matrices.
 *
 * Balancing involves:
 * 1. Permuting S to isolate eigenvalues in first 1:ILO-1 diagonal elements
 * 2. Diagonal similarity transformation on rows/columns ILO:N, N+ILO:2*N
 *
 * @param[in] job Operations to perform:
 *                'N' = none, set ILO=1, SCALE=1.0
 *                'P' = permute only
 *                'S' = scale only
 *                'B' = both permute and scale
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in,out] a N-by-N matrix A, balanced on exit
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] qg N-by-(N+1) array containing:
 *                   - Q (strictly lower triangular, cols 1:N)
 *                   - G (strictly upper triangular, cols 2:N+1)
 *                   Balanced on exit
 * @param[in] ldqg Leading dimension of QG (ldqg >= max(1,n))
 * @param[out] ilo ILO-1 is number of deflated eigenvalues
 * @param[out] scale Permutation/scaling factors, dimension (n):
 *                   - j=1,...,ILO-1: P(j)=SCALE(j) is permutation info
 *                   - j=ILO,...,N: scaling factor for row/column j
 * @param[out] info Error indicator:
 *                  0 = success
 *                  < 0 = parameter -info had illegal value
 */
void mb04ds(const char *job, i32 n, f64 *a, i32 lda,
            f64 *qg, i32 ldqg, i32 *ilo, f64 *scale, i32 *info);

/**
 * @brief MB04DY - Symplectic scaling of a Hamiltonian matrix
 *
 * Performs symplectic scaling on a Hamiltonian matrix:
 *     H = [ A    G  ]
 *         [ Q   -A' ]
 * where A is N-by-N and G, Q are symmetric N-by-N matrices.
 *
 * Scaling strategies:
 * - 'S': Symplectic scaling using DGEBAL + equilibration
 * - '1'/'O': 1-norm scaling by power of machine base
 * - 'N': No scaling
 *
 * @param[in] jobscl Scaling strategy:
 *                   'S' = symplectic scaling
 *                   '1'/'O' = 1-norm scaling
 *                   'N' = no scaling
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in,out] a N-by-N matrix A, scaled on exit
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] qg N-by-(N+1) array containing:
 *                   - Q (lower triangular, cols 0:N-1)
 *                   - G (upper triangular, cols 1:N)
 *                   Scaled on exit
 * @param[in] ldqg Leading dimension of QG (ldqg >= max(1,n))
 * @param[out] d Scaling factors:
 *               - jobscl='S': dimension N, diagonal scaling matrix
 *               - jobscl='1'/'O': dimension 1, tau value
 *               - jobscl='N': not referenced
 * @param[out] dwork Workspace, dimension (N) if jobscl != 'N'
 * @param[out] info Error indicator:
 *                  0 = success
 *                  < 0 = parameter -info had illegal value
 */
void mb04dy(const char *jobscl, i32 n, f64 *a, i32 lda,
            f64 *qg, i32 ldqg, f64 *d, f64 *dwork, i32 *info);

/**
 * @brief Balance a complex Hamiltonian matrix.
 *
 * Balances a complex Hamiltonian matrix H = [A, G; Q, -A^H] where A is NxN
 * and G, Q are NxN Hermitian matrices. Involves permuting to isolate
 * eigenvalues and diagonal similarity transformations.
 *
 * @param[in] job 'N' = none, 'P' = permute only, 'S' = scale only, 'B' = both
 * @param[in] n Order of matrix A (N >= 0)
 * @param[in,out] a N-by-N complex matrix A, balanced on exit
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] qg N-by-(N+1) complex matrix with lower tri of Q and upper tri of G
 * @param[in] ldqg Leading dimension of QG (ldqg >= max(1,n))
 * @param[out] ilo Number of deflated eigenvalues + 1
 * @param[out] scale Permutation and scaling factors, dimension N
 * @param[out] info Exit code (0 = success, <0 = invalid parameter -i)
 */
void mb04dz(const char *job, i32 n, c128 *a, i32 lda,
            c128 *qg, i32 ldqg, i32 *ilo, f64 *scale, i32 *info);

/**
 * @brief Eigenvalues and orthogonal decomposition of a skew-Hamiltonian/
 *        skew-Hamiltonian pencil in factored form.
 *
 * Computes eigenvalues of aS - bT where S = J*Z'*J'*Z and T = [[B,F],[G,B']].
 * Optionally reduces to structured Schur form with orthogonal Q and
 * orthogonal symplectic U.
 *
 * @param[in] job 'E' = eigenvalues only, 'T' = also structured Schur form
 * @param[in] compq 'N' = don't compute Q, 'I' = initialize and compute Q
 * @param[in] compu 'N' = don't compute U, 'I' = init and compute U, 'U' = update U0
 * @param[in] n Order of pencil (N >= 0, even)
 * @param[in,out] z N-by-N matrix Z, on exit Zout if JOB='T'
 * @param[in] ldz Leading dimension of Z (ldz >= max(1,n))
 * @param[in,out] b N/2-by-N/2 matrix B, on exit Bout if JOB='T'
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n/2))
 * @param[in,out] fg N/2-by-(N/2+1) array containing skew-symmetric G (lower) and F (upper)
 * @param[in] ldfg Leading dimension of FG (ldfg >= max(1,n/2))
 * @param[out] q N-by-N orthogonal transformation matrix Q
 * @param[in] ldq Leading dimension of Q (ldq >= max(1,n))
 * @param[in,out] u1 N/2-by-N/2 upper left block of orthogonal symplectic U
 * @param[in] ldu1 Leading dimension of U1
 * @param[in,out] u2 N/2-by-N/2 upper right block of orthogonal symplectic U
 * @param[in] ldu2 Leading dimension of U2
 * @param[out] alphar Real parts of eigenvalues, dimension N/2
 * @param[out] alphai Imaginary parts of eigenvalues, dimension N/2
 * @param[out] beta Denominators of eigenvalues, dimension N/2
 * @param[out] iwork Integer workspace, dimension LIWORK
 * @param[in] liwork Dimension of IWORK (liwork >= n+9)
 * @param[out] dwork Double workspace, dimension LDWORK
 * @param[in] ldwork Dimension of DWORK
 * @param[out] info Error indicator:
 *                  0 = success, <0 = param -info illegal,
 *                  1 = eigenvalue computation problem,
 *                  2 = periodic QZ did not converge,
 *                  3 = some eigenvalues may be inaccurate (warning)
 */
void mb04ed(const char *job, const char *compq, const char *compu,
            i32 n, f64 *z, i32 ldz, f64 *b, i32 ldb, f64 *fg, i32 ldfg,
            f64 *q, i32 ldq, f64 *u1, i32 ldu1, f64 *u2, i32 ldu2,
            f64 *alphar, f64 *alphai, f64 *beta,
            i32 *iwork, i32 liwork, f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Eigenvalues and orthogonal decomposition of a skew-Hamiltonian/
 *        skew-Hamiltonian pencil.
 *
 * Computes eigenvalues of aS - bT where:
 *   S = [[A, D], [E, A']] with D, E skew-symmetric
 *   T = [[B, F], [G, B']] with F, G skew-symmetric
 *
 * Optionally transforms to structured Schur form with orthogonal Q.
 *
 * @param[in] job 'E' = eigenvalues only, 'T' = also structured Schur form
 * @param[in] compq 'N' = don't compute Q, 'I' = initialize Q, 'U' = update Q0
 * @param[in] n Order of pencil (N >= 0, even)
 * @param[in,out] a N/2-by-N/2 matrix A, on exit Aout if JOB='T'
 * @param[in] lda Leading dimension of A (lda >= max(1,n/2))
 * @param[in,out] de N/2-by-(N/2+1) array with strictly lower E and strictly upper D
 * @param[in] ldde Leading dimension of DE (ldde >= max(1,n/2))
 * @param[in,out] b N/2-by-N/2 matrix B, on exit Bout if JOB='T'
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n/2))
 * @param[in,out] fg N/2-by-(N/2+1) array with strictly lower G and strictly upper F
 * @param[in] ldfg Leading dimension of FG (ldfg >= max(1,n/2))
 * @param[in,out] q N-by-N orthogonal transformation matrix Q
 * @param[in] ldq Leading dimension of Q (ldq >= 1 if compq='N', else ldq >= n)
 * @param[out] alphar Real parts of eigenvalues, dimension N/2
 * @param[out] alphai Imaginary parts of eigenvalues, dimension N/2
 * @param[out] beta Denominators of eigenvalues, dimension N/2
 * @param[out] iwork Integer workspace, dimension N/2+1. On exit, iwork[0] = count of
 *                   possibly inaccurate eigenvalues
 * @param[out] dwork Double workspace, dimension LDWORK
 * @param[in] ldwork Dimension of DWORK. Requirements depend on JOB and COMPQ.
 * @param[out] info Error indicator:
 *                  0 = success, <0 = param -info illegal,
 *                  1 = QZ iteration failed,
 *                  2 = numerically singular pencil (warning)
 */
void mb04fd(const char *job, const char *compq, i32 n,
            f64 *a, i32 lda, f64 *de, i32 ldde, f64 *b, i32 ldb, f64 *fg, i32 ldfg,
            f64 *q, i32 ldq, f64 *alphar, f64 *alphai, f64 *beta,
            i32 *iwork, f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Eigenvalues of skew-Hamiltonian/skew-Hamiltonian pencil (panel version).
 *
 * Computes eigenvalues of a real N-by-N skew-Hamiltonian/skew-Hamiltonian pencil
 * aS - bT with S = [[A, D], [E, A']] and T = [[B, F], [G, B']] where D, E, F, G
 * are skew-symmetric. This is a panel-based version of MB04FD for better
 * performance on large matrices.
 *
 * For small matrices (M <= 32) or when panel size equals M, delegates to MB04FD.
 *
 * @param[in] job 'E' = eigenvalues only, 'T' = also Schur form
 * @param[in] compq 'N' = don't compute Q, 'I' = initialize Q to identity, 'U' = update Q
 * @param[in] n Order of pencil (n >= 0, must be even)
 * @param[in,out] a N/2-by-N/2 matrix A. On exit if job='T': upper triangular Aout
 * @param[in] lda Leading dimension of A (lda >= max(1, n/2))
 * @param[in,out] de N/2-by-(N/2+1) with E (strictly lower in col 0) and D (strictly upper in cols 1:N/2)
 * @param[in] ldde Leading dimension of DE (ldde >= max(1, n/2))
 * @param[in,out] b N/2-by-N/2 matrix B. On exit if job='T': upper quasi-triangular Bout
 * @param[in] ldb Leading dimension of B (ldb >= max(1, n/2))
 * @param[in,out] fg N/2-by-(N/2+1) with G (strictly lower in col 0) and F (strictly upper in cols 1:N/2)
 * @param[in] ldfg Leading dimension of FG (ldfg >= max(1, n/2))
 * @param[in,out] q If compq='U': input Q0, output Q0*Q. If compq='I': output Q
 * @param[in] ldq Leading dimension of Q (ldq >= 1, ldq >= n if compq != 'N')
 * @param[out] alphar Real parts of eigenvalue numerators, dimension N/2
 * @param[out] alphai Imaginary parts of eigenvalue numerators, dimension N/2
 * @param[out] beta Eigenvalue denominators, dimension N/2
 * @param[out] iwork Integer workspace, dimension N/2+1
 * @param[out] dwork Double workspace, dimension LDWORK
 * @param[in] ldwork Dimension of DWORK. -1 for workspace query
 * @param[in,out] info On entry: desired panel size (<=0 for auto). On exit: error indicator
 */
void mb04fp(const char *job, const char *compq, i32 n,
            f64 *a, i32 lda, f64 *de, i32 ldde, f64 *b, i32 ldb, f64 *fg, i32 ldfg,
            f64 *q, i32 ldq, f64 *alphar, f64 *alphai, f64 *beta,
            i32 *iwork, f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief RQ factorization with row pivoting.
 *
 * Computes P*A = R*Q where P is a permutation matrix, R is upper triangular
 * (or trapezoidal if m >= n), and Q is orthogonal.
 *
 * @param[in] m Number of rows of A (m >= 0)
 * @param[in] n Number of columns of A (n >= 0)
 * @param[in,out] a M-by-N matrix. On exit, contains R and Householder reflectors
 * @param[in] lda Leading dimension of A (lda >= max(1,m))
 * @param[in,out] jpvt Pivot indices, dimension M.
 *                     On entry: if jpvt[i] != 0, row i is constrained to bottom.
 *                     On exit: jpvt[i] = k means row i of P*A was row k of A.
 * @param[out] tau Scalar factors of reflectors, dimension min(m,n)
 * @param[out] dwork Workspace, dimension 3*M
 * @param[out] info Error indicator: 0 = success, <0 = param -info illegal
 */
void mb04gd(i32 m, i32 n, f64 *a, i32 lda, i32 *jpvt, f64 *tau,
            f64 *dwork, i32 *info);

/**
 * @brief QR factorization of complex matrix with lower-left zero triangle.
 *
 * Computes A = Q * R where A is n-by-m complex matrix with a p-by-min(p,m)
 * zero triangle in the lower left corner. Optionally applies transformations
 * to an n-by-l matrix B from the left. Exploits the zero structure.
 *
 * @param[in] n Number of rows of A (n >= 0)
 * @param[in] m Number of columns of A (m >= 0)
 * @param[in] p Order of zero triangle (p >= 0)
 * @param[in] l Number of columns of B (l >= 0)
 * @param[in,out] a N-by-M complex matrix. On exit, upper triangle contains R,
 *                  elements below diagonal contain Householder vectors
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] b N-by-L complex matrix, transformed by Q^H. Not referenced if l=0
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n) if l>0, else ldb >= 1)
 * @param[out] tau Scalar factors of elementary reflectors, dimension min(n,m)
 * @param[out] zwork Complex workspace, dimension lzwork. On exit zwork[0] = optimal size
 * @param[in] lzwork Workspace size (lzwork >= max(1,m-1,m-p,l)). If -1, workspace query
 * @param[out] info Exit code: 0=success, <0=invalid parameter -info
 */
void mb04iz(i32 n, i32 m, i32 p, i32 l, c128 *a, i32 lda, c128 *b, i32 ldb,
            c128 *tau, c128 *zwork, i32 lzwork, i32 *info);

/**
 * @brief LQ factorization of matrix with upper-right zero triangle.
 *
 * Computes A = L * Q where A is n-by-m matrix with a min(n,p)-by-p
 * zero triangle in the upper right corner. Optionally applies transformations
 * to an l-by-m matrix B from the right. Exploits the zero structure.
 *
 * @param[in] n Number of rows of A (n >= 0)
 * @param[in] m Number of columns of A (m >= 0)
 * @param[in] p Order of zero triangle (p >= 0)
 * @param[in] l Number of rows of B (l >= 0)
 * @param[in,out] a N-by-M matrix. On exit, lower triangle contains L,
 *                  elements above diagonal contain Householder vectors
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] b L-by-M matrix, transformed by Q. Not referenced if l=0
 * @param[in] ldb Leading dimension of B (ldb >= max(1,l))
 * @param[out] tau Scalar factors of elementary reflectors, dimension min(n,m)
 * @param[out] dwork Workspace, dimension ldwork. On exit dwork[0] = optimal size
 * @param[in] ldwork Workspace size (ldwork >= max(1,n-1,n-p,l))
 * @param[out] info Exit code: 0=success, <0=invalid parameter -info
 */
void mb04jd(i32 n, i32 m, i32 p, i32 l, f64 *a, i32 lda, f64 *b, i32 ldb,
            f64 *tau, f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Eigenvalues of complex skew-Hamiltonian/Hamiltonian pencil.
 *
 * Computes eigenvalues of complex N-by-N skew-Hamiltonian/Hamiltonian pencil
 * aS - bH, with S = J*Z^H*J^T*Z and H = [[B, F], [G, -B^H]].
 *
 * Optionally transforms to structured Schur form with unitary Q and
 * unitary symplectic U.
 *
 * @param[in] job 'E' = eigenvalues only, 'T' = also structured Schur form
 * @param[in] compq 'N' = don't compute Q, 'C' = compute Q
 * @param[in] compu 'N' = don't compute U, 'C' = compute U
 * @param[in] n Order of pencil, n >= 0 and even
 * @param[in,out] z N-by-N complex matrix. On exit if job='T', contains BA
 * @param[in] ldz Leading dimension of z (ldz >= max(1,n))
 * @param[in,out] b M-by-M (job='E') or N-by-N (job='T') complex matrix
 * @param[in] ldb Leading dimension of b
 * @param[in,out] fg M-by-(M+1) (job='E') or N-by-N (job='T') complex matrix
 * @param[in] ldfg Leading dimension of fg
 * @param[out] d N-by-N complex matrix BD (if job='T')
 * @param[in] ldd Leading dimension of d
 * @param[out] c N-by-N complex matrix BC (if job='T')
 * @param[in] ldc Leading dimension of c
 * @param[out] q 2N-by-2N complex unitary matrix (if compq='C')
 * @param[in] ldq Leading dimension of q
 * @param[out] u N-by-2N complex unitary symplectic matrix (if compu='C')
 * @param[in] ldu Leading dimension of u
 * @param[out] alphar Real parts of eigenvalues, dimension n
 * @param[out] alphai Imaginary parts of eigenvalues, dimension n
 * @param[out] beta Scaling factors for eigenvalues, dimension n
 * @param[out] iwork Integer workspace, dimension liwork
 * @param[in] liwork Size of iwork (liwork >= 2*n+9)
 * @param[out] dwork Real workspace, dimension ldwork
 * @param[in] ldwork Size of dwork. If -1, workspace query
 * @param[out] zwork Complex workspace, dimension lzwork
 * @param[in] lzwork Size of zwork. If -1, workspace query
 * @param[out] bwork Logical workspace, dimension n (if job='T')
 * @param[out] info 0=success, <0=param -info invalid, 1-3=algorithm error
 */
void mb04az(const char *job, const char *compq, const char *compu,
            i32 n, c128 *z, i32 ldz, c128 *b, i32 ldb, c128 *fg, i32 ldfg,
            c128 *d, i32 ldd, c128 *c, i32 ldc, c128 *q, i32 ldq,
            c128 *u, i32 ldu, f64 *alphar, f64 *alphai, f64 *beta,
            i32 *iwork, i32 liwork, f64 *dwork, i32 ldwork,
            c128 *zwork, i32 lzwork, bool *bwork, i32 *info);

/**
 * @brief Reduce block (anti-)diagonal skew-Hamiltonian/Hamiltonian pencil
 *        to generalized Schur form.
 *
 * Computes the transformed matrices A and B, using orthogonal matrices Q1 and Q2
 * for a real N-by-N regular pencil:
 *
 *               ( A11   0  )     (  0   B12 )
 *   aA - bB = a (          ) - b (          ),
 *               (  0   A22 )     ( B21   0  )
 *
 * where A11, A22 and B12 are upper triangular, B21 is upper quasi-triangular
 * and the generalized matrix product A11^{-1} B12 A22^{-1} B21 is in periodic
 * Schur form, such that Q2' A Q1 is upper triangular, Q2' B Q1 is upper
 * quasi-triangular and the transformed pencil a(Q2' A Q1) - b(Q2' B Q1) is
 * in generalized Schur form.
 *
 * @param[in] compq1 'N' = don't compute Q1, 'I' = init to I and compute Q1,
 *                   'U' = update input Q1
 * @param[in] compq2 'N' = don't compute Q2, 'I' = init to I and compute Q2,
 *                   'U' = update input Q2
 * @param[in] n Order of pencil, n >= 0 and even
 * @param[in,out] a N-by-N matrix. On entry, block diagonal matrix. On exit,
 *                transformed upper triangular matrix.
 * @param[in] lda Leading dimension of a (lda >= max(1,n))
 * @param[in,out] b N-by-N matrix. On entry, block anti-diagonal matrix.
 *                On exit, transformed upper quasi-triangular matrix.
 * @param[in] ldb Leading dimension of b (ldb >= max(1,n))
 * @param[in,out] q1 N-by-N orthogonal transformation matrix
 * @param[in] ldq1 Leading dimension of q1
 * @param[in,out] q2 N-by-N orthogonal transformation matrix
 * @param[in] ldq2 Leading dimension of q2
 * @param[out] iwork Integer workspace, dimension liwork
 * @param[in] liwork Size of iwork (liwork >= max(n/2+1, 32))
 * @param[out] dwork Real workspace, dimension ldwork
 * @param[in] ldwork Size of dwork (ldwork >= 2*n*n + max(n/2+168, 272))
 *                   If -1, workspace query
 * @param[out] bwork Logical workspace, dimension n/2
 * @param[out] info 0=success, <0=param -info invalid, 1=MB03KD failed,
 *                  2=QZ failed in DGGES/DHGEQZ, 3=DTGEX2 reorder failed,
 *                  4=DTGSEN reorder failed
 */
void mb04hd(const char *compq1, const char *compq2,
            i32 n, f64 *a, i32 lda, f64 *b, i32 ldb,
            f64 *q1, i32 ldq1, f64 *q2, i32 ldq2,
            i32 *iwork, i32 liwork, f64 *dwork, i32 ldwork,
            bool *bwork, i32 *info);

/**
 * @brief Balances a general real matrix to reduce its 1-norm.
 *
 * To reduce the 1-norm of a general real matrix A by balancing.
 * This involves diagonal similarity transformations applied
 * iteratively to A to make the rows and columns as close in norm as
 * possible.
 *
 * @param[in] n Order of the matrix A (n >= 0)
 * @param[in,out] maxred On entry, max allowed reduction (>1 or <=0 for default 10).
 *                       On exit, ratio of original 1-norm to balanced 1-norm.
 * @param[in,out] a N-by-N matrix. On entry, input matrix. On exit, balanced matrix.
 * @param[in] lda Leading dimension of a (lda >= max(1,n))
 * @param[out] scale Scaling factors applied to A
 * @param[out] info 0=success, <0=param -info invalid
 */
void mb04md(i32 n, f64 *maxred, f64 *a, i32 lda, f64 *scale, i32 *info);

/**
 * @brief Reduce (skew-)Hamiltonian like matrix with orthogonal symplectic transformation.
 *
 * Reduces a Hamiltonian like matrix H = [A G; Q -A'] or skew-Hamiltonian like
 * matrix W = [A G; Q A'] so that elements below the (k+1)-th subdiagonal in
 * the first nb columns of A, and offdiagonal elements in the first nb columns
 * and rows of Q are zero. Returns matrices U, XA, XG, XQ, YA for blocked updates.
 *
 * This is an auxiliary routine called by MB04PB.
 *
 * @param[in] lham true for Hamiltonian (G=G', Q=Q'), false for skew-Hamiltonian (G=-G', Q=-Q')
 * @param[in] n Number of columns of A, n >= 0
 * @param[in] k Offset of reduction, k >= 0
 * @param[in] nb Number of columns/rows to reduce, n > nb >= 0
 * @param[in,out] a (k+n)-by-n matrix A, modified on output
 * @param[in] lda Leading dimension of a, lda >= max(1, k+n)
 * @param[in,out] qg (n+k)-by-(n+1) matrix containing Q (lower) and G (upper)
 * @param[in] ldqg Leading dimension of qg, ldqg >= max(1, n+k)
 * @param[out] xa n-by-(2*nb) matrix XA
 * @param[in] ldxa Leading dimension of xa, ldxa >= max(1, n)
 * @param[out] xg (k+n)-by-(2*nb) matrix XG
 * @param[in] ldxg Leading dimension of xg, ldxg >= max(1, k+n)
 * @param[out] xq n-by-(2*nb) matrix XQ
 * @param[in] ldxq Leading dimension of xq, ldxq >= max(1, n)
 * @param[out] ya (k+n)-by-(2*nb) matrix YA
 * @param[in] ldya Leading dimension of ya, ldya >= max(1, k+n)
 * @param[out] cs Cosines and sines of symplectic Givens rotations, dimension (2*nb)
 * @param[out] tau Scalar factors of elementary reflectors, dimension (nb)
 * @param[out] dwork Workspace, dimension (3*nb)
 */
void mb04pa(bool lham, i32 n, i32 k, i32 nb,
            f64 *a, i32 lda, f64 *qg, i32 ldqg,
            f64 *xa, i32 ldxa, f64 *xg, i32 ldxg,
            f64 *xq, i32 ldxq, f64 *ya, i32 ldya,
            f64 *cs, f64 *tau, f64 *dwork);

/**
 * @brief Reduce Hamiltonian matrix to Paige/Van Loan (PVL) form (blocked).
 *
 * Reduces a Hamiltonian matrix H = [A G; Q -A^T] where A is N-by-N and G,Q
 * are N-by-N symmetric matrices, to PVL form using blocked algorithm.
 * An orthogonal symplectic U is computed so that U^T H U has upper
 * Hessenberg A and diagonal Q.
 *
 * @param[in] n Order of matrix A, n >= 0
 * @param[in] ilo Starting index (1 <= ilo <= max(1,n) for n>0, ilo=1 for n=0)
 * @param[in,out] a N-by-N matrix. On entry A, on exit Aout (upper Hessenberg)
 *                  with reflector info in subdiagonal zeros
 * @param[in] lda Leading dimension of a, lda >= max(1,n)
 * @param[in,out] qg N-by-(N+1) matrix. On entry: Q (lower tri) and G (upper tri).
 *                   On exit: diagonal of Qout, upper tri of Gout, reflector info
 * @param[in] ldqg Leading dimension of qg, ldqg >= max(1,n)
 * @param[out] cs Cosines and sines of symplectic Givens rotations, dim (2*n-2)
 * @param[out] tau Scalar factors of elementary reflectors, dim (n-1)
 * @param[out] dwork Workspace. On exit dwork[0] = optimal ldwork
 * @param[in] ldwork Workspace size, ldwork >= max(1, n-1). Use -1 for query.
 * @param[out] info 0=success, <0=parameter -info invalid
 */
void mb04pb(i32 n, i32 ilo, f64 *a, i32 lda,
            f64 *qg, i32 ldqg, f64 *cs, f64 *tau,
            f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Reduce Hamiltonian matrix to Paige/Van Loan (PVL) form (unblocked).
 *
 * Reduces a Hamiltonian matrix H = [A G; Q -A^T] where A is N-by-N and G,Q
 * are N-by-N symmetric matrices, to PVL form. An orthogonal symplectic U
 * is computed so that U^T H U has upper Hessenberg A and diagonal Q.
 *
 * @param[in] n Order of matrix A, n >= 0
 * @param[in] ilo Starting index (1 <= ilo <= max(1,n) for n>0, ilo=1 for n=0)
 * @param[in,out] a N-by-N matrix. On entry A, on exit Aout (upper Hessenberg)
 *                  with reflector info in subdiagonal zeros
 * @param[in] lda Leading dimension of a, lda >= max(1,n)
 * @param[in,out] qg N-by-(N+1) matrix. On entry: Q (lower tri) and G (upper tri).
 *                   On exit: diagonal of Qout, upper tri of Gout, reflector info
 * @param[in] ldqg Leading dimension of qg, ldqg >= max(1,n)
 * @param[out] cs Cosines and sines of symplectic Givens rotations, dimension (2*n-2)
 * @param[out] tau Scalar factors of elementary reflectors, dimension (n-1)
 * @param[out] dwork Workspace. On exit dwork[0] = optimal ldwork
 * @param[in] ldwork Workspace size, ldwork >= max(1, n-1)
 * @param[out] info 0=success, <0=parameter -info invalid
 */
void mb04pu(i32 n, i32 ilo, f64 *a, i32 lda,
            f64 *qg, i32 ldqg, f64 *cs, f64 *tau,
            f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Multiply by product of symplectic reflectors and Givens rotations.
 *
 * Overwrites general real m-by-n matrices C and D with
 *   U * [op(C); op(D)]   if TRANU = 'N', or
 *   U^T * [op(C); op(D)] if TRANU = 'T',
 * where U is defined as the product of symplectic reflectors and
 * Givens rotations:
 *   U = diag(H(1),H(1)) G(1) diag(F(1),F(1))
 *       diag(H(2),H(2)) G(2) diag(F(2),F(2)) ...
 *       diag(H(k),H(k)) G(k) diag(F(k),F(k)),
 * with k = m-1, as returned by MB04PU or MB04RU.
 *
 * @param[in] tranc 'N' for op(C)=C, 'T'/'C' for op(C)=C^T
 * @param[in] trand 'N' for op(D)=D, 'T'/'C' for op(D)=D^T
 * @param[in] tranu 'N' to apply U, 'T' to apply U^T
 * @param[in] m Number of rows of op(C) and op(D), m >= 0
 * @param[in] n Number of columns of op(C) and op(D), n >= 0
 * @param[in] ilo Index from previous MB04PU/MB04RU call, 1 <= ilo <= m+1
 * @param[in] v M-by-M matrix containing reflector vectors H(i)
 * @param[in] ldv Leading dimension of v, ldv >= max(1,m)
 * @param[in] w M-by-M matrix containing reflector vectors F(i)
 * @param[in] ldw Leading dimension of w, ldw >= max(1,m)
 * @param[in,out] c Matrix C (M-by-N if TRANC='N', N-by-M if TRANC='T'/'C')
 * @param[in] ldc Leading dimension of c
 * @param[in,out] d Matrix D (M-by-N if TRAND='N', N-by-M if TRAND='T'/'C')
 * @param[in] ldd Leading dimension of d
 * @param[in] cs Cosines/sines of Givens rotations G(i), dimension (2*N-2)
 * @param[in] tau Scalar factors of reflectors F(i), dimension (N-1)
 * @param[out] dwork Workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size, ldwork >= max(1,n). Use -1 for query.
 * @param[out] info Exit code: 0=success, <0=invalid parameter -info
 */
void mb04qs(const char *tranc, const char *trand, const char *tranu,
            i32 m, i32 n, i32 ilo,
            const f64 *v, i32 ldv, const f64 *w, i32 ldw,
            f64 *c, i32 ldc, f64 *d, i32 ldd,
            const f64 *cs, const f64 *tau,
            f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Solve generalized real Sylvester equation with Schur form matrices.
 *
 * Solves the generalized real Sylvester equation:
 *   A * R - L * B = scale * C,
 *   D * R - L * E = scale * F,
 *
 * where R and L are unknown M-by-N matrices, and (A, D), (B, E) and
 * (C, F) are given matrix pairs. (A, D) and (B, E) must be in generalized
 * Schur canonical form, i.e., A, B are upper quasi-triangular and D, E
 * are upper triangular.
 *
 * The solution (R, L) overwrites (C, F). 0 <= scale <= 1 is an output
 * scaling factor chosen to avoid overflow.
 *
 * @param[in] m Order of A and D, row dimension of C, F, R, L. m >= 0.
 * @param[in] n Order of B and E, column dimension of C, F, R, L. n >= 0.
 * @param[in] pmax Upper bound for absolute value of solution elements. pmax >= 1.
 * @param[in] a M-by-M upper quasi-triangular matrix (Schur form)
 * @param[in] lda Leading dimension of a, lda >= max(1, m)
 * @param[in] b N-by-N upper quasi-triangular matrix (Schur form)
 * @param[in] ldb Leading dimension of b, ldb >= max(1, n)
 * @param[in,out] c M-by-N matrix. On entry, RHS of first equation.
 *                  On exit, solution R if info=0.
 * @param[in] ldc Leading dimension of c, ldc >= max(1, m)
 * @param[in] d M-by-M upper triangular matrix (Schur form)
 * @param[in] ldd Leading dimension of d, ldd >= max(1, m)
 * @param[in] e N-by-N upper triangular matrix (Schur form)
 * @param[in] lde Leading dimension of e, lde >= max(1, n)
 * @param[in,out] f M-by-N matrix. On entry, RHS of second equation.
 *                  On exit, solution L if info=0.
 * @param[in] ldf Leading dimension of f, ldf >= max(1, m)
 * @param[out] scale Output scaling factor, 0 <= scale <= 1
 * @param[out] iwork Integer workspace, dimension (m+n+2)
 * @param[out] info Exit code: 0=success, 1=element > pmax, 2=singular system
 */
void mb04rs(i32 m, i32 n, f64 pmax,
            const f64 *a, i32 lda, const f64 *b, i32 ldb,
            f64 *c, i32 ldc, const f64 *d, i32 ldd,
            const f64 *e, i32 lde, f64 *f, i32 ldf,
            f64 *scale, i32 *iwork, i32 *info);

/**
 * @brief Blocked solver for generalized real Sylvester equation using Level 3 BLAS.
 *
 * Solves the generalized real Sylvester equation:
 *     A * R - L * B = scale * C
 *     D * R - L * E = scale * F
 *
 * where (A,D) and (B,E) are in generalized Schur form (A,B upper quasi-triangular,
 * D,E upper triangular). Solution overwrites (C,F). Early termination if
 * any element exceeds PMAX.
 *
 * This is the blocked version of MB04RS using Level 3 BLAS.
 *
 * @param[in] m Order of A and D, row dimension of C/F. m >= 0
 * @param[in] n Order of B and E, column dimension of C/F. n >= 0
 * @param[in] pmax Upper bound for solution elements. pmax >= 1.0
 * @param[in] a M-by-M upper quasi-triangular matrix (Schur form)
 * @param[in] lda Leading dimension of a, lda >= max(1, m)
 * @param[in] b N-by-N upper quasi-triangular matrix (Schur form)
 * @param[in] ldb Leading dimension of b, ldb >= max(1, n)
 * @param[in,out] c M-by-N matrix. On entry, RHS of first equation.
 *                  On exit, solution R if info=0.
 * @param[in] ldc Leading dimension of c, ldc >= max(1, m)
 * @param[in] d M-by-M upper triangular matrix (Schur form)
 * @param[in] ldd Leading dimension of d, ldd >= max(1, m)
 * @param[in] e N-by-N upper triangular matrix (Schur form)
 * @param[in] lde Leading dimension of e, lde >= max(1, n)
 * @param[in,out] f M-by-N matrix. On entry, RHS of second equation.
 *                  On exit, solution L if info=0.
 * @param[in] ldf Leading dimension of f, ldf >= max(1, m)
 * @param[out] scale Output scaling factor, 0 <= scale <= 1
 * @param[out] iwork Integer workspace, dimension (m+n+6)
 * @param[out] info Exit code: 0=success, 1=element > pmax, 2=singular system
 */
void mb04rt(i32 m, i32 n, f64 pmax,
            const f64 *a, i32 lda, const f64 *b, i32 ldb,
            f64 *c, i32 ldc, const f64 *d, i32 ldd,
            const f64 *e, i32 lde, f64 *f, i32 ldf,
            f64 *scale, i32 *iwork, i32 *info);

/**
 * @brief Reduce skew-Hamiltonian matrix to PVL form (unblocked).
 *
 * Reduces a skew-Hamiltonian matrix W to Paige/Van Loan (PVL) form:
 *
 *           [  A   G  ]              [  Aout  Gout  ]
 *     W  =  [       T ]   ->   U'WU = [            T ]
 *           [  Q   A  ]              [    0   Aout  ]
 *
 * where Aout is upper Hessenberg, G and Q are skew-symmetric, and
 * U is orthogonal symplectic.
 *
 * @param[in] n Order of matrix A. n >= 0.
 * @param[in] ilo Lower index of non-triangular block. 1 <= ilo <= n+1 (n>0).
 * @param[in,out] a N-by-N matrix. On exit, upper Hessenberg Aout with
 *                  reflector info below subdiagonal.
 * @param[in] lda Leading dimension of a, lda >= max(1,n).
 * @param[in,out] qg N-by-(N+1) array. On entry, cols 1:N contain strictly
 *                   lower triangular Q, cols 2:N+1 contain strictly upper
 *                   triangular G. On exit, reflector info and Gout.
 * @param[in] ldqg Leading dimension of qg, ldqg >= max(1,n).
 * @param[out] cs Cosines/sines of Givens rotations, dimension (2*N-2).
 * @param[out] tau Scalar factors of reflectors F(i), dimension (N-1).
 * @param[out] dwork Workspace, dimension (ldwork).
 * @param[in] ldwork Workspace size, ldwork >= max(1,n-1).
 * @param[out] info Exit code: 0=success, <0=invalid parameter -info.
 */
void mb04ru(i32 n, i32 ilo,
            f64 *a, i32 lda,
            f64 *qg, i32 ldqg,
            f64 *cs, f64 *tau,
            f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Reduce skew-Hamiltonian matrix to PVL form (blocked).
 *
 * Reduces a skew-Hamiltonian matrix W to Paige/Van Loan (PVL) form:
 *
 *           [  A   G  ]              [  Aout  Gout  ]
 *     W  =  [       T ]   ->   U'WU = [            T ]
 *           [  Q   A  ]              [    0   Aout  ]
 *
 * where Aout is upper Hessenberg, G and Q are skew-symmetric, and
 * U is orthogonal symplectic. This is the blocked version of MB04RU.
 *
 * @param[in] n Order of matrix A. n >= 0.
 * @param[in] ilo Lower index of non-triangular block. 1 <= ilo <= n+1 (n>0).
 * @param[in,out] a N-by-N matrix. On exit, upper Hessenberg Aout with
 *                  reflector info below subdiagonal.
 * @param[in] lda Leading dimension of a, lda >= max(1,n).
 * @param[in,out] qg N-by-(N+1) array. On entry, cols 1:N contain strictly
 *                   lower triangular Q, cols 2:N+1 contain strictly upper
 *                   triangular G. On exit, reflector info and Gout.
 * @param[in] ldqg Leading dimension of qg, ldqg >= max(1,n).
 * @param[out] cs Cosines/sines of Givens rotations, dimension (2*N-2).
 * @param[out] tau Scalar factors of reflectors F(i), dimension (N-1).
 * @param[out] dwork Workspace, dimension (ldwork). Returns optimal size.
 * @param[in] ldwork Workspace size, ldwork >= max(1,n-1). If -1, query mode.
 * @param[out] info Exit code: 0=success, <0=invalid parameter -info.
 */
void mb04rb(i32 n, i32 ilo,
            f64 *a, i32 lda,
            f64 *qg, i32 ldqg,
            f64 *cs, f64 *tau,
            f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Eigenvalues of complex skew-Hamiltonian/Hamiltonian pencil.
 *
 * Computes eigenvalues of a complex N-by-N skew-Hamiltonian/Hamiltonian
 * pencil aS - bH, with S = [[A, D], [E, A^H]] and H = [[B, F], [G, -B^H]],
 * using an embedding to a real skew-Hamiltonian/skew-Hamiltonian pencil.
 *
 * @param[in] job 'E' = eigenvalues only, 'T' = also compute Schur form
 * @param[in] compq 'N' = don't compute Q, 'C' = compute Q
 * @param[in] n Order of pencil, n >= 0, must be even
 * @param[in,out] a On entry: N/2-by-N/2 matrix A. On exit if job='T': N-by-N upper triangular BA
 * @param[in] lda Leading dimension of a, lda >= max(1, k) where k=N/2 if job='E', k=N if job='T'
 * @param[in,out] de On entry: N/2-by-(N/2+1) with E (lower tri) and D (upper tri in cols 2:N/2+1).
 *                   On exit if job='T': N-by-N skew-Hermitian BD
 * @param[in] ldde Leading dimension of de, ldde >= max(1, k)
 * @param[in,out] b On entry: N/2-by-N/2 matrix B. On exit if job='T': N-by-N upper triangular BB
 * @param[in] ldb Leading dimension of b, ldb >= max(1, k)
 * @param[in,out] fg On entry: N/2-by-(N/2+1) with G (lower tri) and F (upper tri in cols 2:N/2+1).
 *                   On exit if job='T': N-by-N Hermitian BF
 * @param[in] ldfg Leading dimension of fg, ldfg >= max(1, k)
 * @param[out] q If compq='C': 2N-by-2N unitary transformation matrix
 * @param[in] ldq Leading dimension of q, ldq >= 1, ldq >= max(1, 2*n) if compq='C'
 * @param[out] alphar Real parts of eigenvalue numerators, dimension (n)
 * @param[out] alphai Imaginary parts of eigenvalue numerators, dimension (n)
 * @param[out] beta Eigenvalue denominators, dimension (n)
 * @param[out] iwork Integer workspace, dimension (2*n+4)
 * @param[out] dwork Real workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size. Min: 4*n*n+3*n if job='E' compq='N', 5*n*n+3*n if job='T' compq='N',
 *                   11*n*n+2*n if compq='C'. -1 for query.
 * @param[out] zwork Complex workspace, dimension (lzwork)
 * @param[in] lzwork Complex workspace size. Min: 1 if job='E', 6*n+4 if job='T' compq='N',
 *                   8*n+4 if job='T' compq='C'. -1 for query.
 * @param[out] bwork Logical workspace, dimension (0 if job='E', n if job='T')
 * @param[out] info 0=success, 1=MB04FD QZ failed, 2=ZHGEQZ failed, 3=pencil numerically singular
 */
void mb04bz(const char *job, const char *compq, i32 n,
            c128 *a, i32 lda, c128 *de, i32 ldde, c128 *b, i32 ldb,
            c128 *fg, i32 ldfg, c128 *q, i32 ldq,
            f64 *alphar, f64 *alphai, f64 *beta,
            i32 *iwork, f64 *dwork, i32 ldwork,
            c128 *zwork, i32 lzwork, bool *bwork, i32 *info);

/**
 * @brief Givens transformation with interchange (modified DROT).
 *
 * Applies a row-permuted Givens transformation:
 *
 *     |X(i)|    | 0   1 |   | C   S |   |X(i)|
 *     |    | := |       | x |       | x |    |, i = 1,...N.
 *     |Y(i)|    | 1   0 |   |-S   C |   |Y(i)|
 *
 * This computes: X_new = C*Y - S*X, Y_new = C*X + S*Y
 *
 * NOTE: This is NOT standard DROT which computes X' = CX + SY, Y' = -SX + CY.
 *
 * @param[in] n Number of elements to transform, n >= 0
 * @param[in,out] x Array of dimension at least (1+(n-1)*|incx|)
 * @param[in] incx Increment for x
 * @param[in,out] y Array of dimension at least (1+(n-1)*|incy|)
 * @param[in] incy Increment for y
 * @param[in] c Cosine of Givens rotation
 * @param[in] s Sine of Givens rotation
 */
void mb04tu(i32 n, f64 *x, i32 incx, f64 *y, i32 incy, f64 c, f64 s);

/**
 * @brief Reduce submatrix A(k) to upper triangular form using column Givens rotations.
 *
 * Reduces A(k) = A(IFIRA:ma, IFICA:na) where ma = IFIRA-1+NRA, na = IFICA-1+NCA
 * to upper triangular form using column Givens rotations only.
 * Matrix A(k) is assumed to have full row rank.
 *
 * The same column transformations are applied to E(k) = E(1:IFIRA-1, IFICA:na).
 * Note: E uses the same column indices but different row indices than A.
 *
 * @param[in] updatz If true, accumulate column transformations in Z
 * @param[in] n Number of columns of A and E, n >= 0
 * @param[in] nra Number of rows in A to be transformed, 0 <= NRA <= LDA
 * @param[in] nca Number of columns in A to be transformed, 0 <= NCA <= N
 * @param[in] ifira Index of first row in A to be transformed (1-based)
 * @param[in] ifica Index of first column in A to be transformed (1-based)
 * @param[in,out] a On entry: submatrix A(k) with full row rank
 *                  On exit: transformed matrix with A(k) upper triangular
 * @param[in] lda Leading dimension of A, lda >= max(1, NRA)
 * @param[in,out] e On entry: submatrix E(k) = E(1:IFIRA-1, IFICA:na)
 *                  On exit: transformed E matrix
 * @param[in] lde Leading dimension of E, lde >= max(1, IFIRA-1)
 * @param[in,out] z If updatz=true: on entry contains given matrix,
 *                  on exit contains Z * (column transformation matrix)
 * @param[in] ldz Leading dimension of Z. If updatz=true, ldz >= max(1,n); else ldz >= 1
 * @param[out] info 0 = success
 */
void mb04tv(bool updatz, i32 n, i32 nra, i32 nca, i32 ifira, i32 ifica,
            f64 *a, i32 lda, f64 *e, i32 lde, f64 *z, i32 ldz, i32 *info);

/**
 * @brief Reduce submatrix E(k) to upper triangular form using row Givens rotations.
 *
 * Reduces E(k) = E(IFIRE:me, IFICE:ne) where me = IFIRE-1+NRE, ne = IFICE-1+NCE
 * to upper triangular form using row Givens rotations only.
 * Matrix E(k) is assumed to have full column rank.
 *
 * The same row transformations are applied to A(k) = A(IFIRE:me, IFICA:N).
 * Note: A uses the same row indices but different column indices than E.
 *
 * @param[in] updatq If true, accumulate row transformations in Q
 * @param[in] m Number of rows of A and E, m >= 0
 * @param[in] n Number of columns of A and E, n >= 0
 * @param[in] nre Number of rows in E to be transformed, 0 <= NRE <= M
 * @param[in] nce Number of columns in E to be transformed, 0 <= NCE <= N
 * @param[in] ifire Index of first row in E to be transformed (1-based)
 * @param[in] ifice Index of first column in E to be transformed (1-based)
 * @param[in] ifica Index of first column in A to be transformed (1-based)
 * @param[in,out] a On entry: submatrix A(k) = A(IFIRE:me, IFICA:N)
 *                  On exit: transformed A matrix
 * @param[in] lda Leading dimension of A, lda >= max(1, M)
 * @param[in,out] e On entry: submatrix E(k) with full column rank
 *                  On exit: transformed matrix with E(k) upper triangular
 * @param[in] lde Leading dimension of E, lde >= max(1, M)
 * @param[in,out] q If updatq=true: on entry contains given matrix,
 *                  on exit contains Q * (row transformation matrix)^T
 * @param[in] ldq Leading dimension of Q. If updatq=true, ldq >= max(1,M); else ldq >= 1
 * @param[out] info 0 = success
 */
void mb04tw(bool updatq, i32 m, i32 n, i32 nre, i32 nce, i32 ifire, i32 ifice,
            i32 ifica, f64 *a, i32 lda, f64 *e, i32 lde, f64 *q, i32 ldq,
            i32 *info);

/**
 * @brief Triangularize full rank submatrices in staircase pencil.
 *
 * Performs triangularization of submatrices having full row and column rank
 * in the pencil s*E(eps,inf)-A(eps,inf) using Algorithm 3.3.1 from Beelen's thesis.
 *
 * On entry, matrices A and E are assumed to have been transformed to generalized
 * Schur form and the pencil s*E(eps,inf)-A(eps,inf) is in staircase form.
 *
 * @param[in] updatq If true, accumulate row transformations in Q
 * @param[in] updatz If true, accumulate column transformations in Z
 * @param[in] m Number of rows of A and E. m >= 0
 * @param[in] n Number of columns of A and E. n >= 0
 * @param[in] nblcks Number of submatrices with full row rank (possibly 0) in A(eps,inf)
 * @param[in] inuk Array of dimension nblcks with row dimensions nu(k)
 * @param[in] imuk Array of dimension nblcks with column dimensions mu(k)
 * @param[in,out] a M-by-N matrix A. On exit: transformed matrix
 * @param[in] lda Leading dimension of A. lda >= max(1,m)
 * @param[in,out] e M-by-N matrix E. On exit: transformed matrix
 * @param[in] lde Leading dimension of E. lde >= max(1,m)
 * @param[in,out] q If updatq=true: M-by-M matrix updated with row transformations
 * @param[in] ldq Leading dimension of Q. If updatq=true, ldq >= max(1,m); else ldq >= 1
 * @param[in,out] z If updatz=true: N-by-N matrix updated with column transformations
 * @param[in] ldz Leading dimension of Z. If updatz=true, ldz >= max(1,n); else ldz >= 1
 * @param[out] info 0 = success;
 *                  1 = incorrect dimensions of full column rank submatrix (mu(k+1) > nu(k));
 *                  2 = incorrect dimensions of full row rank submatrix (nu(k) > mu(k))
 */
void mb04ty(bool updatq, bool updatz, i32 m, i32 n, i32 nblcks,
            const i32 *inuk, const i32 *imuk, f64 *a, i32 lda,
            f64 *e, i32 lde, f64 *q, i32 ldq, f64 *z, i32 ldz, i32 *info);

/**
 * @brief Separate pencils s*E(eps)-A(eps) and s*E(inf)-A(inf).
 *
 * Separates the epsilon and infinite parts of the pencil s*E(eps,inf)-A(eps,inf)
 * in staircase form using Algorithm 3.3.3 from Beelen's thesis.
 *
 * On entry, matrices A and E are assumed to be in staircase form from prior
 * algorithms (MB04UD, etc.). On exit, the pencil is separated into:
 * - s*E(eps)-A(eps): contains all Kronecker column indices
 * - s*E(inf)-A(inf): contains all infinite elementary divisors
 *
 * @param[in] updatq If true, accumulate row transformations in Q
 * @param[in] updatz If true, accumulate column transformations in Z
 * @param[in] m Number of rows of A and E. m >= 0
 * @param[in] n Number of columns of A and E. n >= 0
 * @param[in,out] nblcks On entry: number of full row rank submatrices in A(eps,inf).
 *                       On exit: may be reduced by 1 if last block becomes empty.
 * @param[in,out] inuk Array of dimension nblcks. On entry: row dimensions nu(k).
 *                     On exit: row dimensions of s*E(eps)-A(eps) submatrices.
 * @param[in,out] imuk Array of dimension nblcks. On entry: column dimensions mu(k).
 *                     On exit: column dimensions of s*E(eps)-A(eps) submatrices.
 * @param[in,out] a M-by-N matrix A. On exit: transformed matrix.
 * @param[in] lda Leading dimension of A. lda >= max(1,m)
 * @param[in,out] e M-by-N matrix E. On exit: transformed matrix.
 * @param[in] lde Leading dimension of E. lde >= max(1,m)
 * @param[in,out] q If updatq=true: M-by-M matrix updated with row transformations.
 * @param[in] ldq Leading dimension of Q. If updatq=true, ldq >= max(1,m); else ldq >= 1.
 * @param[in,out] z If updatz=true: N-by-N matrix updated with column transformations.
 * @param[in] ldz Leading dimension of Z. If updatz=true, ldz >= max(1,n); else ldz >= 1.
 * @param[out] mnei Array of dimension 4:
 *                  mnei[0] = meps = row dimension of s*E(eps)-A(eps)
 *                  mnei[1] = neps = column dimension of s*E(eps)-A(eps)
 *                  mnei[2] = minf = row dimension of s*E(inf)-A(inf)
 *                  mnei[3] = ninf = column dimension of s*E(inf)-A(inf)
 */
void mb04tx(bool updatq, bool updatz, i32 m, i32 n, i32 *nblcks,
            i32 *inuk, i32 *imuk, f64 *a, i32 lda, f64 *e, i32 lde,
            f64 *q, i32 ldq, f64 *z, i32 ldz, i32 *mnei);

/**
 * @brief Upper block triangular form for rectangular pencil sE-A.
 *
 * Computes orthogonal transformations Q and Z such that the transformed
 * pencil Q'(sE-A)Z is in upper block triangular form, where E is an M-by-N
 * matrix in column echelon form and A is an M-by-N matrix.
 *
 * MODE='B': Basic reduction to staircase form (1).
 * MODE='T': Triangularization of full rank submatrices in (1).
 * MODE='S': Separation of epsilon and infinite parts as in (2).
 *
 * @param[in] mode 'B', 'T', or 'S' - specifies desired structure.
 * @param[in] jobq 'N', 'I', or 'U' - how to handle Q matrix.
 * @param[in] jobz 'N', 'I', or 'U' - how to handle Z matrix.
 * @param[in] m Number of rows in A, E (order of Q). m >= 0.
 * @param[in] n Number of columns in A, E (order of Z). n >= 0.
 * @param[in] ranke Rank of matrix E in column echelon form. ranke >= 0.
 * @param[in,out] a M-by-N matrix A. On exit: transformed matrix.
 * @param[in] lda Leading dimension of A, lda >= max(1,m).
 * @param[in,out] e M-by-N matrix E in column echelon form. On exit: transformed.
 * @param[in] lde Leading dimension of E, lde >= max(1,m).
 * @param[in,out] q M-by-M orthogonal transformation matrix.
 * @param[in] ldq Leading dimension of Q. ldq >= max(1,m) if jobq='I'/'U', else 1.
 * @param[in,out] z N-by-N orthogonal transformation matrix.
 * @param[in] ldz Leading dimension of Z. ldz >= max(1,n) if jobz='I'/'U', else 1.
 * @param[in,out] istair Array of dimension m with column echelon info.
 *                       istair[i]=+j if E(i,j) is corner point, -j otherwise.
 * @param[out] nblcks Number of full row rank submatrices detected.
 * @param[out] nblcki Number of diagonal submatrices in sE(inf)-A(inf) (mode='S').
 * @param[out] imuk Array of dimension max(n,m+1). Column dimensions mu(k).
 * @param[out] inuk Array of dimension max(n,m+1). Row dimensions nu(k).
 * @param[out] imuk0 Array. If mode='S', dimensions of sE(inf)-A(inf) diagonals.
 * @param[out] mnei Array of dimension 3 with dimension info.
 * @param[in] tol Tolerance for zero elements. If <= 0, computed automatically.
 * @param[out] iwork Workspace array of dimension n.
 * @param[out] info 0=success; <0=-i means i-th argument invalid; >0=algorithm error.
 */
void mb04vd(const char *mode, const char *jobq, const char *jobz,
            i32 m, i32 n, i32 ranke,
            f64 *a, i32 lda, f64 *e, i32 lde,
            f64 *q, i32 ldq, f64 *z, i32 ldz,
            i32 *istair, i32 *nblcks, i32 *nblcki,
            i32 *imuk, i32 *inuk, i32 *imuk0, i32 *mnei,
            f64 tol, i32 *iwork, i32 *info);

/**
 * @brief Separate epsilon and infinite pencils from staircase form.
 *
 * Separates s*E(eps)-A(eps) and s*E(inf)-A(inf) in the pencil
 * s*E(eps,inf)-A(eps,inf). Similar to MB04TX but NBLCKS is input-only
 * and MNEI has 3 elements.
 *
 * @param[in] updatq Whether to accumulate row transformations in Q.
 * @param[in] updatz Whether to accumulate column transformations in Z.
 * @param[in] m Number of rows in A and E.
 * @param[in] n Number of columns in A and E.
 * @param[in] nblcks Number of full-row-rank submatrices in A(eps,inf).
 * @param[in,out] inuk Array of row dimensions nu(k) of full-row-rank submatrices.
 *                     On exit: dimensions for s*E(eps)-A(eps).
 * @param[in,out] imuk Array of column dimensions mu(k) of full-column-rank submatrices.
 *                     On exit: dimensions for s*E(eps)-A(eps).
 * @param[in,out] a M-by-N matrix A to be reduced.
 * @param[in] lda Leading dimension of A, lda >= max(1,m).
 * @param[in,out] e M-by-N matrix E to be reduced.
 * @param[in] lde Leading dimension of E, lde >= max(1,m).
 * @param[in,out] q Row transformation matrix. Updated if updatq=true.
 * @param[in] ldq Leading dimension of Q.
 * @param[in,out] z Column transformation matrix. Updated if updatz=true.
 * @param[in] ldz Leading dimension of Z.
 * @param[out] mnei Output array of dimension 3:
 *                  mnei[0] = meps = row dimension of s*E(eps)-A(eps);
 *                  mnei[1] = neps = column dimension of s*E(eps)-A(eps);
 *                  mnei[2] = minf = order of s*E(inf)-A(inf).
 */
void mb04vx(bool updatq, bool updatz, i32 m, i32 n, i32 nblcks,
            i32 *inuk, i32 *imuk, f64 *a, i32 lda, f64 *e, i32 lde,
            f64 *q, i32 ldq, f64 *z, i32 ldz, i32 *mnei);

/**
 * @brief Compute symplectic QR decomposition of a real 2M-by-N matrix [A; B].
 *
 * Computes [A; B] = Q * R where Q is symplectic orthogonal, R11 is upper
 * triangular, and R21 is strictly upper triangular. Unblocked version.
 *
 * Q = diag(H(1),H(1)) G(1) diag(F(1),F(1)) ... diag(H(k),H(k)) G(k) diag(F(k),F(k))
 * where k = min(m,n), H(i) and F(i) are Householder reflectors, and G(i) are
 * Givens rotations.
 *
 * @param[in] m Number of rows in A and B, m >= 0
 * @param[in] n Number of columns in A and B, n >= 0
 * @param[in,out] a On entry: M-by-N matrix A
 *                  On exit: [R11 R12] and reflector info in zero parts
 * @param[in] lda Leading dimension of A, lda >= max(1,m)
 * @param[in,out] b On entry: M-by-N matrix B
 *                  On exit: [R21 R22] and reflector info in zero parts
 * @param[in] ldb Leading dimension of B, ldb >= max(1,m)
 * @param[out] cs Array of dimension 2*min(m,n) containing cosines and sines
 *                of Givens rotations. CS(2*i-1) = cosine, CS(2*i) = sine.
 * @param[out] tau Array of dimension min(m,n) containing scalar factors of
 *                 elementary reflectors F(i)
 * @param[out] dwork Workspace array of dimension ldwork
 * @param[in] ldwork Workspace size, ldwork >= max(1,n)
 * @param[out] info 0 = success, < 0 = -i means i-th argument invalid
 */
void mb04su(i32 m, i32 n, f64 *a, i32 lda, f64 *b, i32 ldb,
            f64 *cs, f64 *tau, f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Reduce pencil sE-A to column echelon form with orthogonal Q and Z.
 *
 * Computes orthogonal transformations Q and Z such that the transformed
 * pencil Q'(sE-A)Z has the E matrix in column echelon form (trapezoidal).
 *
 * Column echelon form: first (N-r) columns are zero, and in remaining
 * columns the last nonzero element has strictly increasing row index.
 *
 * @param[in] jobq 'N' = don't form Q, 'I' = init to I and compute Q,
 *                 'U' = update input Q
 * @param[in] jobz 'N' = don't form Z, 'I' = init to I and compute Z,
 *                 'U' = update input Z
 * @param[in] m Number of rows in A and E, order of Q. m >= 0
 * @param[in] n Number of columns in A and E, order of Z. n >= 0
 * @param[in,out] a M-by-N matrix. On entry, matrix A of pencil sE-A.
 *                  On exit, transformed Q'*A*Z.
 * @param[in] lda Leading dimension of A, lda >= max(1,m)
 * @param[in,out] e M-by-N matrix. On entry, matrix E of pencil sE-A.
 *                  On exit, transformed Q'*E*Z in column echelon form.
 * @param[in] lde Leading dimension of E, lde >= max(1,m)
 * @param[in,out] q M-by-M orthogonal matrix. If jobq='U', on entry contains
 *                  initial Q. On exit (jobq='I' or 'U'), accumulated transformations.
 * @param[in] ldq Leading dimension of Q. If jobq='I' or 'U', ldq >= max(1,m);
 *                else ldq >= 1.
 * @param[in,out] z N-by-N orthogonal matrix. If jobz='U', on entry contains
 *                  initial Z. On exit (jobz='I' or 'U'), accumulated transformations.
 * @param[in] ldz Leading dimension of Z. If jobz='I' or 'U', ldz >= max(1,n);
 *                else ldz >= 1.
 * @param[out] ranke Computed rank of the transformed matrix E.
 * @param[out] istair Integer array of dimension M. ISTAIR(i) = +j if E(i,j) is
 *                    a corner point, -j if E(i,j) is on boundary but not corner.
 * @param[in] tol Tolerance for zero elements. If tol <= 0, use eps*||E||_max.
 * @param[out] dwork Workspace array of dimension max(m,n)
 * @param[out] info 0=success, <0=param -info invalid
 */
void mb04ud(const char *jobq, const char *jobz, i32 m, i32 n,
            f64 *a, i32 lda, f64 *e, i32 lde,
            f64 *q, i32 ldq, f64 *z, i32 ldz,
            i32 *ranke, i32 *istair, f64 tol, f64 *dwork, i32 *info);

/**
 * @brief Solve the generalized complex Sylvester equation.
 *
 * Solves the generalized complex Sylvester equation:
 *
 *     A * R - L * B = scale * C
 *     D * R - L * E = scale * F
 *
 * where R and L are unknown M-by-N matrices, and (A, D), (B, E) and (C, F)
 * are given matrix pairs of size M-by-M, N-by-N and M-by-N, respectively.
 * A, B, D and E are complex upper triangular (i.e., (A,D) and (B,E) are in
 * generalized Schur form).
 *
 * The solution (R, L) overwrites (C, F). 0 <= SCALE <= 1 is an output scaling
 * factor chosen to avoid overflow.
 *
 * This routine is intended to be called only by SLICOT Library routine MB04RW.
 * For efficiency purposes, the computations are aborted when the absolute
 * value of an element of R or L is greater than a given value PMAX.
 *
 * @param[in] m Order of matrices A and D, and row dimension of C, F, R and L.
 *              m >= 0.
 * @param[in] n Order of matrices B and E, and column dimension of C, F, R and L.
 *              n >= 0.
 * @param[in] pmax Upper bound for the "absolute value" of solution elements.
 *                 pmax >= 1.0.
 * @param[in] a M-by-M upper triangular complex matrix in generalized Schur form.
 * @param[in] lda Leading dimension of a. lda >= max(1, m).
 * @param[in] b N-by-N upper triangular complex matrix in generalized Schur form.
 * @param[in] ldb Leading dimension of b. ldb >= max(1, n).
 * @param[in,out] c On entry: M-by-N right-hand-side of first equation.
 *                  On exit: M-by-N solution R.
 * @param[in] ldc Leading dimension of c. ldc >= max(1, m).
 * @param[in] d M-by-M upper triangular complex matrix in generalized Schur form.
 *              Diagonal elements are non-negative real.
 * @param[in] ldd Leading dimension of d. ldd >= max(1, m).
 * @param[in] e N-by-N upper triangular complex matrix in generalized Schur form.
 *              Diagonal elements are non-negative real.
 * @param[in] lde Leading dimension of e. lde >= max(1, n).
 * @param[in,out] f On entry: M-by-N right-hand-side of second equation.
 *                  On exit: M-by-N solution L.
 * @param[in] ldf Leading dimension of f. ldf >= max(1, m).
 * @param[out] scale Scaling factor (0 <= scale <= 1). Normally scale = 1.
 * @param[out] info 0: success; 1: element of R or L exceeded PMAX;
 *                  2: (A,D) and (B,E) have common or close eigenvalues.
 */
void mb04rv(i32 m, i32 n, f64 pmax,
            const c128 *a, i32 lda, const c128 *b, i32 ldb,
            c128 *c, i32 ldc, const c128 *d, i32 ldd,
            const c128 *e, i32 lde, c128 *f, i32 ldf,
            f64 *scale, i32 *info);

/**
 * @brief Blocked complex generalized Sylvester equation solver (Level 3 BLAS).
 *
 * Solves the generalized complex Sylvester equation:
 *
 *     A * R - L * B = scale * C
 *     D * R - L * E = scale * F
 *
 * using Level 3 BLAS, where R and L are unknown M-by-N matrices, and (A, D),
 * (B, E) and (C, F) are given matrix pairs of size M-by-M, N-by-N and M-by-N,
 * respectively. A, B, D and E are complex upper triangular (i.e., (A,D) and
 * (B,E) are in generalized Schur form).
 *
 * The solution (R, L) overwrites (C, F). 0 <= SCALE <= 1 is an output scaling
 * factor chosen to avoid overflow.
 *
 * This routine is intended to be called only by SLICOT Library routine MB04RZ.
 * For efficiency purposes, the computations are aborted when the absolute
 * value of an element of R or L is greater than a given value PMAX.
 *
 * @param[in] m Order of matrices A and D, and row dimension of C, F, R and L.
 *              m >= 0.
 * @param[in] n Order of matrices B and E, and column dimension of C, F, R and L.
 *              n >= 0.
 * @param[in] pmax Upper bound for the "absolute value" of solution elements.
 *                 pmax >= 1.0.
 * @param[in] a M-by-M upper triangular complex matrix in generalized Schur form.
 * @param[in] lda Leading dimension of a. lda >= max(1, m).
 * @param[in] b N-by-N upper triangular complex matrix in generalized Schur form.
 * @param[in] ldb Leading dimension of b. ldb >= max(1, n).
 * @param[in,out] c On entry: M-by-N right-hand-side of first equation.
 *                  On exit: M-by-N solution R.
 * @param[in] ldc Leading dimension of c. ldc >= max(1, m).
 * @param[in] d M-by-M upper triangular complex matrix in generalized Schur form.
 *              Diagonal elements are non-negative real.
 * @param[in] ldd Leading dimension of d. ldd >= max(1, m).
 * @param[in] e N-by-N upper triangular complex matrix in generalized Schur form.
 *              Diagonal elements are non-negative real.
 * @param[in] lde Leading dimension of e. lde >= max(1, n).
 * @param[in,out] f On entry: M-by-N right-hand-side of second equation.
 *                  On exit: M-by-N solution L.
 * @param[in] ldf Leading dimension of f. ldf >= max(1, m).
 * @param[out] scale Scaling factor (0 <= scale <= 1). Normally scale = 1.
 * @param[out] iwork Integer workspace array of dimension (m+n+2).
 * @param[out] info 0: success; 1: element of R or L exceeded PMAX;
 *                  2: (A,D) and (B,E) have common or close eigenvalues.
 */
void mb04rw(i32 m, i32 n, f64 pmax,
            const c128 *a, i32 lda, const c128 *b, i32 ldb,
            c128 *c, i32 ldc, const c128 *d, i32 ldd,
            const c128 *e, i32 lde, c128 *f, i32 ldf,
            f64 *scale, i32 *iwork, i32 *info);

/**
 * @brief Row compression with column echelon form preservation.
 *
 * Transforms submatrices (AA, EE) of A and E such that Aj is row compressed
 * while keeping Ej in column echelon form. This is step j of Algorithm 3.2.1
 * from Beelen's thesis for computing Kronecker structure of matrix pencils.
 *
 * Let AA and EE be the following submatrices of A and E:
 *   AA := A(IFIRA : M ; IFICA : N)
 *   EE := E(IFIRA : M ; IFICA : N)
 * Let Aj and Ej be the following submatrices:
 *   Aj := A(IFIRA : M ; IFICA : IFICA + NCA - 1)
 *   Ej := E(IFIRA : M ; IFICA + NCA : N)
 *
 * @param[in] updatq If true, update Q with row transformations
 * @param[in] updatz If true, update Z with column transformations
 * @param[in] m Number of rows in A, E, and Q. m >= 0.
 * @param[in] n Number of columns in A, E, and Z. n >= 0.
 * @param[in] ifira First row index of submatrices Aj and Ej (1-based).
 * @param[in] ifica First column index of submatrices (1-based).
 * @param[in] nca Number of columns in submatrix Aj.
 * @param[in,out] a M-by-N matrix. On entry, contains Aj in submatrix position.
 *                  On exit, contains row-compressed result.
 * @param[in] lda Leading dimension of A. lda >= max(1, m).
 * @param[in,out] e M-by-N matrix. On entry, Ej is in column echelon form.
 *                  On exit, Ej remains in column echelon form (may differ).
 * @param[in] lde Leading dimension of E. lde >= max(1, m).
 * @param[in,out] q M-by-M orthogonal matrix. Updated if updatq is true.
 * @param[in] ldq Leading dimension of Q. If updatq, ldq >= max(1, m); else ldq >= 1.
 * @param[in,out] z N-by-N orthogonal matrix. Updated if updatz is true.
 * @param[in] ldz Leading dimension of Z. If updatz, ldz >= max(1, n); else ldz >= 1.
 * @param[in,out] istair Integer array of dimension M. Encodes column echelon form:
 *                       +j: E(i,j) is a corner point; -j: boundary but not corner.
 * @param[out] rank Numerical rank of submatrix Aj based on tol.
 * @param[in] tol Tolerance for zero elements.
 * @param[out] iwork Integer workspace of dimension N.
 */
void mb04tt(bool updatq, bool updatz, i32 m, i32 n,
            i32 ifira, i32 ifica, i32 nca,
            f64 *a, i32 lda, f64 *e, i32 lde,
            f64 *q, i32 ldq, f64 *z, i32 ldz,
            i32 *istair, i32 *rank, f64 tol,
            i32 *iwork);

/**
 * @brief Generate matrix Q with orthogonal columns from symplectic reflectors and Givens rotations
 *
 * Generates a matrix Q with orthogonal columns (spanning an isotropic subspace),
 * which is defined as the first n columns of a product of symplectic reflectors
 * and Givens rotations:
 *
 *     Q = diag(H(1),H(1)) G(1) diag(F(1),F(1))
 *         diag(H(2),H(2)) G(2) diag(F(2),F(2))
 *                         ....
 *         diag(H(k),H(k)) G(k) diag(F(k),F(k)).
 *
 * The matrix Q is returned in terms of its first 2*M rows:
 *
 *                  [  op(Q1)   op(Q2) ]
 *              Q = [                  ]
 *                  [ -op(Q2)   op(Q1) ]
 *
 * @param[in] tranq1 If true, op(Q1) = Q1'; if false, op(Q1) = Q1.
 * @param[in] tranq2 If true, op(Q2) = Q2'; if false, op(Q2) = Q2.
 * @param[in] m Number of rows of Q1 and Q2. m >= 0.
 * @param[in] n Number of columns of Q1 and Q2. m >= n >= 0.
 * @param[in] k Number of symplectic Givens rotations. n >= k >= 0.
 * @param[in,out] q1 On entry (tranq1=false), M-by-K part contains reflector F(i) vectors.
 *                   On entry (tranq1=true), K-by-M part contains reflector F(i) vectors.
 *                   On exit (tranq1=false), M-by-N part contains matrix Q1.
 *                   On exit (tranq1=true), N-by-M part contains matrix Q1'.
 * @param[in] ldq1 Leading dimension of Q1. If tranq1=false, ldq1 >= max(1,m);
 *                 if tranq1=true, ldq1 >= max(1,n).
 * @param[in,out] q2 On entry (tranq2=false), M-by-K part contains reflector H(i) vectors
 *                   with scalar factors on diagonal.
 *                   On entry (tranq2=true), K-by-M part contains reflector H(i) vectors
 *                   with scalar factors on diagonal.
 *                   On exit (tranq2=false), M-by-N part contains matrix Q2.
 *                   On exit (tranq2=true), N-by-M part contains matrix Q2'.
 * @param[in] ldq2 Leading dimension of Q2. If tranq2=false, ldq2 >= max(1,m);
 *                 if tranq2=true, ldq2 >= max(1,n).
 * @param[in] cs Array of dimension 2*k containing cosines and sines of
 *               symplectic Givens rotations G(i). cs[2*i-2] = cos, cs[2*i-1] = sin.
 * @param[in] tau Array of dimension k containing scalar factors of
 *                elementary reflectors F(i).
 * @param[out] dwork Workspace array of dimension ldwork. On exit, dwork[0]
 *                   contains optimal ldwork. If info = -13, dwork[0] = minimum ldwork.
 * @param[in] ldwork Length of dwork. ldwork >= max(1, m+n).
 * @param[out] info 0 = success; < 0 = -i means i-th argument had illegal value.
 */
void mb04wu(bool tranq1, bool tranq2, i32 m, i32 n, i32 k,
            f64 *q1, i32 ldq1, f64 *q2, i32 ldq2,
            const f64 *cs, const f64 *tau, f64 *dwork,
            i32 ldwork, i32 *info);

/**
 * @brief Apply Householder transformations from bidiagonalization.
 *
 * Applies the Householder transformations Pj stored in factored form
 * into the columns of array X, to desired columns of matrix U by
 * premultiplication, and/or the Householder transformations Qj stored
 * in factored form into the rows of array X, to desired columns of
 * matrix V by premultiplication. The Householder transformations Pj
 * and Qj are stored as produced by LAPACK routine DGEBRD.
 *
 * @param[in] jobu 'N' = don't transform U;
 *                 'A' = transform U (U has M columns);
 *                 'S' = transform U (U has min(M,N) columns).
 * @param[in] jobv 'N' = don't transform V;
 *                 'A' = transform V (V has N columns);
 *                 'S' = transform V (V has min(M,N) columns).
 * @param[in] m Number of rows of matrix X. m >= 0.
 * @param[in] n Number of columns of matrix X. n >= 0.
 * @param[in,out] x M-by-N matrix containing Householder transformations
 *                  Pj in columns of lower triangle and Qj in rows of
 *                  upper triangle (as produced by DGEBRD).
 *                  Modified during computation but restored on exit.
 * @param[in] ldx Leading dimension of X. ldx >= max(1,m).
 * @param[in] taup Scalar factors of Householder transformations Pj.
 *                 Array of dimension min(m,n).
 * @param[in] tauq Scalar factors of Householder transformations Qj.
 *                 Array of dimension min(m,n).
 * @param[in,out] u On entry: M-by-M (jobu='A') or M-by-min(M,N) (jobu='S')
 *                  matrix U. On exit: Pj applied to columns where inul[i]=true.
 *                  Not referenced if jobu='N'.
 * @param[in] ldu Leading dimension of U. ldu >= max(1,m) if jobu='A'/'S';
 *                ldu >= 1 if jobu='N'.
 * @param[in,out] v On entry: N-by-N (jobv='A') or N-by-min(M,N) (jobv='S')
 *                  matrix V. On exit: Qj applied to columns where inul[i]=true.
 *                  Not referenced if jobv='N'.
 * @param[in] ldv Leading dimension of V. ldv >= max(1,n) if jobv='A'/'S';
 *                ldv >= 1 if jobv='N'.
 * @param[in] inul Boolean array of dimension max(m,n). inul[i]=true if
 *                 column i of U and/or V is to be transformed.
 * @param[out] info 0 = success; < 0 = -i means i-th argument had illegal value.
 */
void mb04xy(const char *jobu, const char *jobv, i32 m, i32 n,
            f64 *x, i32 ldx, const f64 *taup, const f64 *tauq,
            f64 *u, i32 ldu, f64 *v, i32 ldv,
            const bool *inul, i32 *info);

/**
 * @brief Compute left/right singular subspace basis for smallest singular values.
 *
 * Computes a basis for the left and/or right singular subspace of an M-by-N
 * matrix A corresponding to its smallest singular values.
 *
 * Uses the Partial Singular Value Decomposition (PSVD) algorithm which is
 * more efficient than full SVD when only a singular subspace is needed.
 *
 * @param[in] jobu Specifies left singular subspace computation:
 *                 'N' = do not compute left singular subspace;
 *                 'A' = return (M-RANK) M-dimensional base vectors in U;
 *                 'S' = return first (min(M,N)-RANK) base vectors in U.
 * @param[in] jobv Specifies right singular subspace computation:
 *                 'N' = do not compute right singular subspace;
 *                 'A' = return (N-RANK) N-dimensional base vectors in V;
 *                 'S' = return first (min(M,N)-RANK) base vectors in V.
 * @param[in] m Number of rows in matrix A. m >= 0.
 * @param[in] n Number of columns in matrix A. n >= 0.
 * @param[in,out] rank On entry: if < 0, compute rank as number of singular values > THETA;
 *                     otherwise, specifies the desired rank.
 *                     On exit: computed or adjusted rank.
 * @param[in,out] theta On entry: if rank < 0, upper bound on smallest singular values;
 *                      if rank >= 0, initial estimate for computing upper bound.
 *                      On exit: if rank >= 0 on entry, the computed upper bound.
 * @param[in,out] a M-by-N matrix A. Destroyed on exit. Dimension (lda,n).
 * @param[in] lda Leading dimension of A. lda >= max(1,m).
 * @param[out] u If jobu='A': M-by-M matrix containing (M-RANK) base vectors.
 *               If jobu='S': M-by-min(M,N) matrix with first (min(M,N)-RANK) vectors.
 *               Vectors stored in columns where inul[i]=true.
 * @param[in] ldu Leading dimension of U. ldu >= max(1,m) if jobu!='N'; else >= 1.
 * @param[out] v If jobv='A': N-by-N matrix containing (N-RANK) base vectors.
 *               If jobv='S': N-by-min(M,N) matrix with first (min(M,N)-RANK) vectors.
 *               Vectors stored in columns where inul[i]=true.
 * @param[in] ldv Leading dimension of V. ldv >= max(1,n) if jobv!='N'; else >= 1.
 * @param[out] q Partially diagonalized bidiagonal matrix. Dimension (2*min(M,N)-1).
 *               q[0:p-1] contains diagonal; q[p:2p-2] contains superdiagonal.
 * @param[out] inul Boolean array of dimension max(m,n). True entries indicate
 *                  columns of U/V containing base vectors.
 * @param[in] tol Tolerance for singular value multiplicity and negligible elements.
 *                If <= 0, default from MB04YD is used.
 * @param[in] reltol Minimum relative interval width for bisection.
 *                   If < BASE*EPS, BASE*EPS is used.
 * @param[out] dwork Workspace array of dimension ldwork.
 * @param[in] ldwork Workspace size. See formula in routine documentation.
 *                   ldwork = -1: workspace query.
 * @param[out] iwarn 0 = no warning; 1 = rank lowered due to singular value multiplicity.
 * @param[out] info 0 = success; < 0 = -i means i-th arg invalid;
 *                  1 = max QR/QL iterations (30*min(M,N)) exceeded.
 */
void mb04xd(const char *jobu, const char *jobv, i32 m, i32 n,
            i32 *rank, f64 *theta, f64 *a, i32 lda,
            f64 *u, i32 ldu, f64 *v, i32 ldv,
            f64 *q, bool *inul, f64 tol, f64 reltol,
            f64 *dwork, i32 ldwork, i32 *iwarn, i32 *info);

/**
 * @brief Block-diagonalization of generalized real Schur form.
 *
 * Reduces a matrix pair (A,B) in generalized real Schur form to block-diagonal
 * form using well-conditioned non-orthogonal equivalence transformations.
 * The condition numbers of the transformations are roughly bounded by PMAX.
 * Optionally reorders diagonal blocks so clustered eigenvalues are grouped.
 *
 * @param[in] jobx 'N' = don't accumulate left transformations;
 *                 'U' = accumulate left transformations in X.
 * @param[in] joby 'N' = don't accumulate right transformations;
 *                 'U' = accumulate right transformations in Y.
 * @param[in] sort 'N' = no reordering;
 *                 'S' = reorder blocks to cluster eigenvalues;
 *                 'C' = closest-neighbour strategy, no reordering;
 *                 'B' = closest-neighbour with reordering.
 * @param[in] n Order of matrices A, B, X, Y. n >= 0.
 * @param[in] pmax Upper bound for transformation element magnitudes. pmax >= 1.
 * @param[in,out] a On entry: N-by-N upper quasi-triangular matrix A.
 *                  On exit: block-diagonal matrix in real Schur canonical form.
 * @param[in] lda Leading dimension of A. lda >= max(1,n).
 * @param[in,out] b On entry: N-by-N upper triangular matrix B with non-negative diagonal.
 *                  On exit: upper triangular block-diagonal matrix.
 * @param[in] ldb Leading dimension of B. ldb >= max(1,n).
 * @param[in,out] x On entry (if jobx='U'): N-by-N transformation matrix.
 *                  On exit (if jobx='U'): updated with left transformation.
 * @param[in] ldx Leading dimension of X. ldx >= 1; ldx >= n if jobx='U'.
 * @param[in,out] y On entry (if joby='U'): N-by-N transformation matrix.
 *                  On exit (if joby='U'): updated with right transformation.
 * @param[in] ldy Leading dimension of Y. ldy >= 1; ldy >= n if joby='U'.
 * @param[out] nblcks Number of diagonal blocks.
 * @param[out] blsize Array of dimension N with block sizes (first nblcks elements used).
 * @param[out] alphar Array of dimension N with real parts of eigenvalue numerators.
 * @param[out] alphai Array of dimension N with imag parts of eigenvalue numerators.
 * @param[out] beta Array of dimension N with eigenvalue denominators (non-negative).
 * @param[in] tol Tolerance for eigenvalue clustering (if sort='S' or 'B').
 *                tol > 0: absolute tolerance; tol < 0: relative tolerance;
 *                tol = 0: default relative tolerance sqrt(sqrt(eps)).
 * @param[out] iwork Integer workspace array of dimension N+6.
 * @param[out] dwork Workspace array of dimension ldwork.
 * @param[in] ldwork Length of dwork. ldwork >= 1 if n <= 1; >= 4*n+16 if n > 1.
 *                   ldwork = -1: workspace query.
 * @param[out] info 0 = success; < 0 = -i means i-th arg had illegal value;
 *                  1 = singular pencil.
 */
void mb04rd(const char *jobx, const char *joby, const char *sort, i32 n,
            f64 pmax, f64 *a, i32 lda, f64 *b, i32 ldb, f64 *x, i32 ldx,
            f64 *y, i32 ldy, i32 *nblcks, i32 *blsize, f64 *alphar,
            f64 *alphai, f64 *beta, f64 tol, i32 *iwork, f64 *dwork,
            i32 ldwork, i32 *info);

/**
 * @brief Block-diagonalize a complex pencil in generalized Schur form.
 *
 * Reduces a complex matrix pair (A,B) in generalized complex Schur form to
 * block-diagonal form using well-conditioned non-unitary equivalence
 * transformations. The condition numbers of the transformations are roughly
 * bounded by PMAX. Transformations are optionally accumulated in X and Y.
 * The Schur form may be optionally reordered to group clustered eigenvalues.
 *
 * @param[in] jobx 'N' = don't accumulate left transforms; 'U' = accumulate in X.
 * @param[in] joby 'N' = don't accumulate right transforms; 'U' = accumulate in Y.
 * @param[in] sort 'N' = no reordering; 'S' = reorder to cluster eigenvalues;
 *                 'C' = closest-neighbour strategy; 'B' = both reorder and closest.
 * @param[in] n Order of matrices A, B, X, Y. n >= 0.
 * @param[in] pmax Upper bound for transformation element magnitudes. pmax >= 1.0.
 * @param[in,out] a On entry: N-by-N upper triangular complex matrix A (Schur form).
 *                  On exit: block-diagonal matrix.
 * @param[in] lda Leading dimension of A. lda >= max(1,n).
 * @param[in,out] b On entry: N-by-N upper triangular complex matrix B (Schur form).
 *                  Diagonal elements are real non-negative.
 *                  On exit: block-diagonal matrix with real non-negative diagonal.
 * @param[in] ldb Leading dimension of B. ldb >= max(1,n).
 * @param[in,out] x On entry (if jobx='U'): N-by-N transformation matrix.
 *                  On exit (if jobx='U'): updated with left transformation.
 * @param[in] ldx Leading dimension of X. ldx >= 1; ldx >= n if jobx='U'.
 * @param[in,out] y On entry (if joby='U'): N-by-N transformation matrix.
 *                  On exit (if joby='U'): updated with right transformation.
 * @param[in] ldy Leading dimension of Y. ldy >= 1; ldy >= n if joby='U'.
 * @param[out] nblcks Number of diagonal blocks.
 * @param[out] blsize Array of dimension N with block sizes (first nblcks elements used).
 * @param[out] alpha Array of dimension N with complex eigenvalue numerators.
 * @param[out] beta Array of dimension N with complex eigenvalue denominators
 *                  (real non-negative, imaginary parts are zero).
 * @param[in] tol Tolerance for eigenvalue clustering (if sort='S' or 'B').
 *                tol > 0: absolute tolerance; tol < 0: relative tolerance;
 *                tol = 0: default relative tolerance sqrt(sqrt(eps)).
 * @param[out] iwork Integer workspace array of dimension N+2.
 * @param[out] info 0 = success; < 0 = -i means i-th arg had illegal value;
 *                  1 = singular pencil.
 */
void mb04rz(const char *jobx, const char *joby, const char *sort, i32 n,
            f64 pmax, c128 *a, i32 lda, c128 *b, i32 ldb, c128 *x, i32 ldx,
            c128 *y, i32 ldy, i32 *nblcks, i32 *blsize, c128 *alpha,
            c128 *beta, f64 tol, i32 *iwork, i32 *info);

/**
 * @brief Generate orthogonal matrix Q spanning isotropic subspace (blocked version).
 *
 * MB04WD generates a matrix Q with orthogonal columns (spanning an
 * isotropic subspace), which is defined as the first n columns
 * of a product of symplectic reflectors and Givens rotations.
 * This is the blocked version of MB04WU.
 *
 * @param[in] tranq1 If false: op(Q1) = Q1; if true: op(Q1) = Q1'.
 * @param[in] tranq2 If false: op(Q2) = Q2; if true: op(Q2) = Q2'.
 * @param[in] m Number of rows of Q1 and Q2. m >= 0.
 * @param[in] n Number of columns of Q1 and Q2. m >= n >= 0.
 * @param[in] k Number of symplectic Givens rotations. n >= k >= 0.
 * @param[in,out] q1 On entry (tranq1=false), M-by-K part contains reflector F(i) vectors.
 *                   On entry (tranq1=true), K-by-M part contains reflector F(i) vectors.
 *                   On exit (tranq1=false), M-by-N part contains matrix Q1.
 *                   On exit (tranq1=true), N-by-M part contains matrix Q1'.
 * @param[in] ldq1 Leading dimension of Q1. If tranq1=false, ldq1 >= max(1,m);
 *                 if tranq1=true, ldq1 >= max(1,n).
 * @param[in,out] q2 On entry (tranq2=false), M-by-K part contains reflector H(i) vectors
 *                   with scalar factors on diagonal.
 *                   On entry (tranq2=true), K-by-M part contains reflector H(i) vectors
 *                   with scalar factors on diagonal.
 *                   On exit (tranq2=false), M-by-N part contains matrix Q2.
 *                   On exit (tranq2=true), N-by-M part contains matrix Q2'.
 * @param[in] ldq2 Leading dimension of Q2. If tranq2=false, ldq2 >= max(1,m);
 *                 if tranq2=true, ldq2 >= max(1,n).
 * @param[in] cs Array of dimension 2*k containing cosines and sines of
 *               symplectic Givens rotations G(i). cs[2*i-2] = cos, cs[2*i-1] = sin.
 * @param[in] tau Array of dimension k containing scalar factors of
 *                elementary reflectors F(i).
 * @param[out] dwork Workspace array of dimension ldwork. On exit, dwork[0]
 *                   contains optimal ldwork. If info = -13, dwork[0] = minimum ldwork.
 * @param[in] ldwork Length of dwork. ldwork >= max(1, m+n).
 *                   ldwork = -1: workspace query.
 * @param[out] info 0 = success; < 0 = -i means i-th argument had illegal value.
 */
void mb04wd(bool tranq1, bool tranq2, i32 m, i32 n, i32 k,
            f64 *q1, i32 ldq1, f64 *q2, i32 ldq2,
            const f64 *cs, const f64 *tau, f64 *dwork,
            i32 ldwork, i32 *info);

/**
 * @brief Generate orthogonal symplectic matrix from MB04PU output.
 *
 * Generates an orthogonal symplectic matrix U defined as a product of
 * symplectic reflectors and Givens rotations as returned by MB04PU.
 * The matrix U is returned in terms of its first N rows:
 *     U = [[U1, U2], [-U2, U1]]
 *
 * @param[in] n Order of matrices U1 and U2. n >= 0.
 * @param[in] ilo Same value as in previous MB04PU call. U is identity
 *                except in submatrix U([ilo+1:n n+ilo+1:2n], [ilo+1:n n+ilo+1:2n]).
 *                1 <= ilo <= n if n > 0; ilo = 1 if n = 0.
 * @param[in,out] u1 N-by-N matrix. On entry: i-th column contains vector
 *                   defining elementary reflector F(i). On exit: U1.
 * @param[in] ldu1 Leading dimension of u1. ldu1 >= max(1,n).
 * @param[in,out] u2 N-by-N matrix. On entry: i-th column contains vector
 *                   defining reflector H(i) and subdiagonal scalar of H(i).
 *                   On exit: U2.
 * @param[in] ldu2 Leading dimension of u2. ldu2 >= max(1,n).
 * @param[in] cs Array of 2n-2 elements: cosines and sines of symplectic
 *               Givens rotations G(i).
 * @param[in] tau Array of n-1 elements: scalar factors of reflectors F(i).
 * @param[out] dwork Workspace of dimension ldwork. On exit: dwork[0] = optimal ldwork.
 * @param[in] ldwork Workspace length. ldwork >= max(1, 2*(n-ilo)).
 *                   ldwork = -1: workspace query.
 * @param[out] info 0 = success; < 0 = -i means i-th argument had illegal value.
 */
void mb04wp(i32 n, i32 ilo, f64 *u1, i32 ldu1, f64 *u2, i32 ldu2,
            const f64 *cs, const f64 *tau, f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Perform one QR or QL iteration step on bidiagonal submatrix.
 *
 * Performs either one QR or QL iteration step onto the unreduced
 * bidiagonal submatrix Jk (from index l to k) of a bidiagonal matrix J.
 * The submatrix Jk is transformed to S' Jk T where S and T are products
 * of Givens rotations. Optionally accumulates these rotations into U and V.
 *
 * @param[in] qrit If true: QR iteration (chase bulge top to bottom);
 *                 if false: QL iteration (chase bulge bottom to top).
 * @param[in] updatu If true: accumulate left rotations S into U.
 * @param[in] updatv If true: accumulate right rotations T into V.
 * @param[in] m Number of rows of matrix U. m >= 0.
 * @param[in] n Number of rows of matrix V. n >= 0.
 * @param[in] l Index of first diagonal entry of submatrix (1-based). l >= 1.
 * @param[in] k Index of last diagonal entry of submatrix (1-based). k <= p.
 * @param[in] shift Shift value for QR/QL iteration step.
 * @param[in,out] d Diagonal entries of bidiagonal matrix J, dimension p=min(m,n).
 *                  On exit: diagonal of transformed matrix S' J T.
 * @param[in,out] e Superdiagonal entries of J, dimension p-1.
 *                  On exit: superdiagonal of transformed matrix S' J T.
 * @param[in,out] u M-by-p left transformation matrix (if updatu=true).
 *                  On exit: U * S is returned.
 * @param[in] ldu Leading dimension of U. ldu >= max(1,m) if updatu, else ldu >= 1.
 * @param[in,out] v N-by-p right transformation matrix (if updatv=true).
 *                  On exit: V * T is returned.
 * @param[in] ldv Leading dimension of V. ldv >= max(1,n) if updatv, else ldv >= 1.
 * @param[out] dwork Workspace. Size: 4*p-4 if both updatu and updatv;
 *                   2*p-2 if only one; 1 if neither.
 */
void mb04yw(bool qrit, bool updatu, bool updatv,
            i32 m, i32 n, i32 l, i32 k, f64 shift,
            f64 *d, f64 *e, f64 *u, i32 ldu, f64 *v, i32 ldv, f64 *dwork);

/**
 * @brief Partial diagonalization of a bidiagonal matrix.
 *
 * Partially diagonalizes a bidiagonal matrix J using QR or QL iterations
 * such that J is split into unreduced bidiagonal submatrices whose singular
 * values are either all larger than a given bound or all smaller than (or
 * equal to) this bound.
 *
 * @param[in] jobu 'N': do not form U; 'I': initialize U to identity and
 *                 accumulate left rotations; 'U': update given matrix U.
 * @param[in] jobv 'N': do not form V; 'I': initialize V to identity and
 *                 accumulate right rotations; 'U': update given matrix V.
 * @param[in] m Number of rows in matrix U. m >= 0.
 * @param[in] n Number of rows in matrix V. n >= 0.
 * @param[in,out] rank On entry: if < 0, compute rank as number of singular
 *                     values > theta; else specifies desired rank. rank <= min(m,n).
 *                     On exit: computed or adjusted rank.
 * @param[in,out] theta On entry: if rank < 0, upper bound for smallest singular
 *                      values (theta >= 0); else initial estimate (may be < 0
 *                      to let routine compute). On exit: if rank >= 0 on entry,
 *                      computed upper bound such that rank singular values > theta.
 * @param[in,out] q Diagonal elements of J. Length min(m,n).
 * @param[in,out] e Superdiagonal elements of J. Length min(m,n)-1.
 * @param[in,out] u M-by-min(M,N) matrix for left Givens rotations.
 *                  Not referenced if jobu='N'.
 * @param[in] ldu Leading dimension of U. ldu >= max(1,m) if jobu != 'N', else >= 1.
 * @param[in,out] v N-by-min(M,N) matrix for right Givens rotations.
 *                  Not referenced if jobv='N'.
 * @param[in] ldv Leading dimension of V. ldv >= max(1,n) if jobv != 'N', else >= 1.
 * @param[in,out] inul Length min(m,n). On entry: inul[i]=true if column i of
 *                     U/V already contains computed base vector. On exit:
 *                     inul[i]=true indicates diagonal entry i belongs to
 *                     submatrix with singular values <= theta.
 * @param[in] tol Tolerance for negligible elements and singular value
 *                multiplicity. If <= 0, taken as eps * max(|q|, |e|).
 * @param[in] reltol Relative tolerance for interval width in bisection.
 *                   If < base*eps, taken as base*eps.
 * @param[out] dwork Workspace. Length >= max(1, 6*min(m,n)-5) if jobu or jobv
 *                   is 'I' or 'U'; else >= max(1, 4*min(m,n)-3).
 * @param[in] ldwork Workspace length.
 * @param[out] iwarn 0: no warning; 1: rank lowered due to singular value
 *                   multiplicity.
 * @param[out] info 0: success; < 0: -i means i-th argument illegal;
 *                  1: exceeded 30*min(m,n) QR/QL iterations.
 */
void mb04yd(const char *jobu, const char *jobv, i32 m, i32 n,
            i32 *rank, f64 *theta, f64 *q, f64 *e, f64 *u, i32 ldu,
            f64 *v, i32 ldv, bool *inul, f64 tol, f64 reltol,
            f64 *dwork, i32 ldwork, i32 *iwarn, i32 *info);

/**
 * @brief Generate orthogonal symplectic matrices U or V from symplectic reflectors.
 *
 * Generates orthogonal symplectic matrices U or V, defined as products of
 * symplectic reflectors and Givens rotations, as returned by MB04TS or MB04TB.
 *
 * The matrices U and V are returned in terms of their first N/2 rows:
 *     U = [[U1, U2], [-U2, U1]]
 *     V = [[V1, V2], [-V2, V1]]
 *
 * @param[in] job 'U': generate U; 'V': generate V.
 * @param[in] trans Must match TRANA (if job='U') or TRANB (if job='V')
 *                  from prior MB04TS/MB04TB call.
 *                  'N': no transpose; 'T'/'C': transpose.
 * @param[in] n Order of matrices Q1 and Q2. n >= 0.
 * @param[in] ilo Must match ILO from prior MB04TS/MB04TB call.
 *                1 <= ilo <= n if n > 0; ilo = 1 if n = 0.
 * @param[in,out] q1 N-by-N matrix. On entry: reflector vectors FU(i) or FV(i).
 *                   On exit: U1, U1^T, V1^T, or V1 depending on job/trans.
 * @param[in] ldq1 Leading dimension of Q1. ldq1 >= max(1,n).
 * @param[in,out] q2 N-by-N matrix. On entry: reflector vectors HU(i) or HV(i).
 *                   On exit: U2 or V2^T depending on job.
 * @param[in] ldq2 Leading dimension of Q2. ldq2 >= max(1,n).
 * @param[in] cs Cosines and sines of Givens rotations. Length 2n for job='U',
 *               2n-2 for job='V'.
 * @param[in] tau Scalar factors of reflectors. Length n for job='U',
 *                n-1 for job='V'.
 * @param[out] dwork Workspace of dimension ldwork. On exit: dwork[0] = optimal ldwork.
 * @param[in] ldwork Workspace length. ldwork >= max(1, 2*(n-ilo+1)).
 *                   ldwork = -1: workspace query.
 * @param[out] info 0 = success; < 0 = -i means i-th argument had illegal value.
 */
void mb04wr(const char *job, const char *trans, i32 n, i32 ilo,
            f64 *q1, i32 ldq1, f64 *q2, i32 ldq2,
            const f64 *cs, const f64 *tau, f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Apply inverse balancing transformation to complex skew-Hamiltonian/Hamiltonian eigenvectors.
 *
 * Applies from the left the inverse of a balancing transformation, computed by
 * MB4DPZ, to the complex matrix [[V1], [sgn*V2]] where sgn is +1 or -1.
 *
 * @param[in] job Transformation type:
 *                'N': do nothing;
 *                'P': inverse permutation only;
 *                'S': inverse scaling only;
 *                'B': both permutation and scaling.
 * @param[in] sgn Sign for V2: 'P' for +1, 'N' for -1.
 * @param[in] n Number of rows of V1 and V2. n >= 0.
 * @param[in] ilo Index from MB4DPZ. 1 <= ilo <= n+1.
 * @param[in] lscale Permutation and scaling factors for left, dimension n.
 * @param[in] rscale Permutation and scaling factors for right, dimension n.
 * @param[in] m Number of columns of V1 and V2. m >= 0.
 * @param[in,out] v1 Complex N-by-M matrix V1. Modified in place.
 * @param[in] ldv1 Leading dimension of V1. ldv1 >= max(1,n).
 * @param[in,out] v2 Complex N-by-M matrix V2. Modified in place.
 * @param[in] ldv2 Leading dimension of V2. ldv2 >= max(1,n).
 * @param[out] info 0: success; < 0: -i means i-th argument illegal.
 */
void mb4dbz(const char *job, const char *sgn, i32 n, i32 ilo,
            const f64 *lscale, const f64 *rscale, i32 m,
            c128 *v1, i32 ldv1, c128 *v2, i32 ldv2, i32 *info);

/**
 * @brief Balance a complex matrix pencil (A,B).
 *
 * Balances a pair of N-by-N complex matrices (A,B). This involves, first,
 * permuting A and B by equivalence transformations to isolate eigenvalues
 * in the first 1 to ILO-1 and last IHI+1 to N elements on the diagonal.
 * Second, applying a diagonal equivalence transformation to rows and columns
 * ILO to IHI to make the rows and columns as close in 1-norm as possible.
 *
 * @param[in] job Operation type:
 *                'N': none (ILO=1, IHI=N, scales=1);
 *                'P': permute only;
 *                'S': scale only;
 *                'B': both permute and scale.
 * @param[in] n Order of matrices A and B. n >= 0.
 * @param[in] thresh Threshold for scaling. See SLICOT documentation for details.
 *                   thresh >= 0: elements <= thresh*MXNORM ignored;
 *                   thresh < 0: automatic threshold search.
 * @param[in,out] a Complex N-by-N matrix A. Balanced on output.
 * @param[in] lda Leading dimension of A. lda >= max(1,n).
 * @param[in,out] b Complex N-by-N matrix B. Balanced on output.
 * @param[in] ldb Leading dimension of B. ldb >= max(1,n).
 * @param[out] ilo,ihi Indices such that A(i,j)=B(i,j)=0 for i>j, j<ILO or i>IHI.
 * @param[out] lscale Left permutations/scaling factors, dimension n.
 * @param[out] rscale Right permutations/scaling factors, dimension n.
 * @param[out] dwork Workspace. dwork[0:1]=initial norms, dwork[2:3]=final norms,
 *                   dwork[4]=threshold used. Dimension 6*n (thresh>=0) or 8*n (thresh<0).
 * @param[out] iwarn 0=no warning; 1=scaling reset to 1 (thresh=-2 or -4).
 * @param[out] info 0=success; <0=-i means i-th argument illegal.
 */
void mb4dlz(const char *job, i32 n, f64 thresh, c128 *a, i32 lda,
            c128 *b, i32 ldb, i32 *ilo, i32 *ihi,
            f64 *lscale, f64 *rscale, f64 *dwork, i32 *iwarn, i32 *info);

/**
 * @brief Balance a complex skew-Hamiltonian/Hamiltonian pencil.
 *
 * Balances the 2*N-by-2*N complex skew-Hamiltonian/Hamiltonian pencil aS - bH:
 *       S = [A  D ]     H = [C  V ]   where A, C are N-by-N,
 *           [E  A']         [W -C']
 * D, E are skew-Hermitian; V, W are Hermitian; ' = conjugate transpose.
 *
 * First permutes to isolate eigenvalues, then applies diagonal scaling
 * to make row/column pairs close in 1-norm.
 *
 * @param[in] job Operation type:
 *                'N': none (ILO=1, scales=1);
 *                'P': permute only;
 *                'S': scale only;
 *                'B': both permute and scale.
 * @param[in] n Order of matrices A, C, D, E, V, W. n >= 0.
 * @param[in] thresh Threshold for scaling. See SLICOT documentation.
 *                   thresh >= 0: elements <= thresh*MXNORM ignored;
 *                   thresh < 0: automatic threshold search (-1,-2,-3,-4 or <=-10).
 * @param[in,out] a Complex N-by-N matrix A. Balanced on output.
 * @param[in] lda Leading dimension of A. lda >= max(1,n).
 * @param[in,out] de Complex N-by-(N+1) matrix. Lower tri of E in cols 1:N,
 *                   upper tri of D in cols 2:N+1. Balanced on output.
 * @param[in] ldde Leading dimension of DE. ldde >= max(1,n).
 * @param[in,out] c Complex N-by-N matrix C. Balanced on output.
 * @param[in] ldc Leading dimension of C. ldc >= max(1,n).
 * @param[in,out] vw Complex N-by-(N+1) matrix. Lower tri of W in cols 1:N,
 *                   upper tri of V in cols 2:N+1. Balanced on output.
 * @param[in] ldvw Leading dimension of VW. ldvw >= max(1,n).
 * @param[out] ilo ILO-1 = number of deflated eigenvalues. ILO=1 if JOB='N'/'S'.
 * @param[out] lscale Left permutations/scaling factors, dimension n.
 * @param[out] rscale Right permutations/scaling factors, dimension n.
 * @param[out] dwork Workspace. dwork[0:1]=initial norms (S,H), dwork[2:3]=final,
 *                   dwork[4]=threshold used. Dimension 6*n (thresh>=0) or 8*n (thresh<0).
 * @param[out] iwarn 0=no warning; 1=scaling reset to 1 (thresh=-2 or -4).
 * @param[out] info 0=success; <0=-i means i-th argument illegal.
 */
void mb4dpz(const char *job, i32 n, f64 thresh, c128 *a, i32 lda,
            c128 *de, i32 ldde, c128 *c, i32 ldc, c128 *vw, i32 ldvw,
            i32 *ilo, f64 *lscale, f64 *rscale, f64 *dwork,
            i32 *iwarn, i32 *info);

#ifdef __cplusplus
}
#endif

#endif /* SLICOT_MB04_H */
