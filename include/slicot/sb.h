/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#ifndef SLICOT_SB_H
#define SLICOT_SB_H

#include "../slicot_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Multi-input pole assignment using state feedback.
 *
 * Computes the state feedback matrix F for a given system (A,B) such that
 * the closed-loop state matrix A + B*F has specified eigenvalues.
 * Uses the robust pole assignment algorithm based on Schur decomposition.
 *
 * @param[in] dico Specifies system type: 'C' = continuous, 'D' = discrete
 * @param[in] n State dimension (n >= 0)
 * @param[in] m Input dimension (m >= 0)
 * @param[in] np Number of eigenvalues to assign
 * @param[in] alpha Threshold for "good" eigenvalues (Re < alpha for 'C',
 *                  |lambda| < alpha for 'D')
 * @param[in,out] a On entry: N-by-N state matrix A. On exit: Z'*(A+B*F)*Z
 *                  in real Schur form.
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in] b N-by-M input matrix B
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in,out] wr,wi On entry: desired eigenvalues (real/imag parts).
 *                      On exit: assigned eigenvalues in leading NAP positions.
 * @param[out] nfp Number of fixed (unmodified) eigenvalues
 * @param[out] nap Number of assigned eigenvalues
 * @param[out] nup Number of uncontrollable eigenvalues
 * @param[out] f M-by-N state feedback matrix
 * @param[in] ldf Leading dimension of F (ldf >= max(1,m))
 * @param[out] z N-by-N orthogonal transformation matrix
 * @param[in] ldz Leading dimension of Z (ldz >= max(1,n))
 * @param[in] tol Tolerance for controllability tests (0 = default)
 * @param[out] dwork Workspace array
 * @param[in] ldwork Workspace size (>= max(1, 5*m, 5*n, 2*n+4*m))
 * @param[out] iwarn Warning count (stability violations)
 * @param[out] info Exit code: 0 = success, < 0 = invalid arg,
 *                  1 = Schur reduction failed, 2 = reordering failed,
 *                  3 = insufficient eigenvalues, 4 = incompatible poles
 */
void sb01bd(const char* dico, i32 n, i32 m, i32 np, f64 alpha,
            f64* a, i32 lda, const f64* b, i32 ldb,
            f64* wr, f64* wi, i32* nfp, i32* nap, i32* nup,
            f64* f, i32 ldf, f64* z, i32 ldz, f64 tol,
            f64* dwork, i32 ldwork, i32* iwarn, i32* info);

/**
 * @brief Solve continuous/discrete-time algebraic Riccati equation.
 *
 * Solves for X the continuous-time algebraic Riccati equation
 *     Q + A'*X + X*A - X*G*X = 0                          (DICO='C')
 *
 * or the discrete-time algebraic Riccati equation
 *     X = A'*X*A - A'*X*B*(R + B'*X*B)^-1 B'*X*A + Q      (DICO='D')
 *
 * where G = B*R^-1*B' must be provided on input instead of B and R.
 *
 * Uses Laub's Schur vector approach. Returns the solution X and the
 * closed-loop spectrum (stable eigenvalues of the optimal system).
 *
 * @param[in] dico 'C' for continuous-time, 'D' for discrete-time
 * @param[in] hinv For dico='D': 'D' for symplectic matrix H, 'I' for inverse of H
 *                 Not referenced if dico='C'
 * @param[in] uplo 'U' = upper triangle of G, Q stored
 *                 'L' = lower triangle of G, Q stored
 * @param[in] scal 'G' = general scaling, 'N' = no scaling
 * @param[in] sort 'S' = stable eigenvalues first, 'U' = unstable first
 * @param[in] n Order of matrices A, Q, G, X (n >= 0)
 * @param[in,out] a N-by-N matrix A, dimension (lda,n)
 *                  On exit for dico='D': contains A^(-1)
 *                  Unchanged for dico='C'
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in] g N-by-N symmetric matrix G = B*R^(-1)*B', dimension (ldg,n)
 *              Only upper or lower triangle referenced per uplo
 * @param[in] ldg Leading dimension of G (ldg >= max(1,n))
 * @param[in,out] q N-by-N symmetric matrix Q, dimension (ldq,n)
 *                  On entry: upper or lower triangle per uplo
 *                  On exit: solution matrix X
 * @param[in] ldq Leading dimension of Q (ldq >= max(1,n))
 * @param[out] rcond Reciprocal condition number of U(1,1) block
 * @param[out] wr Real parts of 2N eigenvalues, dimension (2*n)
 *                First N elements are closed-loop eigenvalues
 * @param[out] wi Imaginary parts of 2N eigenvalues, dimension (2*n)
 * @param[out] s 2N-by-2N ordered real Schur form, dimension (lds,2*n)
 * @param[in] lds Leading dimension of S (lds >= max(1,2*n))
 * @param[out] u 2N-by-2N orthogonal transformation matrix, dimension (ldu,2*n)
 * @param[in] ldu Leading dimension of U (ldu >= max(1,2*n))
 * @param[out] iwork Integer workspace, dimension (2*n)
 * @param[out] dwork Double workspace, dimension (ldwork)
 *                   dwork[0] = optimal ldwork, dwork[1] = scaling factor
 *                   For dico='D': dwork[2] = reciprocal condition of A
 * @param[in] ldwork Workspace size: >= max(2,6*n) for dico='C'
 *                                   >= max(3,6*n) for dico='D'
 * @param[out] bwork Logical workspace, dimension (2*n)
 * @param[out] info Exit code:
 *                  0 = success
 *                  < 0 = invalid parameter -info
 *                  1 = A is singular (discrete-time)
 *                  2 = H cannot be reduced to Schur form
 *                  3 = Schur form cannot be ordered
 *                  4 = H has less than N stable eigenvalues
 *                  5 = U(1,1) block is singular
 */
/**
 * @brief Select eigenvalue(s) closest to a given value.
 *
 * Chooses a real eigenvalue or a pair of complex conjugate eigenvalues
 * at minimal distance to a given real or complex value. Reorders the
 * eigenvalue arrays so selected eigenvalue(s) appear at the end.
 *
 * @param[in] reig true = select real eigenvalue, false = select complex pair.
 * @param[in] n Number of eigenvalues in WR and WI. N >= 1.
 * @param[in] xr Real part (or value if reig=true) of target.
 * @param[in] xi Imaginary part of target. Ignored if reig=true.
 * @param[in,out] wr Real parts of eigenvalues. On exit, reordered.
 * @param[in,out] wi Imaginary parts. Ignored if reig=true. On exit, reordered.
 * @param[out] s For real: selected eigenvalue. For complex: sum of pair.
 * @param[out] p For real: selected eigenvalue. For complex: product of pair.
 */
void sb01bx(bool reig, i32 n, f64 xr, f64 xi, f64* wr, f64* wi, f64* s, f64* p);

/**
 * @brief Pole placement for N=1 or N=2 systems.
 *
 * Constructs feedback matrix F such that A + B*F has prescribed eigenvalues.
 * Eigenvalues specified by sum S (and product P for N=2). F has minimum
 * Frobenius norm.
 *
 * @param[in] n Order of A; 1 or 2.
 * @param[in] m Number of columns of B (inputs). M >= 1.
 * @param[in] s Sum of eigenvalues (or eigenvalue if N=1).
 * @param[in] p Product of eigenvalues. Ignored if N=1.
 * @param[in,out] a N-by-N matrix. Destroyed on exit.
 * @param[in,out] b N-by-M matrix. Destroyed on exit.
 * @param[out] f M-by-N feedback matrix. If info=1 and N=2, contains rotation.
 * @param[in] tol Tolerance for controllability test.
 * @param[out] dwork Workspace, dimension M.
 * @param[out] info 0=success, 1=uncontrollable pair (A,B).
 */
void sb01by(i32 n, i32 m, f64 s, f64 p, f64* a, f64* b, f64* f, f64 tol,
            f64* dwork, i32* info);

/**
 * @brief Eigenstructure assignment for multi-input system in orthogonal canonical form.
 *
 * Computes feedback matrix G such that A - B*G has the desired eigenstructure,
 * specified by desired eigenvalues and free eigenvector elements.
 * The pair (A, B) must be in orthogonal canonical form as returned by AB01ND.
 *
 * @param[in] n Order of matrix A and number of rows of B (n >= 0)
 * @param[in] m Number of columns of matrix B (m >= 0)
 * @param[in] indcon Controllability index (0 <= indcon <= n)
 * @param[in,out] a On entry: N-by-N matrix A in canonical form.
 *                  On exit: real Schur form of A - B*G
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] b On entry: N-by-M matrix B in canonical form.
 *                  On exit: transformed matrix B
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in] nblk Block sizes of diagonal blocks (INDCON elements)
 * @param[in,out] wr,wi Real and imaginary parts of desired poles.
 *                      Complex conjugate pairs must be consecutive.
 * @param[in,out] z On entry: orthogonal matrix from AB01ND.
 *                  On exit: orthogonal matrix reducing A-B*G to Schur form
 * @param[in] ldz Leading dimension of Z (ldz >= max(1,n))
 * @param[in] y Free parameters for eigenvector design (M*N elements)
 * @param[out] count Actual number of Y elements used
 * @param[out] g M-by-N feedback matrix
 * @param[in] ldg Leading dimension of G (ldg >= max(1,m))
 * @param[in] tol Tolerance for rank determination (0 = default)
 * @param[out] iwork Integer workspace (M elements)
 * @param[out] dwork Double workspace (ldwork elements)
 * @param[in] ldwork Workspace size (>= max(M*N, M*M+2*N+4*M+1))
 * @param[out] info 0=success, <0=invalid arg -info, 1=not controllable
 */
void sb01dd(i32 n, i32 m, i32 indcon, f64* a, i32 lda, f64* b, i32 ldb,
            const i32* nblk, f64* wr, f64* wi, f64* z, i32 ldz, const f64* y,
            i32* count, f64* g, i32 ldg, f64 tol, i32* iwork, f64* dwork,
            i32 ldwork, i32* info);

/**
 * @brief Inner denominator of right-coprime factorization (order 1 or 2).
 *
 * Computes state-feedback F and matrix V such that (A+B*F, B*V, F, V) is inner.
 * A must be unstable (eigenvalues with positive real parts for continuous-time,
 * or moduli > 1 for discrete-time).
 *
 * @param[in] discr False for continuous-time, true for discrete-time
 * @param[in] n System order (1 or 2)
 * @param[in] m Number of inputs
 * @param[in] a State matrix A (n x n)
 * @param[in] lda Leading dimension of A
 * @param[in] b Input matrix B (n x m)
 * @param[in] ldb Leading dimension of B
 * @param[out] f State-feedback matrix F (m x n)
 * @param[in] ldf Leading dimension of F
 * @param[out] v Matrix V (m x m, upper triangular)
 * @param[in] ldv Leading dimension of V
 * @param[out] info 0=success, 1=uncontrollable, 2=stable/at limit, 3=real eigenvalues (N=2)
 */
void sb01fy(bool discr, i32 n, i32 m, const f64* a, i32 lda, const f64* b, i32 ldb,
            f64* f, i32 ldf, f64* v, i32 ldv, i32* info);

/**
 * @brief Solve algebraic Riccati equations using Schur vectors.
 *
 * Solves continuous-time: Q + A'X + XA - XGX = 0
 * or discrete-time: X = A'XA - A'XB(R + B'XB)^(-1)B'XA + Q
 * where G = B*R^(-1)*B' (use SB02MT to compute G from B,R).
 *
 * @param[in] dico 'C' continuous-time, 'D' discrete-time
 * @param[in] hinv For DICO='D': 'D' use H, 'I' use H^(-1)
 * @param[in] uplo 'U' upper triangle of G,Q stored, 'L' lower
 * @param[in] scal 'G' general scaling, 'N' no scaling
 * @param[in] sort 'S' stable eigenvalues first, 'U' unstable first
 * @param[in] n Order of matrices (N >= 0)
 * @param[in,out] a State matrix (N x N). On exit, contains Schur form.
 * @param[in] lda Leading dimension of A
 * @param[in] g Symmetric matrix G = B*R^(-1)*B' (N x N)
 * @param[in] ldg Leading dimension of G
 * @param[in,out] q Symmetric weighting matrix (N x N). On exit, solution X.
 * @param[in] ldq Leading dimension of Q
 * @param[out] rcond Reciprocal condition number estimate
 * @param[out] wr Real parts of closed-loop eigenvalues (N)
 * @param[out] wi Imaginary parts of closed-loop eigenvalues (N)
 * @param[out] s Hamiltonian/symplectic Schur form (2N x 2N)
 * @param[in] lds Leading dimension of S
 * @param[out] u Schur transformation matrix (2N x 2N)
 * @param[in] ldu Leading dimension of U
 * @param[out] iwork Integer workspace (2N)
 * @param[out] dwork Workspace, dwork[0] = optimal LDWORK on exit
 * @param[in] ldwork Workspace size
 * @param[out] bwork Logical workspace (2N)
 * @param[out] info 0=success, <0=arg error, 1=no solution, 2=condition
 */
void sb02md(
    const char* dico,
    const char* hinv,
    const char* uplo,
    const char* scal,
    const char* sort,
    const i32 n,
    f64* a,
    const i32 lda,
    const f64* g,
    const i32 ldg,
    f64* q,
    const i32 ldq,
    f64* rcond,
    f64* wr,
    f64* wi,
    f64* s,
    const i32 lds,
    f64* u,
    const i32 ldu,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* bwork,
    i32* info);

/**
 * @brief Riccati preprocessing - convert coupling weight problems to standard form.
 *
 * Computes:
 *   G = B*R^(-1)*B'
 *   A_bar = A - B*R^(-1)*L'
 *   Q_bar = Q - L*R^(-1)*L'
 *
 * where A, B, Q, R, L, and G are N-by-N, N-by-M, N-by-N, M-by-M,
 * N-by-M, and N-by-N matrices, respectively, with Q, R and G symmetric.
 *
 * @param[in] jobg 'G' to compute G, 'N' to not compute G
 * @param[in] jobl 'Z' if L is zero, 'N' if L is nonzero
 * @param[in] fact 'N' R unfactored, 'C' Cholesky factor, 'U' UdU'/LdL' factor
 * @param[in] uplo 'U' upper triangle stored, 'L' lower triangle stored
 * @param[in] n Order of A, Q, G; rows of B, L (n >= 0)
 * @param[in] m Order of R; columns of B, L (m >= 0)
 * @param[in,out] a N-by-N matrix A (if jobl='N'), dimension (lda,n)
 *                  On exit: A_bar = A - B*R^(-1)*L'
 * @param[in] lda Leading dimension of A (lda >= max(1,n) if jobl='N', else >= 1)
 * @param[in,out] b N-by-M matrix B, dimension (ldb,m)
 *                  On exit if oufact=1: B*chol(R)^(-1)
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in,out] q N-by-N symmetric matrix Q (if jobl='N'), dimension (ldq,n)
 *                  On exit: Q_bar = Q - L*R^(-1)*L'
 * @param[in] ldq Leading dimension of Q (ldq >= max(1,n) if jobl='N', else >= 1)
 * @param[in,out] r M-by-M symmetric matrix R, dimension (ldr,m)
 *                  On exit if oufact=1: Cholesky factor
 *                  On exit if oufact=2: UdU'/LdL' factors
 * @param[in] ldr Leading dimension of R (ldr >= max(1,m))
 * @param[in,out] l N-by-M matrix L (if jobl='N'), dimension (ldl,m)
 *                  On exit if oufact=1: L*chol(R)^(-1)
 * @param[in] ldl Leading dimension of L (ldl >= max(1,n) if jobl='N', else >= 1)
 * @param[in,out] ipiv Pivot indices for UdU'/LdL' (dimension m)
 * @param[out] oufact 0=no factorization (m=0), 1=Cholesky, 2=UdU'/LdL'
 * @param[out] g N-by-N matrix G = B*R^(-1)*B' (if jobg='G'), dimension (ldg,n)
 * @param[in] ldg Leading dimension of G (ldg >= max(1,n) if jobg='G', else >= 1)
 * @param[out] iwork Integer workspace, dimension (m). Not referenced if fact='C'/'U'.
 * @param[out] dwork Double workspace, dimension (ldwork)
 *                   On exit: dwork[0]=optimal ldwork, dwork[1]=rcond (if fact='N')
 * @param[in] ldwork Workspace size. Required sizes depend on fact, jobg, jobl.
 * @param[out] info 0=success, <0=invalid param, 1..m=singular d factor, m+1=R singular
 */
void sb02mt(
    const char* jobg,
    const char* jobl,
    const char* fact,
    const char* uplo,
    const i32 n,
    const i32 m,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* q,
    const i32 ldq,
    f64* r,
    const i32 ldr,
    f64* l,
    const i32 ldl,
    i32* ipiv,
    i32* oufact,
    f64* g,
    const i32 ldg,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info);

/**
 * @brief Extended Riccati preprocessing - convert coupling weight problems.
 *
 * Computes:
 *   G = B*R^(-1)*B'
 *   A_bar = A +/- op(B*R^(-1)*L')
 *   Q_bar = Q +/- L*R^(-1)*L'
 *
 * Extended version of SB02MT with additional TRANS, FLAG, and DEF parameters.
 *
 * @param[in] jobg 'G' to compute G, 'N' to not compute G
 * @param[in] jobl 'Z' if L is zero, 'N' if L is nonzero
 * @param[in] fact 'N' R unfactored, 'C' Cholesky factor, 'U' UdU'/LdL' factor
 * @param[in] uplo 'U' upper triangle stored, 'L' lower triangle stored
 * @param[in] trans 'N' for op(W)=W, 'T'/'C' for op(W)=W'
 * @param[in] flag 'P' for plus sign, 'M' for minus sign
 * @param[in] def 'D' R assumed positive definite, 'I' R assumed indefinite
 *                (only used when fact='N')
 * @param[in] n Order of A, Q, G; rows of B, L (n >= 0)
 * @param[in] m Order of R; columns of B, L (m >= 0)
 * @param[in,out] a N-by-N matrix A (if jobl='N'), dimension (lda,n)
 * @param[in] lda Leading dimension of A (lda >= max(1,n) if jobl='N', else >= 1)
 * @param[in,out] b N-by-M matrix B, dimension (ldb,m)
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in,out] q N-by-N symmetric matrix Q (if jobl='N'), dimension (ldq,n)
 * @param[in] ldq Leading dimension of Q (ldq >= max(1,n) if jobl='N', else >= 1)
 * @param[in,out] r M-by-M symmetric matrix R, dimension (ldr,m)
 * @param[in] ldr Leading dimension of R (ldr >= max(1,m))
 * @param[in,out] l N-by-M matrix L (if jobl='N'), dimension (ldl,m)
 * @param[in] ldl Leading dimension of L (ldl >= max(1,n) if jobl='N', else >= 1)
 * @param[in,out] ipiv Pivot indices for UdU'/LdL' (dimension m)
 * @param[out] oufact 0=no factorization (m=0), 1=Cholesky, 2=UdU'/LdL'
 * @param[out] g N-by-N matrix G = B*R^(-1)*B' (if jobg='G'), dimension (ldg,n)
 * @param[in] ldg Leading dimension of G (ldg >= max(1,n) if jobg='G', else >= 1)
 * @param[out] iwork Integer workspace, dimension (m). Not referenced if fact='C'/'U'.
 * @param[out] dwork Double workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size
 * @param[out] info 0=success, <0=invalid param, 1..m=singular d factor, m+1=R singular
 */
void sb02mx(
    const char* jobg,
    const char* jobl,
    const char* fact,
    const char* uplo,
    const char* trans,
    const char* flag_ch,
    const char* def,
    const i32 n,
    const i32 m,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* q,
    const i32 ldq,
    f64* r,
    const i32 ldr,
    f64* l,
    const i32 ldl,
    i32* ipiv,
    i32* oufact,
    f64* g,
    const i32 ldg,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info);

/**
 * @brief Hamiltonian or symplectic matrix construction for Riccati equations.
 *
 * Constructs the 2n-by-2n Hamiltonian or symplectic matrix S associated
 * to the linear-quadratic optimization problem, used to solve the
 * continuous- or discrete-time algebraic Riccati equation.
 *
 * For continuous-time (dico='C'), constructs Hamiltonian matrix:
 *     S = [  A   -G ]
 *         [ -Q   -A']
 *
 * For discrete-time (dico='D', hinv='D'), constructs symplectic matrix:
 *     S = [  A^(-1)        A^(-1)*G     ]
 *         [ Q*A^(-1)   A' + Q*A^(-1)*G  ]
 *
 * For discrete-time (dico='D', hinv='I'), constructs inverse symplectic:
 *     S = [ A + G*A^(-T)*Q   -G*A^(-T) ]
 *         [    -A^(-T)*Q       A^(-T)  ]
 *
 * @param[in] dico 'C' for continuous-time, 'D' for discrete-time
 * @param[in] hinv For dico='D': 'D' for matrix (2), 'I' for matrix (3)
 *                 Not referenced if dico='C'
 * @param[in] uplo 'U' = upper triangle of G, Q stored
 *                 'L' = lower triangle of G, Q stored
 * @param[in] n Order of matrices A, G, Q (n >= 0)
 * @param[in,out] a N-by-N matrix A, dimension (lda,n)
 *                  For dico='D': on exit contains A^(-1) if info=0
 *                  For dico='C': unchanged on exit
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in] g N-by-N symmetric matrix G, dimension (ldg,n)
 *              Only upper or lower triangle referenced per uplo
 * @param[in] ldg Leading dimension of G (ldg >= max(1,n))
 * @param[in] q N-by-N symmetric matrix Q, dimension (ldq,n)
 *              Only upper or lower triangle referenced per uplo
 * @param[in] ldq Leading dimension of Q (ldq >= max(1,n))
 * @param[out] s 2N-by-2N Hamiltonian or symplectic matrix, dimension (lds,2*n)
 * @param[in] lds Leading dimension of S (lds >= max(1,2*n))
 * @param[out] iwork Integer workspace, dimension (2*n)
 * @param[out] dwork Double workspace, dimension (ldwork)
 *                   On exit: dwork[0] = optimal ldwork
 *                   For dico='D': dwork[1] = reciprocal condition number of A
 * @param[in] ldwork Workspace size:
 *                   >= 1 if dico='C'
 *                   >= max(2,4*n) if dico='D'
 *                   If ldwork=-1: workspace query
 * @param[out] info Exit code:
 *                  0 = success
 *                  < 0 = invalid parameter -info
 *                  1..n = leading i-by-i submatrix of A is singular (discrete)
 *                  n+1 = A is numerically singular (discrete)
 */
void sb02mu(
    const char* dico,
    const char* hinv,
    const char* uplo,
    const i32 n,
    f64* a,
    const i32 lda,
    const f64* g,
    const i32 ldg,
    const f64* q,
    const i32 ldq,
    f64* s,
    const i32 lds,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info);

/**
 * @brief Optimal state feedback matrix for optimal control problem.
 *
 * Computes the optimal feedback matrix F for the problem of optimal control:
 *
 *   F = (R + B'XB)^(-1) (B'XA + L')   [discrete-time, DICO='D']
 *   F = R^(-1) (B'X + L')             [continuous-time, DICO='C']
 *
 * where A is N-by-N, B is N-by-M, L is N-by-M, R and X are M-by-M and N-by-N
 * symmetric matrices respectively.
 *
 * @param[in] dico 'D' for discrete-time, 'C' for continuous-time
 * @param[in] fact Specifies how R is given:
 *                 'N' = R is unfactored
 *                 'D' = R contains P-by-M matrix D where R = D'D
 *                 'C' = R contains Cholesky factor
 *                 'U' = R contains UdU'/LdL' factorization (continuous only)
 * @param[in] uplo 'U' = upper triangle stored, 'L' = lower triangle
 * @param[in] jobl 'Z' = L is zero, 'N' = L is nonzero
 * @param[in] n Order of matrices A and X (n >= 0)
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] p Number of rows of D (fact='D' only, p >= m for continuous)
 * @param[in] a N-by-N state matrix A (discrete only), dimension (lda,n)
 * @param[in] lda Leading dimension of A (lda >= max(1,n) if discrete, else >= 1)
 * @param[in,out] b N-by-M input matrix B, dimension (ldb,m)
 *                  May be modified on exit for discrete-time with factored R
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in,out] r M-by-M symmetric input weighting matrix R, dimension (ldr,m)
 *                  On exit: Cholesky or UdU'/LdL' factorization
 *                  For fact='D': P-by-M matrix D on entry
 * @param[in] ldr Leading dimension of R (ldr >= max(1,m), or max(1,m,p) for fact='D')
 * @param[in,out] ipiv Pivot indices for UdU'/LdL' factorization, dimension (m)
 *                     Input for fact='U', output for oufact[0]=2
 * @param[in] l N-by-M cross weighting matrix L (if jobl='N'), dimension (ldl,m)
 * @param[in] ldl Leading dimension of L (ldl >= max(1,n) if jobl='N', else >= 1)
 * @param[in,out] x N-by-N Riccati solution matrix X, dimension (ldx,n)
 *                  May be modified for discrete-time with factored R
 * @param[in] ldx Leading dimension of X (ldx >= max(1,n))
 * @param[in] rnorm 1-norm of original R (required for fact='U' only)
 * @param[out] f M-by-N optimal feedback matrix F, dimension (ldf,n)
 * @param[in] ldf Leading dimension of F (ldf >= max(1,m))
 * @param[out] oufact Array of dimension 2:
 *                    oufact[0]: 1=Cholesky of R/R+B'XB, 2=UdU'/LdL'
 *                    oufact[1]: 1=Cholesky of X, 2=spectral (discrete+factored only)
 * @param[out] dwork Workspace, dimension (ldwork)
 *                   On exit: dwork[0]=optimal ldwork, dwork[1]=rcond
 *                   If oufact[1]=2: dwork[2..n+1] contain eigenvalues of X
 * @param[in] ldwork Workspace size:
 *                   fact='U': >= max(2,2*m)
 *                   fact!='U', dico='C': >= max(2,3*m)
 *                   fact='N', dico='D': >= max(2,3*m,n)
 *                   fact!='N', dico='D': >= max(n+3*m+2,4*n+1)
 * @param[out] info 0=success, <0=invalid param, 1..m=singular d factor,
 *                  m+1=R singular, m+2=eigenvalue convergence, m+3=X indefinite
 */
void sb02nd(
    const char* dico,
    const char* fact,
    const char* uplo,
    const char* jobl,
    const i32 n,
    const i32 m,
    const i32 p,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* r,
    const i32 ldr,
    i32* ipiv,
    f64* l,
    const i32 ldl,
    f64* x,
    const i32 ldx,
    const f64 rnorm,
    f64* f,
    const i32 ldf,
    i32* oufact,
    f64* dwork,
    const i32 ldwork,
    i32* info);

/**
 * @brief Solve continuous- or discrete-time algebraic Riccati equations.
 *
 * Solves for X either the continuous-time algebraic Riccati equation:
 *     Q + A'X + XA - (L+XB)R^(-1)(L+XB)' = 0
 *
 * or the discrete-time algebraic Riccati equation:
 *     X = A'XA - (L+A'XB)(R + B'XB)^(-1)(L+A'XB)' + Q
 *
 * Uses the method of deflating subspaces based on reordering eigenvalues
 * in a generalized Schur matrix pair. Optionally, G = BR^(-1)B' may be
 * given instead of B and R. Suited for ill-conditioned R (uses internal scaling).
 *
 * @param[in] dico System type: 'C'=continuous, 'D'=discrete
 * @param[in] jobb Input type: 'B'=B,R given, 'G'=G given
 * @param[in] fact Factorization: 'N'=Q,R given, 'C'=C given (Q=C'C), 'D'=D given (R=D'D), 'B'=both
 * @param[in] uplo Triangle: 'U'=upper, 'L'=lower (for Q,R,G)
 * @param[in] jobl Cross-weight: 'Z'=L is zero, 'N'=L is nonzero
 * @param[in] sort Eigenvalue order: 'S'=stable first, 'U'=unstable first
 * @param[in] n State dimension (order of A, Q, X) (n >= 0)
 * @param[in] m Number of inputs (order of R, columns of B) (m >= 0 if JOBB='B')
 * @param[in] p Number of outputs (rows of C/D) (p >= 0 if FACT='C','D','B')
 * @param[in] a N-by-N state matrix A, dimension (lda,n)
 * @param[in] lda Leading dimension of A (>= max(1,n))
 * @param[in] b If JOBB='B': N-by-M input matrix B; If JOBB='G': N-by-N symmetric G
 * @param[in] ldb Leading dimension of B (>= max(1,n))
 * @param[in,out] q If FACT='N','D': N-by-N symmetric Q; If FACT='C','B': P-by-N matrix C
 * @param[in] ldq Leading dimension of Q
 * @param[in,out] r If FACT='N','C': M-by-M symmetric R; If FACT='D','B': P-by-M matrix D (not used if JOBB='G')
 * @param[in] ldr Leading dimension of R
 * @param[in,out] l N-by-M cross-weighting matrix L (JOBL='N', JOBB='B')
 * @param[in] ldl Leading dimension of L
 * @param[out] rcond Reciprocal condition number of solution system
 * @param[out] x N-by-N solution matrix X, dimension (ldx,n)
 * @param[in] ldx Leading dimension of X (>= max(1,n))
 * @param[out] alfar Real parts of eigenvalues, dimension (2*n)
 * @param[out] alfai Imaginary parts of eigenvalues, dimension (2*n)
 * @param[out] beta Denominators of eigenvalues, dimension (2*n)
 * @param[out] s 2N-by-2N ordered Schur form, dimension (lds, 2*n+m or 2*n)
 * @param[in] lds Leading dimension of S (>= max(1,2*n+m) if JOBB='B', >= max(1,2*n) otherwise)
 * @param[out] t 2N-by-2N upper triangular form (not used if DICO='C' and JOBB='G')
 * @param[in] ldt Leading dimension of T
 * @param[out] u 2N-by-2N transformation matrix, dimension (ldu,2*n)
 * @param[in] ldu Leading dimension of U (>= max(1,2*n))
 * @param[in] tol Tolerance for singularity test (if <= 0, machine epsilon used)
 * @param[out] iwork Integer workspace
 * @param[out] dwork Double workspace. dwork[0]=optimal work, dwork[1]=rcond (JOBB='B'), dwork[2]=scale
 * @param[in] ldwork Workspace size (>= max(7*(2*n+1)+16, 16*n, 2*n+m, 3*m) for JOBB='B')
 * @param[out] info 0=success, 1=singular pencil, 2=QZ/QR failed, 3=reordering failed,
 *                  4=eigenvalue instability, 5=dimension mismatch, 6=singular solution
 */
void sb02od(
    const char* dico,
    const char* jobb,
    const char* fact,
    const char* uplo,
    const char* jobl,
    const char* sort,
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
    i32* info);

/**
 * @brief Construct extended Hamiltonian/symplectic matrix pairs for Riccati equations.
 *
 * Constructs extended matrix pairs for algebraic Riccati equations in optimal
 * control and spectral factorization problems. The (2N+M)-dimensional pencils are:
 *
 * Discrete-time:              Continuous-time:
 * |A   0   B|     |E   0   0|    |A   0   B|     |E   0   0|
 * |Q  -E'  L| - z |0  -A'  0|,   |Q   A'  L| - s |0  -E'  0|
 * |L'  0   R|     |0  -B'  0|    |L'  B'  R|     |0   0   0|
 *
 * For JOBB='G', directly constructs 2N-by-2N pairs using G = B*R^(-1)*B'.
 * For JOBB='B', compresses the extended pencil to 2N-by-2N using QL factorization.
 *
 * @param[in] type_str Problem type: 'O'=optimal control, 'S'=spectral factorization
 * @param[in] dico System type: 'C'=continuous, 'D'=discrete
 * @param[in] jobb Input type: 'B'=B,R given, 'G'=G given
 * @param[in] fact Factorization: 'N'=Q,R given, 'C'=C given Q=C'C, 'D'=D given R=D'D, 'B'=both
 * @param[in] uplo Triangle: 'U'=upper, 'L'=lower (for Q,R,G when JOBB='G' or FACT='N')
 * @param[in] jobl Cross-weight: 'Z'=L is zero, 'N'=L is nonzero (JOBB='B' only)
 * @param[in] jobe Descriptor: 'I'=E is identity, 'N'=E is general
 * @param[in] n Order of A, Q, E (n >= 0)
 * @param[in] m Order of R and columns of B (JOBB='B'), unused for JOBB='G'
 * @param[in] p Rows of C and/or D (FACT='C','D','B' or TYPE='S')
 * @param[in] a N-by-N state matrix A, dimension (lda,n)
 * @param[in] lda Leading dimension of A (>= max(1,n))
 * @param[in] b If JOBB='B': N-by-M input matrix B; If JOBB='G': N-by-N symmetric G
 * @param[in] ldb Leading dimension of B (>= max(1,n))
 * @param[in] q If FACT='N','D': N-by-N symmetric Q; If FACT='C','B': P-by-N output matrix C
 * @param[in] ldq Leading dimension of Q (>= max(1,n) or max(1,p) depending on FACT)
 * @param[in] r If FACT='N','C': M-by-M symmetric R; If FACT='D','B': P-by-M matrix D
 * @param[in] ldr Leading dimension of R (varies by JOBB and FACT)
 * @param[in] l N-by-M cross-weighting matrix L (JOBL='N', JOBB='B')
 * @param[in] ldl Leading dimension of L (>= max(1,n) if JOBL='N')
 * @param[in] e N-by-N descriptor matrix E (JOBE='N')
 * @param[in] lde Leading dimension of E (>= max(1,n) if JOBE='N')
 * @param[out] af 2N-by-2N output matrix Af, dimension (ldaf, 2*n+m or 2*n)
 * @param[in] ldaf Leading dimension of AF (>= 2*n+m if JOBB='B', >= 2*n otherwise)
 * @param[out] bf 2N-by-2N output matrix Bf (not used if DICO='C', JOBB='G', JOBE='I')
 * @param[in] ldbf Leading dimension of BF
 * @param[in] tol Tolerance for singularity test (if <= 0, machine epsilon used)
 * @param[out] iwork Integer workspace (m if JOBB='B', 1 otherwise)
 * @param[out] dwork Double workspace. On exit: dwork[0]=optimal work, dwork[1]=rcond (JOBB='B')
 * @param[in] ldwork Workspace size (>= max(1, 2*n+m, 3*m) for JOBB='B', >= 1 otherwise)
 * @param[out] info 0=success, <0=invalid param, 1=singular extended pencil
 */
void sb02oy(
    const char* type_str,
    const char* dico,
    const char* jobb,
    const char* fact,
    const char* uplo,
    const char* jobl,
    const char* jobe,
    const i32 n,
    const i32 m,
    const i32 p,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    const f64* q,
    const i32 ldq,
    const f64* r,
    const i32 ldr,
    const f64* l,
    const i32 ldl,
    const f64* e,
    const i32 lde,
    f64* af,
    const i32 ldaf,
    f64* bf,
    const i32 ldbf,
    const f64 tol,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info);

/**
 * @brief Continuous-time algebraic Riccati equation solver using matrix sign function.
 *
 * Solves the real continuous-time matrix algebraic Riccati equation:
 *     op(A)'*X + X*op(A) + Q - X*G*X = 0
 * where op(A) = A or A' and G, Q are symmetric (G = G', Q = Q').
 * The matrices A, G, Q are N-by-N and solution X is symmetric.
 *
 * Optionally computes error bound and condition estimate.
 *
 * @param[in] job 'X' = solution only, 'A' = all (solution, rcond, ferr).
 * @param[in] trana 'N' = op(A) = A, 'T'/'C' = op(A) = A'.
 * @param[in] uplo 'U' = upper triangle of G,Q stored, 'L' = lower.
 * @param[in] n Order of matrices A, G, Q, X. n >= 0.
 * @param[in] a N-by-N coefficient matrix A.
 * @param[in] lda Leading dimension of A. lda >= max(1,n).
 * @param[in] g N-by-N symmetric matrix G (upper or lower per UPLO).
 * @param[in] ldg Leading dimension of G. ldg >= max(1,n).
 * @param[in] q N-by-N symmetric matrix Q (upper or lower per UPLO).
 * @param[in] ldq Leading dimension of Q. ldq >= max(1,n).
 * @param[out] x N-by-N symmetric solution matrix X.
 * @param[in] ldx Leading dimension of X. ldx >= max(1,n).
 * @param[out] rcond Reciprocal condition number (if JOB='A').
 * @param[out] ferr Forward error bound (if JOB='A').
 * @param[out] wr Real parts of closed-loop eigenvalues (if JOB='A'), dimension N.
 * @param[out] wi Imaginary parts of closed-loop eigenvalues (if JOB='A'), dimension N.
 * @param[out] iwork Integer workspace. 2*N if JOB='X', max(2*N,N*N) if JOB='A'.
 * @param[out] dwork Real workspace, dimension LDWORK.
 * @param[in] ldwork Workspace size. 4*N*N+8*N+1 if JOB='X',
 *                   max(4*N*N+8*N+1, 6*N*N) if JOB='A'. Use -1 for query.
 * @param[out] info 0 = success, <0 = invalid arg,
 *                  1 = Hamiltonian has imaginary eigenvalues,
 *                  2 = sign function did not converge (approx solution computed),
 *                  3 = linear system singular,
 *                  4 = Schur reduction of A-G*X failed.
 */
void sb02pd(const char* job, const char* trana, const char* uplo, i32 n,
            const f64* a, i32 lda, const f64* g, i32 ldg, const f64* q, i32 ldq,
            f64* x, i32 ldx, f64* rcond, f64* ferr, f64* wr, f64* wi,
            i32* iwork, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Estimate conditioning and error bound for continuous-time Riccati equation.
 *
 * Estimates the conditioning and computes an error bound on the solution
 * of the real continuous-time matrix algebraic Riccati equation:
 *     op(A)'*X + X*op(A) + Q - X*G*X = 0
 * where op(A) = A or A^T and Q, G are symmetric.
 *
 * @param[in] job 'C' = condition number only, 'E' = error bound only,
 *                'B' = both condition number and error bound.
 * @param[in] fact 'F' = Schur factorization of Ac provided in T,U
 *                 'N' = compute Schur factorization of Ac.
 * @param[in] trana 'N' = op(A) = A, 'T'/'C' = op(A) = A^T.
 * @param[in] uplo 'U' = upper triangle of Q,G used, 'L' = lower.
 * @param[in] lyapun 'O' = solve original Lyapunov equations,
 *                   'R' = solve reduced Lyapunov equations.
 * @param[in] n Order of matrices A, X, Q, G. n >= 0.
 * @param[in] a N-by-N matrix A. Not referenced if FACT='F' and LYAPUN='R'.
 * @param[in] lda Leading dimension of A.
 * @param[in,out] t N-by-N upper quasi-triangular Schur form of Ac.
 *                  Input if FACT='F', output if FACT='N'.
 * @param[in] ldt Leading dimension of T.
 * @param[in,out] u N-by-N orthogonal Schur transformation matrix.
 *                  Input if FACT='F' and LYAPUN='O', output if FACT='N' and LYAPUN='O'.
 *                  Not referenced if LYAPUN='R'.
 * @param[in] ldu Leading dimension of U.
 * @param[in] g N-by-N symmetric matrix G (upper or lower per UPLO).
 * @param[in] ldg Leading dimension of G.
 * @param[in] q N-by-N symmetric matrix Q (upper or lower per UPLO).
 * @param[in] ldq Leading dimension of Q.
 * @param[in] x N-by-N symmetric solution matrix X.
 * @param[in] ldx Leading dimension of X.
 * @param[out] sep Estimated sep(op(Ac), -op(Ac)') if JOB='C' or 'B'.
 * @param[out] rcond Reciprocal condition number estimate if JOB='C' or 'B'.
 * @param[out] ferr Forward error bound estimate if JOB='E' or 'B'.
 * @param[out] iwork Integer workspace, dimension N*N.
 * @param[out] dwork Real workspace, dimension LDWORK.
 * @param[in] ldwork Workspace size. Use -1 for query.
 * @param[out] info 0 = success, <0 = invalid arg i, >0 = algorithm error.
 */
void sb02qd(const char* job, const char* fact, const char* trana,
            const char* uplo, const char* lyapun, i32 n,
            const f64* a, i32 lda, f64* t, i32 ldt, f64* u, i32 ldu,
            const f64* g, i32 ldg, const f64* q, i32 ldq,
            const f64* x, i32 ldx, f64* sep, f64* rcond, f64* ferr,
            i32* iwork, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Solve algebraic Riccati equation using Schur vector method.
 *
 * Solves continuous-time algebraic Riccati equation:
 *     Q + op(A)'*X + X*op(A) - X*G*X = 0                   (DICO='C')
 *
 * or discrete-time algebraic Riccati equation:
 *     Q + op(A)'*X*(I_n + G*X)^(-1)*op(A) - X = 0         (DICO='D')
 *
 * where op(M) = M or M' (transpose), A, G, Q are N-by-N matrices,
 * G and Q are symmetric, and X is the symmetric solution.
 *
 * The matrix G = op(B)*R^(-1)*op(B)' must be provided instead of B and R.
 * Use SB02MT to compute G from B and R.
 *
 * @param[in] job Computation to perform:
 *                'X' = compute solution only
 *                'C' = compute reciprocal condition number only
 *                'E' = compute error bound only
 *                'A' = compute all (solution, condition, error bound)
 * @param[in] dico Problem type:
 *                 'C' = continuous-time
 *                 'D' = discrete-time
 * @param[in] hinv For discrete-time (DICO='D') with JOB='X'/'A':
 *                 'D' = construct symplectic matrix H
 *                 'I' = construct inverse of H
 * @param[in] trana Form of op(A):
 *                  'N' = op(A) = A
 *                  'T'/'C' = op(A) = A'
 * @param[in] uplo Triangle stored for G and Q:
 *                 'U' = upper triangle
 *                 'L' = lower triangle
 * @param[in] scal Scaling strategy (for JOB='X'/'A'):
 *                 'G' = general scaling
 *                 'N' = no scaling
 * @param[in] sort Eigenvalue ordering (for JOB='X'/'A'):
 *                 'S' = stable eigenvalues first
 *                 'U' = unstable eigenvalues first
 * @param[in] fact Schur factorization (for JOB!='X'):
 *                 'F' = T,V contain Schur factors
 *                 'N' = compute Schur factors
 * @param[in] lyapun Lyapunov equation form (for JOB!='X'):
 *                   'O' = original equations
 *                   'R' = reduced equations
 * @param[in] n Order of matrices A, Q, G, X (n >= 0)
 * @param[in] a N-by-N matrix A, dimension (lda,n)
 * @param[in] lda Leading dimension of A
 * @param[in,out] t N-by-N Schur form matrix (for JOB!='X'), dimension (ldt,n)
 * @param[in] ldt Leading dimension of T
 * @param[in,out] v N-by-N orthogonal Schur matrix (for JOB!='X'), dimension (ldv,n)
 * @param[in] ldv Leading dimension of V
 * @param[in,out] g N-by-N symmetric matrix G, dimension (ldg,n)
 * @param[in] ldg Leading dimension of G
 * @param[in,out] q N-by-N symmetric matrix Q, dimension (ldq,n)
 * @param[in] ldq Leading dimension of Q
 * @param[out] x N-by-N symmetric solution X, dimension (ldx,n)
 * @param[in] ldx Leading dimension of X
 * @param[out] sep Separation or scaling factor
 * @param[out] rcond Reciprocal condition number (for JOB='C'/'A')
 * @param[out] ferr Forward error bound (for JOB='E'/'A')
 * @param[out] wr Real parts of eigenvalues, dimension (2*n)
 * @param[out] wi Imaginary parts of eigenvalues, dimension (2*n)
 * @param[out] s 2N-by-2N ordered Schur form, dimension (lds,2*n)
 * @param[in] lds Leading dimension of S
 * @param[out] iwork Integer workspace
 * @param[out] dwork Double workspace
 * @param[in] ldwork Workspace size (>= 5+max(1,4*n*n+8*n) for JOB='X'/'A')
 * @param[out] bwork Logical workspace, dimension (2*n)
 * @param[out] info 0=success, 1=A singular, 2=Schur failed, 3=ordering failed,
 *                  4=not enough stable eigenvalues, 5=linear system singular,
 *                  6=Ac Schur failed, 7=near-equal eigenvalues (warning)
 */
void sb02rd(
    const char* job,
    const char* dico,
    const char* hinv,
    const char* trana,
    const char* uplo,
    const char* scal,
    const char* sort,
    const char* fact,
    const char* lyapun,
    const i32 n,
    f64* a,
    const i32 lda,
    f64* t,
    const i32 ldt,
    f64* v,
    const i32 ldv,
    f64* g,
    const i32 ldg,
    f64* q,
    const i32 ldq,
    f64* x,
    const i32 ldx,
    f64* sep,
    f64* rcond,
    f64* ferr,
    f64* wr,
    f64* wi,
    f64* s,
    const i32 lds,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* bwork,
    i32* info);

/**
 * @brief Construct Hamiltonian or symplectic matrix for Riccati equation.
 *
 * For continuous-time (DICO='C'):
 *         ( op(A)   -G    )
 *     S = (               )
 *         (  -Q   -op(A)' )
 *
 * For discrete-time (DICO='D'):
 *                 -1              -1
 *         (  op(A)           op(A)  *G       )
 *     S = (        -1                   -1   )  if HINV='D'
 *         ( Q*op(A)     op(A)' + Q*op(A)  *G )
 *
 * @param[in] dico Problem type: 'C'=continuous, 'D'=discrete
 * @param[in] hinv For DICO='D': 'D'=direct, 'I'=inverse
 * @param[in] trana Form of op(A): 'N'=A, 'T'/'C'=A'
 * @param[in] uplo Triangle stored: 'U'=upper, 'L'=lower
 * @param[in] n Order of A, G, Q (n >= 0)
 * @param[in] a N-by-N matrix A, dimension (lda,n)
 * @param[in] lda Leading dimension of A
 * @param[in,out] g N-by-N symmetric matrix G, dimension (ldg,n)
 * @param[in] ldg Leading dimension of G
 * @param[in,out] q N-by-N symmetric matrix Q, dimension (ldq,n)
 * @param[in] ldq Leading dimension of Q
 * @param[out] s 2N-by-2N Hamiltonian/symplectic matrix, dimension (lds,2*n)
 * @param[in] lds Leading dimension of S
 * @param[out] iwork Integer workspace (2*n for discrete)
 * @param[out] dwork Double workspace (6*n for discrete)
 * @param[in] ldwork Workspace size
 * @param[out] info 0=success, 1..n=singular A, n+1=numerically singular A
 */
void sb02ru(
    const char* dico,
    const char* hinv,
    const char* trana,
    const char* uplo,
    const i32 n,
    f64* a,
    const i32 lda,
    f64* g,
    const i32 ldg,
    f64* q,
    const i32 ldq,
    f64* s,
    const i32 lds,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info);

/**
 * @brief Condition and error bound for discrete-time Riccati equation.
 *
 * Estimates conditioning and computes error bound for:
 *     X = op(A)'*X*(I_n + G*X)^-1*op(A) + Q
 * where op(A) = A or A' and Q, G are symmetric.
 *
 * @param[in] job 'C' for rcond only, 'E' for ferr only, 'B' for both
 * @param[in] fact 'N' to compute Schur, 'F' if Schur supplied
 * @param[in] trana Specifies op(A): 'N' = A, 'T' or 'C' = A'
 * @param[in] uplo 'U' for upper, 'L' for lower triangular Q, G
 * @param[in] lyapun 'O' for original, 'R' for reduced Lyapunov
 * @param[in] n Order of matrices (n >= 0)
 * @param[in] a N-by-N matrix A, dimension (lda,n)
 * @param[in] lda Leading dimension of A
 * @param[in,out] t N-by-N Schur matrix T, dimension (ldt,n)
 * @param[in] ldt Leading dimension of T (ldt >= max(1,n))
 * @param[in,out] u N-by-N orthogonal matrix U, dimension (ldu,n)
 * @param[in] ldu Leading dimension of U
 * @param[in] g N-by-N symmetric matrix G, dimension (ldg,n)
 * @param[in] ldg Leading dimension of G (ldg >= max(1,n))
 * @param[in] q N-by-N symmetric matrix Q, dimension (ldq,n)
 * @param[in] ldq Leading dimension of Q (ldq >= max(1,n))
 * @param[in] x N-by-N symmetric solution X, dimension (ldx,n)
 * @param[in] ldx Leading dimension of X (ldx >= max(1,n))
 * @param[out] sepd Estimated separation (if job='C' or 'B')
 * @param[out] rcond Reciprocal condition number (if job='C' or 'B')
 * @param[out] ferr Forward error bound (if job='E' or 'B')
 * @param[out] iwork Integer workspace of dimension n*n
 * @param[out] dwork Double workspace of dimension ldwork
 * @param[in] ldwork Length of dwork (-1 for query)
 * @param[out] info 0 on success, -i if arg i invalid, 1..n for DGEES fail, n+1 for singular
 */
void sb02sd(const char* job, const char* fact, const char* trana, const char* uplo,
            const char* lyapun, i32 n, const f64* a, i32 lda, f64* t, i32 ldt,
            f64* u, i32 ldu, const f64* g, i32 ldg, const f64* q, i32 ldq,
            const f64* x, i32 ldx, f64* sepd, f64* rcond, f64* ferr,
            i32* iwork, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Solve continuous or discrete Lyapunov equation.
 *
 * Solves for X either the continuous-time Lyapunov equation:
 *     op(A)' * X + X * op(A) = scale * C
 * or the discrete-time Lyapunov equation:
 *     op(A)' * X * op(A) - X = scale * C
 *
 * where op(A) = A or A' and C is symmetric.
 * Can also estimate separation (condition number).
 *
 * @param[in] dico Specifies equation: 'C' = continuous, 'D' = discrete
 * @param[in] job Computation: 'X' = solution, 'S' = separation, 'B' = both
 * @param[in] fact Factorization: 'N' = compute Schur, 'F' = provided
 * @param[in] trana Form of op(A): 'N' = A, 'T'/'C' = A'
 * @param[in] n Order of matrices (n >= 0)
 * @param[in,out] a On entry: matrix A. On exit: Schur form (if FACT='N')
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] u On entry (FACT='F'): orthogonal U. On exit: Schur U.
 * @param[in] ldu Leading dimension of U (ldu >= max(1,n))
 * @param[in,out] c On entry: symmetric RHS C. On exit: solution X.
 * @param[in] ldc Leading dimension of C
 * @param[out] scale Scale factor (0 < scale <= 1)
 * @param[out] sep Separation estimate (if JOB='S' or 'B')
 * @param[out] ferr Forward error bound (if JOB='B')
 * @param[out] wr,wi Eigenvalues of A (if FACT='N')
 * @param[out] iwork Integer workspace (N*N, not needed if JOB='X')
 * @param[out] dwork Workspace array
 * @param[in] ldwork Workspace size
 * @param[out] info Exit code: 0 = success, < 0 = invalid arg,
 *                  > 0 = Schur failed, N+1 = nearly singular
 */
void sb03md(const char* dico, const char* job, const char* fact,
            const char* trana, i32 n, f64* a, i32 lda,
            f64* u, i32 ldu, f64* c, i32 ldc, f64* scale,
            f64* sep, f64* ferr, f64* wr, f64* wi, i32* iwork,
            f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Solve 2x2 discrete-time Lyapunov equation.
 *
 * Solves for the 2-by-2 symmetric matrix X in:
 *     op(T)'*X*op(T) - X = SCALE*B
 *
 * where T is 2-by-2, B is symmetric 2-by-2, and op(T) = T or T'.
 *
 * The equivalent linear algebraic system of equations is formed and solved
 * using Gaussian elimination with complete pivoting.
 *
 * @param[in] ltran Specifies op(T): false = T, true = T'
 * @param[in] lupper Specifies which triangle:
 *                   true = use upper triangular part of B, compute upper X
 *                   false = use lower triangular part of B, compute lower X
 * @param[in] t 2-by-2 matrix T, dimension (ldt,2), column-major
 * @param[in] ldt Leading dimension of T (ldt >= 2)
 * @param[in] b 2-by-2 symmetric matrix B, dimension (ldb,2), column-major
 *              Only the triangle specified by lupper is referenced
 * @param[in] ldb Leading dimension of B (ldb >= 2)
 * @param[out] scale Scale factor (0 < scale <= 1), set to prevent overflow
 * @param[out] x 2-by-2 symmetric solution matrix X, dimension (ldx,2)
 *               Only the triangle specified by lupper is computed
 *               Note: X may be identified with B in the calling statement
 * @param[in] ldx Leading dimension of X (ldx >= 2)
 * @param[out] xnorm Infinity norm of the solution X
 * @param[out] info Exit code:
 *                  0 = success
 *                  1 = T has nearly reciprocal eigenvalues (perturbed values used)
 */
void sb03mv(bool ltran, bool lupper, const f64* t, i32 ldt,
            const f64* b, i32 ldb, f64* scale, f64* x, i32 ldx,
            f64* xnorm, i32* info);

/**
 * @brief Solve 2x2 continuous-time Lyapunov equation.
 *
 * Solves for the 2-by-2 symmetric matrix X in:
 *     op(T)' * X + X * op(T) = scale * B
 * where T is 2-by-2, B is symmetric 2-by-2, and op(T) = T or T'.
 *
 * @param[in] ltran Specifies op(T): false = T, true = T'
 * @param[in] lupper Use upper triangle of B, compute upper X
 * @param[in] t 2-by-2 matrix T, dimension (ldt,2)
 * @param[in] ldt Leading dimension of T (ldt >= 2)
 * @param[in] b 2-by-2 symmetric matrix B, dimension (ldb,2)
 * @param[in] ldb Leading dimension of B (ldb >= 2)
 * @param[out] scale Scale factor (0 < scale <= 1)
 * @param[out] x 2-by-2 solution matrix X, dimension (ldx,2)
 * @param[in] ldx Leading dimension of X (ldx >= 2)
 * @param[out] xnorm Infinity norm of solution X
 * @param[out] info Exit code: 0 = success, 1 = nearly reciprocal eigenvalues
 */
void sb03mw(bool ltran, bool lupper, const f64* t, i32 ldt,
            const f64* b, i32 ldb, f64* scale, f64* x, i32 ldx,
            f64* xnorm, i32* info);

/**
 * @brief Solve discrete-time Lyapunov for quasi-triangular A.
 *
 * Solves the discrete-time Lyapunov equation:
 *     op(A)' * X * op(A) - X = scale * C
 * where A is in upper quasi-triangular (Schur) form and C is symmetric.
 *
 * @param[in] trana Specifies op(A): 'N' = A, 'T' or 'C' = A'
 * @param[in] n Order of matrices (n >= 0)
 * @param[in] a N-by-N upper quasi-triangular matrix A, dimension (lda,n)
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] c On entry: symmetric RHS matrix C. On exit: solution X.
 * @param[in] ldc Leading dimension of C (ldc >= max(1,n))
 * @param[out] scale Scale factor (0 < scale <= 1)
 * @param[out] dwork Workspace, dimension at least (2*n)
 * @param[out] info Exit code: 0 = success, < 0 = invalid arg,
 *                  1 = nearly singular (perturbed values used)
 */
void sb03mx(const char* trana, i32 n, const f64* a, i32 lda,
            f64* c, i32 ldc, f64* scale, f64* dwork, i32* info);

/**
 * @brief Solve discrete-time Lyapunov equation with separation estimation.
 *
 * Solves the real discrete-time Lyapunov equation:
 *     op(A)' * X * op(A) - X = scale * C
 * and/or estimates the separation:
 *     sepd(op(A), op(A)') = min norm(op(A)'*X*op(A) - X) / norm(X)
 *
 * where op(A) = A or A' and C is symmetric.
 *
 * @param[in] job Computation to perform: 'X' = solution only,
 *                'S' = separation only, 'B' = both
 * @param[in] fact 'F' = Schur factorization supplied, 'N' = compute Schur
 * @param[in] trana Specifies op(A): 'N' = A, 'T' or 'C' = A'
 * @param[in] n Order of matrices (n >= 0)
 * @param[in,out] a On entry: N-by-N matrix A (or Schur form if FACT='F').
 *                  On exit: upper quasi-triangular Schur form
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] u If FACT='F': input orthogonal matrix from Schur factorization.
 *                  If FACT='N': output orthogonal matrix from Schur factorization
 * @param[in] ldu Leading dimension of U (ldu >= max(1,n))
 * @param[in,out] c On entry: symmetric RHS matrix C (if JOB != 'S').
 *                  On exit: symmetric solution X (if JOB != 'S')
 * @param[in] ldc Leading dimension of C (ldc >= 1 if JOB='S', else >= max(1,n))
 * @param[out] scale Scale factor (0 < scale <= 1)
 * @param[out] sepd Separation estimate (if JOB = 'S' or 'B')
 * @param[out] ferr Forward error bound for solution (if JOB = 'B')
 * @param[out] wr Real parts of eigenvalues (if FACT = 'N')
 * @param[out] wi Imaginary parts of eigenvalues (if FACT = 'N')
 * @param[out] iwork Integer workspace, dimension (n*n), not used if JOB='X'
 * @param[out] dwork Double workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size. If JOB='X': max(n*n, 2*n or 3*n).
 *                   If JOB='S' or 'B': 2*n*n + 2*n
 * @param[out] info Exit code: 0 = success, < 0 = invalid arg,
 *                  > 0 = Schur decomp failed, n+1 = near-singular equation
 */
void sb03pd(const char* job, const char* fact, const char* trana,
            i32 n, f64* a, i32 lda, f64* u, i32 ldu, f64* c, i32 ldc,
            f64* scale, f64* sepd, f64* ferr, f64* wr, f64* wi,
            i32* iwork, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Solve small discrete-time Sylvester equation.
 *
 * Solves for the N1-by-N2 matrix X (1 <= N1,N2 <= 2) in:
 *     ISGN * op(TL) * X * op(TR) - X = SCALE * B
 * where TL is N1-by-N1, TR is N2-by-N2, B is N1-by-N2, and ISGN = 1 or -1.
 * op(T) = T or T' (transpose).
 *
 * The equivalent linear algebraic system is formed and solved using
 * Gaussian elimination with complete pivoting.
 *
 * @param[in] ltranl Specifies op(TL): false = TL, true = TL'
 * @param[in] ltranr Specifies op(TR): false = TR, true = TR'
 * @param[in] isgn Sign of equation: 1 or -1
 * @param[in] n1 Order of TL matrix (0, 1, or 2)
 * @param[in] n2 Order of TR matrix (0, 1, or 2)
 * @param[in] tl N1-by-N1 matrix TL, dimension (ldtl,2)
 * @param[in] ldtl Leading dimension of TL (ldtl >= max(1,n1))
 * @param[in] tr N2-by-N2 matrix TR, dimension (ldtr,2)
 * @param[in] ldtr Leading dimension of TR (ldtr >= max(1,n2))
 * @param[in] b N1-by-N2 RHS matrix B, dimension (ldb,2)
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n1))
 * @param[out] scale Scale factor (0 < scale <= 1)
 * @param[out] x N1-by-N2 solution matrix X, dimension (ldx,n2)
 * @param[in] ldx Leading dimension of X (ldx >= max(1,n1))
 * @param[out] xnorm Infinity norm of solution X
 * @param[out] info Exit code: 0 = success, 1 = nearly reciprocal eigenvalues
 */
void sb03mu(bool ltranl, bool ltranr, i32 isgn, i32 n1, i32 n2,
            const f64* tl, i32 ldtl, const f64* tr, i32 ldtr,
            const f64* b, i32 ldb, f64* scale, f64* x, i32 ldx,
            f64* xnorm, i32* info);

/**
 * @brief Solve continuous-time Lyapunov for quasi-triangular A.
 *
 * Solves the continuous-time Lyapunov equation:
 *     op(A)' * X + X * op(A) = scale * C
 * where A is in upper quasi-triangular (Schur) form and C is symmetric.
 *
 * @param[in] trana Specifies op(A): 'N' = A, 'T' or 'C' = A'
 * @param[in] n Order of matrices (n >= 0)
 * @param[in] a N-by-N upper quasi-triangular matrix A, dimension (lda,n)
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] c On entry: symmetric RHS matrix C. On exit: solution X.
 * @param[in] ldc Leading dimension of C (ldc >= max(1,n))
 * @param[out] scale Scale factor (0 < scale <= 1)
 * @param[out] info Exit code: 0 = success, < 0 = invalid arg,
 *                  1 = nearly singular (perturbed values used)
 */
void sb03my(const char* trana, i32 n, const f64* a, i32 lda,
            f64* c, i32 ldc, f64* scale, i32* info);

/**
 * @brief Estimate conditioning and forward error bound for continuous-time Lyapunov equation.
 *
 * Estimates the conditioning and computes an error bound on the solution of:
 *     op(A)' * X + X * op(A) = scale * C
 * where op(A) = A or A' and C is symmetric.
 *
 * @param[in] job 'C': condition number only, 'E': error bound only, 'B': both
 * @param[in] fact 'F': Schur factorization provided, 'N': compute factorization
 * @param[in] trana 'N': op(A)=A, 'T'/'C': op(A)=A'
 * @param[in] uplo 'U': upper triangle of C, 'L': lower triangle
 * @param[in] lyapun 'O': original equations, 'R': reduced equations
 * @param[in] n Order of matrices (n >= 0)
 * @param[in] scale Scale factor from Lyapunov solver (0 <= scale <= 1)
 * @param[in] a N-by-N original matrix A (if fact='N' or lyapun='O')
 * @param[in] lda Leading dimension of a
 * @param[in,out] t N-by-N upper quasi-triangular Schur form of A
 *                  If fact='F': input; if fact='N': output
 * @param[in] ldt Leading dimension of t
 * @param[in,out] u N-by-N orthogonal transformation matrix
 *                  If lyapun='O' and fact='F': input; if fact='N': output
 * @param[in] ldu Leading dimension of u
 * @param[in] c N-by-N symmetric RHS matrix C (specified triangle only)
 * @param[in] ldc Leading dimension of c
 * @param[in] x N-by-N symmetric solution matrix X
 * @param[in] ldx Leading dimension of x
 * @param[out] sep Estimated sep(op(A),-op(A)') (if job='C' or 'B')
 * @param[out] rcond Reciprocal condition number (if job='C' or 'B')
 * @param[out] ferr Forward error bound (if job='E' or 'B')
 * @param[out] iwork Integer workspace of size N*N
 * @param[out] dwork Double workspace of size ldwork
 * @param[in] ldwork Workspace size (see routine for requirements, -1 for query)
 * @param[out] info 0=success, <0=param error, 1..N=Schur failed, N+1=close eigenvalues
 */
void sb03qd(const char* job, const char* fact, const char* trana,
            const char* uplo, const char* lyapun, i32 n, f64 scale,
            const f64* a, i32 lda, f64* t, i32 ldt, f64* u, i32 ldu,
            const f64* c, i32 ldc, const f64* x, i32 ldx,
            f64* sep, f64* rcond, f64* ferr,
            i32* iwork, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Estimate forward error bound for continuous-time Lyapunov equation.
 *
 * Estimates a forward error bound for the solution X of:
 *     op(A)' * X + X * op(A) = C
 * where op(A) = A or A' and C is symmetric.
 *
 * The matrix A, RHS C, and solution X are N-by-N.
 * An absolute residual matrix R (with rounding error bounds) is provided.
 *
 * @param[in] trana 'N': op(A)=A, 'T'/'C': op(A)=A'
 * @param[in] uplo 'U': upper triangle of R used, 'L': lower triangle
 * @param[in] lyapun 'O': original equations (use U), 'R': reduced only
 * @param[in] n Order of matrices A, R (N >= 0)
 * @param[in] xanorm Max-norm of solution X (XANORM >= 0)
 * @param[in] t N-by-N upper quasi-triangular Schur form of A
 * @param[in] ldt Leading dimension of T (LDT >= max(1,N))
 * @param[in] u N-by-N orthogonal matrix from Schur factorization (if LYAPUN='O')
 * @param[in] ldu Leading dimension of U (LDU >= 1, or max(1,N) if LYAPUN='O')
 * @param[in,out] r On entry: residual matrix (specified triangle).
 *                  On exit: symmetric residual matrix (fully stored).
 * @param[in] ldr Leading dimension of R (LDR >= max(1,N))
 * @param[out] ferr Estimated forward error bound
 * @param[out] iwork Integer workspace of size N*N
 * @param[out] dwork Double workspace of size LDWORK
 * @param[in] ldwork Size of DWORK (LDWORK >= 2*N*N)
 * @param[out] info 0=success, <0=param error, N+1=nearly common eigenvalues
 */
void sb03qx(const char* trana, const char* uplo, const char* lyapun,
            i32 n, f64 xanorm, const f64* t, i32 ldt, const f64* u, i32 ldu,
            f64* r, i32 ldr, f64* ferr, i32* iwork, f64* dwork, i32 ldwork,
            i32* info);

/**
 * @brief Estimate separation and 1-norm of Theta for continuous-time Lyapunov equation.
 *
 * Estimates sep(op(A), -op(A)') = min norm(op(A)'*X + X*op(A))/norm(X)
 * and/or the 1-norm of operator Theta associated with the Lyapunov equation:
 *     op(A)'*X + X*op(A) = C
 *
 * @param[in] job Computation to perform:
 *                'S' = compute separation only
 *                'T' = compute norm of Theta only
 *                'B' = compute both separation and norm of Theta
 * @param[in] trana Form of op(A):
 *                  'N' = op(A) = A
 *                  'T'/'C' = op(A) = A'
 * @param[in] lyapun Lyapunov equation form:
 *                   'O' = original equations, updating with U
 *                   'R' = reduced equations only
 * @param[in] n Order of matrices A and X (n >= 0)
 * @param[in] t N-by-N upper quasi-triangular Schur form of A
 * @param[in] ldt Leading dimension of T (ldt >= max(1,n))
 * @param[in] u N-by-N orthogonal matrix from Schur factorization
 *              (not referenced if lyapun='R')
 * @param[in] ldu Leading dimension of U (ldu >= 1, or ldu >= n if lyapun='O')
 * @param[in] x N-by-N solution of Lyapunov equation
 *              (not referenced if job='S')
 * @param[in] ldx Leading dimension of X (ldx >= 1, or ldx >= n if job='T'/'B')
 * @param[out] sep Estimated separation (if job='S' or 'B')
 * @param[out] thnorm Estimated 1-norm of Theta (if job='T' or 'B')
 * @param[out] iwork Integer workspace, dimension (N*N)
 * @param[out] dwork Double workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size (ldwork >= 2*N*N)
 * @param[out] info Exit status:
 *                  0 = success
 *                  < 0 = illegal argument -info
 *                  N+1 = T and -T' have common/close eigenvalues
 */
void sb03qy(const char* job, const char* trana, const char* lyapun,
            i32 n, const f64* t, i32 ldt, const f64* u, i32 ldu,
            const f64* x, i32 ldx, f64* sep, f64* thnorm,
            i32* iwork, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Continuous-time Lyapunov equation solver with condition/error estimation.
 *
 * Solves the real continuous-time Lyapunov matrix equation
 *     op(A)' * X + X * op(A) = scale * C
 * estimates the conditioning, and computes an error bound on the solution X,
 * where op(A) = A or A' (transpose), the matrix A is N-by-N,
 * the right hand side C and the solution X are N-by-N symmetric matrices.
 *
 * @param[in] job Specifies computation: 'X'=solution, 'S'=separation,
 *                'C'=condition, 'E'=error bound, 'A'=all.
 * @param[in] fact 'F'=Schur factors given, 'N'=compute Schur factorization.
 * @param[in] trana 'N'=op(A)=A, 'T'/'C'=op(A)=A'.
 * @param[in] uplo 'U'=upper triangle of C, 'L'=lower triangle of C.
 * @param[in] lyapun 'O'=original equation, 'R'=reduced (Schur form).
 * @param[in] n Order of matrices A, X, C. n >= 0.
 * @param[in,out] scale If JOB='C'/'E': input scale factor. If JOB='X'/'A': output.
 * @param[in] a N-by-N matrix A (if FACT='N' or (LYAPUN='O' and JOB<>'X')).
 * @param[in] lda Leading dimension of a.
 * @param[in,out] t On entry (FACT='F'): Schur form. On exit: Schur form of A.
 * @param[in] ldt Leading dimension of t.
 * @param[in,out] u On entry (LYAPUN='O',FACT='F'): orthogonal U. On exit: Schur U.
 * @param[in] ldu Leading dimension of u.
 * @param[in,out] c N-by-N symmetric matrix C (upper or lower part).
 * @param[in] ldc Leading dimension of c.
 * @param[in,out] x On entry (JOB='C'/'E'): solution. On exit (JOB='X'/'A'): solution.
 * @param[in] ldx Leading dimension of x.
 * @param[out] sep Estimated separation sep(op(A), -op(A)').
 * @param[out] rcond Reciprocal condition number.
 * @param[out] ferr Estimated forward error bound.
 * @param[out] wr Real parts of eigenvalues of A (if FACT='N').
 * @param[out] wi Imaginary parts of eigenvalues of A (if FACT='N').
 * @param[out] iwork Integer workspace of dimension N*N.
 * @param[out] dwork Workspace array. dwork[0] = optimal workspace on exit.
 * @param[in] ldwork Length of dwork.
 * @param[out] info 0=success, <0=parameter error, 1..N=Schur failure, N+1=close eigenvalues.
 */
void sb03td(const char* job, const char* fact, const char* trana,
            const char* uplo, const char* lyapun, i32 n, f64* scale,
            const f64* a, i32 lda, f64* t, i32 ldt, f64* u, i32 ldu,
            f64* c, i32 ldc, f64* x, i32 ldx,
            f64* sep, f64* rcond, f64* ferr,
            f64* wr, f64* wi, i32* iwork, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Solve Lyapunov equation for Cholesky factor of solution.
 *
 * Solves for X = op(U)'*op(U) either the stable continuous-time Lyapunov equation:
 *
 *     op(A)'*X + X*op(A) = -scale^2*op(B)'*op(B)   (DICO='C')
 *
 * or the convergent discrete-time Lyapunov equation:
 *
 *     op(A)'*X*op(A) - X = -scale^2*op(B)'*op(B)   (DICO='D')
 *
 * where op(K) = K or K' (transpose), A is N-by-N, op(B) is M-by-N, and U is
 * the upper triangular Cholesky factor of the solution X. Scale is set <= 1
 * to avoid overflow in X.
 *
 * For continuous-time (DICO='C'): A must be stable (all eigenvalues have
 * negative real parts).
 * For discrete-time (DICO='D'): A must be convergent (all eigenvalues
 * inside the unit circle).
 *
 * Based on Bartels-Stewart method [1] finding Cholesky factor directly without
 * forming the normal matrix op(B)'*op(B) [2].
 *
 * References:
 * [1] Bartels, Stewart. Solution of A'X + XB = C. CACM 15, 820-826, 1972.
 * [2] Hammarling. Numerical solution of stable Lyapunov equation. IMA J. Num. Anal. 2, 303-325, 1982.
 *
 * @param[in] dico Equation type:
 *                 'C' = continuous-time
 *                 'D' = discrete-time
 * @param[in] fact Schur factorization option:
 *                 'F' = A and Q contain Schur factorization (provided by user)
 *                 'N' = Schur factorization will be computed
 * @param[in] trans Form of op(K):
 *                  'N' = op(K) = K (no transpose)
 *                  'T' = op(K) = K' (transpose)
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in] m Number of rows of op(B) (m >= 0)
 * @param[in,out] a N-by-N matrix A, dimension (lda,n)
 *                  If FACT='F': upper quasi-triangular Schur form
 *                  If FACT='N': general matrix, overwritten with Schur form
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] q N-by-N orthogonal matrix Q, dimension (ldq,n)
 *                  If FACT='F': orthogonal matrix from Schur factorization
 *                  If FACT='N': output orthogonal matrix Q
 * @param[in] ldq Leading dimension of Q (ldq >= max(1,n))
 * @param[in,out] b Coefficient matrix B:
 *                  If TRANS='N': M-by-N on entry, dimension (ldb,n)
 *                  If TRANS='T': N-by-M on entry, dimension (ldb,max(m,n))
 *                  On exit: N-by-N upper triangular Cholesky factor U
 * @param[in] ldb Leading dimension of B
 *                If TRANS='N': ldb >= max(1,n,m)
 *                If TRANS='T': ldb >= max(1,n)
 * @param[out] scale Scale factor (0 < scale <= 1) to prevent overflow
 * @param[out] wr Real parts of eigenvalues of A, dimension (n)
 * @param[out] wi Imaginary parts of eigenvalues of A, dimension (n)
 * @param[out] dwork Workspace array, dimension (ldwork)
 *                   On exit: dwork[0] = optimal ldwork
 * @param[in] ldwork Workspace size:
 *                   If m > 0: ldwork >= max(1,4*n)
 *                   If m = 0: ldwork >= 1
 *                   If ldwork = -1: workspace query
 * @param[out] info Exit code:
 *                  0 = success
 *                  1 = nearly singular (warning): perturbed values used
 *                  2 = A not stable/convergent (FACT='N')
 *                  3 = Schur form S not stable/convergent (FACT='F')
 *                  4 = S has >2x2 diagonal block (FACT='F')
 *                  5 = S has 2x2 block with real eigenvalues (FACT='F')
 *                  6 = DGEES failed to converge (FACT='N')
 */
void sb03od(
    const char* dico,
    const char* fact,
    const char* trans,
    const i32 n,
    const i32 m,
    f64* a,
    const i32 lda,
    f64* q,
    const i32 ldq,
    f64* b,
    const i32 ldb,
    f64* scale,
    f64* wr,
    f64* wi,
    f64* dwork,
    const i32 ldwork,
    i32* info);

/**
 * @brief Solve real quasi-triangular Sylvester equation.
 *
 * Solves for the N-by-M matrix X (M = 1 or 2) in:
 *
 *     op(S)'*X + X*op(A) = scale*C   (DISCR = false, continuous)
 *     op(S)'*X*op(A) - X = scale*C   (DISCR = true, discrete)
 *
 * where op(K) = K or K' (transpose), S is an N-by-N block upper triangular
 * matrix with 1x1 and 2x2 blocks on the diagonal (real Schur form), A is an
 * M-by-M matrix. The solution X overwrites C. Scale is set <= 1 to avoid
 * overflow in X.
 *
 * This is a service routine for the Lyapunov solver SB03OT.
 *
 * @param[in] discr Equation type:
 *                  false = continuous: op(S)'*X + X*op(A) = scale*C
 *                  true = discrete: op(S)'*X*op(A) - X = scale*C
 * @param[in] ltrans Form of op(K):
 *                   false = op(K) = K (no transpose)
 *                   true = op(K) = K' (transpose)
 * @param[in] n Order of matrix S, number of rows of X and C (n >= 0)
 * @param[in] m Order of matrix A, number of columns of X and C (m = 1 or 2)
 * @param[in] s N-by-N block upper triangular matrix (real Schur form),
 *              dimension (lds,n). Elements below upper Hessenberg not referenced.
 * @param[in] lds Leading dimension of S (lds >= max(1,n))
 * @param[in] a M-by-M matrix A, dimension (lda,m)
 * @param[in] lda Leading dimension of A (lda >= m)
 * @param[in,out] c N-by-M matrix, dimension (ldc,m)
 *                  On entry: right-hand side matrix C
 *                  On exit: solution matrix X
 * @param[in] ldc Leading dimension of C (ldc >= max(1,n))
 * @param[out] scale Scale factor (0 < scale <= 1) to prevent overflow
 * @param[out] info Exit code:
 *                  0 = success
 *                  1 = S and -A have common eigenvalues (continuous), or
 *                      S and A have eigenvalues with product = 1 (discrete);
 *                      solution computed using slightly perturbed values
 */
void sb03or(
    const bool discr,
    const bool ltrans,
    const i32 n,
    const i32 m,
    const f64* s,
    const i32 lds,
    const f64* a,
    const i32 lda,
    f64* c,
    const i32 ldc,
    f64* scale,
    i32* info);

/**
 * @brief Solve reduced Lyapunov equation for triangular factors.
 *
 * Solves for X = op(U)'*op(U) either the stable continuous-time Lyapunov equation:
 *
 *     op(S)'*X + X*op(S) = -scale^2*op(R)'*op(R)   (continuous)
 *
 * or the convergent discrete-time Lyapunov equation:
 *
 *     op(S)'*X*op(S) - X = -scale^2*op(R)'*op(R)   (discrete)
 *
 * where op(K) = K or K' (transpose), S is an N-by-N block upper triangular matrix
 * with 1x1 or 2x2 blocks on the diagonal (real Schur form), R is an N-by-N upper
 * triangular matrix. The output U is upper triangular and overwrites R. Scale is
 * an output scale factor set <= 1 to avoid overflow.
 *
 * For continuous-time: S must be stable (all eigenvalues have negative real parts).
 * For discrete-time: S must be convergent (all eigenvalues inside unit circle).
 *
 * Based on Bartels-Stewart backward substitution [1] finding Cholesky factor directly
 * without forming the normal matrix op(R)'*op(R) [2].
 *
 * References:
 * [1] Bartels, Stewart. Solution of A'X + XB = C. CACM 15, 820-826, 1972.
 * [2] Hammarling. Numerical solution of stable Lyapunov equation. IMA J. Num. Anal. 2, 303-325, 1982.
 *
 * @param[in] discr Equation type:
 *                  false = continuous: op(S)'*X + X*op(S) = -scale^2*op(R)'*op(R)
 *                  true = discrete: op(S)'*X*op(S) - X = -scale^2*op(R)'*op(R)
 * @param[in] ltrans Form of op(K):
 *                   false = op(K) = K (no transpose)
 *                   true = op(K) = K' (transpose)
 * @param[in] n Order of matrices S and R (n >= 0)
 * @param[in] s N-by-N block upper triangular matrix (real Schur form),
 *              dimension (lds,n). Upper Hessenberg part used; subdiagonal
 *              elements define 2x2 blocks (must correspond to complex
 *              conjugate eigenvalue pairs only).
 * @param[in] lds Leading dimension of S (lds >= max(1,n))
 * @param[in,out] r DOUBLE PRECISION array, dimension (ldr,n)
 *                  On entry: N-by-N upper triangular matrix R
 *                  On exit: N-by-N upper triangular Cholesky factor U
 * @param[in] ldr Leading dimension of R (ldr >= max(1,n))
 * @param[out] scale Scale factor (0 < scale <= 1) to prevent overflow
 * @param[out] dwork Workspace array, dimension (4*n)
 * @param[out] info Exit code:
 *                  0 = success
 *                  1 = near-singular (warning): perturbed values used
 *                  2 = S not stable (continuous) or not convergent (discrete)
 *                  3 = S has >2x2 diagonal block (consecutive non-zero subdiagonals)
 *                  4 = 2x2 block has real eigenvalues (requires complex conjugate)
 */
void sb03ot(
    const bool discr,
    const bool ltrans,
    const i32 n,
    f64* s,
    const i32 lds,
    f64* r,
    const i32 ldr,
    f64* scale,
    f64* dwork,
    i32* info);

/**
 * @brief Complex triangular Lyapunov equation solver for Cholesky factor.
 *
 * Solves for X = op(U)^H * op(U) either the stable continuous-time Lyapunov equation:
 *     op(S)^H * X + X * op(S) = -scale^2 * op(R)^H * op(R)
 * or the convergent discrete-time Lyapunov equation:
 *     op(S)^H * X * op(S) - X = -scale^2 * op(R)^H * op(R)
 *
 * where op(K) = K or K^H (conjugate transpose), S and R are complex N-by-N
 * upper triangular matrices, and scale is an output scale factor set <= 1
 * to avoid overflow in X. The diagonal elements of R must be real non-negative.
 *
 * For continuous-time, S must be stable (eigenvalues with negative real parts).
 * For discrete-time, S must be convergent (eigenvalue moduli < 1).
 *
 * @param[in] discr Equation type:
 *                  false = continuous-time
 *                  true = discrete-time
 * @param[in] ltrans Form of op(K):
 *                   false = op(K) = K
 *                   true = op(K) = K^H (conjugate transpose)
 * @param[in] n Order of matrices S and R (n >= 0)
 * @param[in] s COMPLEX*16 array, dimension (lds,n)
 *              N-by-N upper triangular matrix S
 * @param[in] lds Leading dimension of S (lds >= max(1,n))
 * @param[in,out] r COMPLEX*16 array, dimension (ldr,n)
 *                  On entry: N-by-N upper triangular matrix R with real
 *                           non-negative diagonal entries
 *                  On exit: N-by-N upper triangular Cholesky factor U with
 *                          real non-negative diagonal entries
 * @param[in] ldr Leading dimension of R (ldr >= max(1,n))
 * @param[out] scale Scale factor (0 < scale <= 1) to prevent overflow
 * @param[out] dwork DOUBLE PRECISION workspace array, dimension (n-1)
 * @param[out] zwork COMPLEX*16 workspace array, dimension (2*n-2)
 * @param[out] info Exit code:
 *                  0 = success
 *                  -i = the i-th argument had an illegal value
 *                  3 = S not stable (continuous) or not convergent (discrete)
 */
void sb03os(
    const bool discr,
    const bool ltrans,
    const i32 n,
    c128* s,
    const i32 lds,
    c128* r,
    const i32 ldr,
    f64* scale,
    f64* dwork,
    c128* zwork,
    i32* info);

/**
 * @brief Solve Lyapunov equation for Cholesky factor of solution.
 *
 * Computes upper triangular matrix U such that X = op(U)'*op(U) satisfies:
 *   Continuous: op(A)'*X + X*op(A) = -scale^2 * op(B)'*op(B)
 *   Discrete:   op(A)'*X*op(A) - X = -scale^2 * op(B)'*op(B)
 *
 * where A is in real Schur form, B is M-by-N (or N-by-M if ltrans),
 * and op(K) = K or K'.
 *
 * @param[in] discr If true, solve discrete equation; otherwise continuous
 * @param[in] ltrans If true, op(K)=K'; otherwise op(K)=K
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in] m Number of rows/columns in B (m >= 0)
 * @param[in] a N-by-N upper quasi-triangular matrix A in real Schur form
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] b On entry: M-by-N (or N-by-M if ltrans) matrix B
 *                  On exit: Modified during computation
 * @param[in] ldb Leading dimension of B
 * @param[out] tau Workspace for QR/RQ factorization, dimension min(m,n)
 * @param[out] u N-by-N upper triangular Cholesky factor U
 * @param[in] ldu Leading dimension of U (ldu >= max(1,n))
 * @param[out] scale Scale factor (0 < scale <= 1) to prevent overflow
 * @param[out] dwork Workspace array, dimension (ldwork)
 * @param[in] ldwork Length of dwork (ldwork >= max(1,4*n))
 * @param[out] info Exit code:
 *                  0 = success
 *                  1 = near-singular (warning)
 *                  2 = A not stable/convergent
 */
void sb03ou(
    const bool discr,
    const bool ltrans,
    const i32 n,
    const i32 m,
    const f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* tau,
    f64* u,
    const i32 ldu,
    f64* scale,
    f64* dwork,
    const i32 ldwork,
    i32* info);

/**
 * @brief Construct complex plane rotation for Lyapunov solver.
 *
 * Constructs a complex plane rotation such that, for a complex number a and
 * a real number b:
 *
 *     ( conjg(c)   s ) * ( a ) = ( d )
 *     (    -s      c )   ( b )   ( 0 )
 *
 * where d is always real and is overwritten on a, so that on return the
 * imaginary part of a is zero. b is unaltered.
 *
 * @param[in,out] a DOUBLE PRECISION array, dimension (2)
 *                  On entry: a[0] and a[1] are real and imaginary parts of a
 *                  On exit: a[0] contains real part of d, a[1] is set to zero
 * @param[in] b The real number b
 * @param[in] small A small real number. If norm d of [a; b] is smaller than
 *                  small, then the rotation is taken as unit matrix, and
 *                  a[0] and a[1] are set to d and 0, respectively.
 * @param[out] c DOUBLE PRECISION array, dimension (2)
 *               c[0] and c[1] are real and imaginary parts of complex cosine
 * @param[out] s The real sine of the plane rotation
 */
void sb03ov(f64* a, const f64 b, const f64 small, f64* c, f64* s);

/**
 * @brief Solve 2x2 Lyapunov equation for Cholesky factor.
 *
 * Solves for the Cholesky factor U of X, where op(U)'*op(U) = X, either
 * the continuous-time two-by-two Lyapunov equation:
 *
 *     op(S)'*X + X*op(S) = -ISGN*scale^2*op(R)'*op(R)     (DISCR=false)
 *
 * or the discrete-time two-by-two Lyapunov equation:
 *
 *     op(S)'*X*op(S) - X = -ISGN*scale^2*op(R)'*op(R)     (DISCR=true)
 *
 * where op(K) = K or K', S is 2x2 with complex conjugate eigenvalues,
 * R is 2x2 upper triangular, ISGN = -1 or 1, and scale is an output
 * scale factor set <= 1 to avoid overflow in X.
 *
 * Also computes matrices B and A so that:
 *   - B*U = U*S and A*U = scale^2*R (if LTRANS=false), or
 *   - U*B = S*U and U*A = scale^2*R (if LTRANS=true)
 *
 * For continuous-time (DISCR=false), ISGN*S must be stable (eigenvalues
 * have strictly negative real parts). For discrete-time (DISCR=true),
 * if ISGN=1, S must be convergent (eigenvalue moduli < 1); if ISGN=-1,
 * S must be completely divergent (eigenvalue moduli > 1).
 *
 * @param[in] discr Equation type:
 *                  false = continuous-time
 *                  true = discrete-time
 * @param[in] ltrans Form of op(K):
 *                   false = op(K) = K (no transpose)
 *                   true = op(K) = K' (transpose)
 * @param[in] isgn Sign of equation: +1 or -1
 * @param[in,out] s DOUBLE PRECISION array, dimension (lds,2)
 *                  On entry: 2x2 matrix S
 *                  On exit: 2x2 matrix B such that B*U = U*S (LTRANS=false)
 *                           or U*B = S*U (LTRANS=true)
 * @param[in] lds Leading dimension of S (lds >= 2)
 * @param[in,out] r DOUBLE PRECISION array, dimension (ldr,2)
 *                  On entry: 2x2 upper triangular matrix R (R[1,0] not referenced)
 *                  On exit: 2x2 upper triangular Cholesky factor U
 * @param[in] ldr Leading dimension of R (ldr >= 2)
 * @param[out] a DOUBLE PRECISION array, dimension (lda,2)
 *               2x2 upper triangular matrix A satisfying:
 *               A*U/scale = scale*R (LTRANS=false) or
 *               U*A/scale = scale*R (LTRANS=true)
 * @param[in] lda Leading dimension of A (lda >= 2)
 * @param[out] scale Scale factor (0 < scale <= 1) to prevent overflow
 * @param[out] info Exit code:
 *                  0 = success
 *                  1 = near-singular (warning): perturbed values used
 *                  2 = stability requirement not satisfied
 *                  4 = S has real eigenvalues (requires complex conjugate)
 *
 * @note In the interests of speed, this routine does not check all inputs
 *       for errors.
 */
void sb03oy(
    const bool discr,
    const bool ltrans,
    const i32 isgn,
    f64* s, const i32 lds,
    f64* r, const i32 ldr,
    f64* a, const i32 lda,
    f64* scale,
    i32* info);

/**
 * @brief Estimate forward error for discrete-time Lyapunov equation.
 *
 * Estimates forward error bound for solution X of:
 *     op(A)' * X * op(A) - X = C
 *
 * @param[in] trana 'N': op(A)=A, 'T'/'C': op(A)=A'
 * @param[in] uplo 'U': upper triangle of R, 'L': lower triangle
 * @param[in] lyapun 'O': transform with U, 'R': reduced only
 * @param[in] n Order of matrices (n >= 0)
 * @param[in] xanorm Infinity norm of solution X (xanorm >= 0)
 * @param[in] t Schur form matrix T, dimension (ldt,n)
 * @param[in] ldt Leading dimension of T
 * @param[in] u Orthogonal Schur matrix U (if lyapun='O'), dimension (ldu,n)
 * @param[in] ldu Leading dimension of U
 * @param[in,out] r Absolute residual matrix, dimension (ldr,n)
 * @param[in] ldr Leading dimension of R
 * @param[out] ferr Forward error bound
 * @param[out] iwork Integer workspace, dimension (n*n)
 * @param[out] dwork Workspace, dimension (ldwork)
 * @param[in] ldwork Length of dwork (>= max(3, 2*n*n) if n>0)
 * @param[out] info 0 on success, -i if arg i invalid, n+1 if nearly singular
 */
void sb03sx(const char* trana, const char* uplo, const char* lyapun, i32 n,
            f64 xanorm, const f64* t, i32 ldt, const f64* u, i32 ldu,
            f64* r, i32 ldr, f64* ferr, i32* iwork, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Estimate separation and Theta norm for discrete-time Lyapunov.
 *
 * Estimates sepd(op(A),op(A)') = min norm(op(A)'*X*op(A) - X)/norm(X)
 * and/or the 1-norm of operator Theta for discrete-time Lyapunov equation.
 *
 * @param[in] job 'S' for separation only, 'T' for Theta only, 'B' for both
 * @param[in] trana Specifies op(A): 'N' = A, 'T' or 'C' = A'
 * @param[in] lyapun 'O' for original form, 'R' for reduced form
 * @param[in] n Order of matrices (n >= 0)
 * @param[in] t N-by-N upper quasi-triangular Schur matrix, dimension (ldt,n)
 * @param[in] ldt Leading dimension of T (ldt >= max(1,n))
 * @param[in] u N-by-N orthogonal transformation matrix, dimension (ldu,n)
 * @param[in] ldu Leading dimension of U (ldu >= 1, >= n if lyapun='O')
 * @param[in] xa N-by-N matrix X*op(A) or U'*X*U*op(T), dimension (ldxa,n)
 * @param[in] ldxa Leading dimension of XA (ldxa >= 1, >= n if job != 'S')
 * @param[out] sepd Estimated separation (if job='S' or 'B')
 * @param[out] thnorm Estimated 1-norm of Theta (if job='T' or 'B')
 * @param[out] iwork Integer workspace of dimension n*n
 * @param[out] dwork Double workspace of dimension ldwork
 * @param[in] ldwork Length of dwork (>= max(3, 2*n*n) if n>0)
 * @param[out] info 0 on success, -i if arg i invalid, n+1 if nearly singular
 */
void sb03sy(const char* job, const char* trana, const char* lyapun, i32 n,
            const f64* t, i32 ldt, const f64* u, i32 ldu, const f64* xa, i32 ldxa,
            f64* sepd, f64* thnorm, i32* iwork, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Solve discrete-time Lyapunov equation with conditioning/error bounds.
 *
 * Solves the real discrete-time Lyapunov matrix equation:
 *   op(A)'*X*op(A) - X = scale*C
 * where op(A) = A or A' (transpose), C and X are N-by-N symmetric matrices.
 * Optionally estimates the conditioning and computes an error bound.
 *
 * @param[in] job Computation to perform:
 *                'X' = solution only,
 *                'S' = separation only,
 *                'C' = reciprocal condition number only,
 *                'E' = error bound only,
 *                'A' = compute all.
 * @param[in] fact 'F' = Schur factorization provided, 'N' = compute factorization.
 * @param[in] trana 'N' for op(A)=A, 'T' or 'C' for op(A)=A'.
 * @param[in] uplo 'U' for upper triangular C, 'L' for lower triangular C.
 * @param[in] lyapun 'O' for original equation, 'R' for reduced (Schur form).
 * @param[in] n Order of matrices A, X, C. n >= 0.
 * @param[in,out] scale If job='C' or 'E', input scale factor (0 <= scale <= 1).
 *                      If job='X' or 'A', output scale factor to prevent overflow.
 * @param[in] a N-by-N matrix A (if fact='N' or lyapun='O' and job!='X').
 * @param[in] lda Leading dimension of a.
 * @param[in,out] t N-by-N Schur form T. Input if fact='F', output if fact='N'.
 * @param[in] ldt Leading dimension of t. ldt >= max(1,n).
 * @param[in,out] u N-by-N orthogonal matrix U from Schur factorization.
 *                  Input if lyapun='O' and fact='F', output if lyapun='O' and fact='N'.
 *                  Not referenced if lyapun='R'.
 * @param[in] ldu Leading dimension of u. ldu >= 1 (>= n if lyapun='O').
 * @param[in] c N-by-N symmetric matrix C (specified triangle only).
 *              Not referenced if job='S'.
 * @param[in] ldc Leading dimension of c. ldc >= 1 (>= n if job!='S').
 * @param[in,out] x N-by-N solution matrix X. Input if job='C' or 'E',
 *                  output if job='X' or 'A'. Not referenced if job='S'.
 * @param[in] ldx Leading dimension of x. ldx >= 1 (>= n if job!='S').
 * @param[out] sepd Estimated separation sepd(op(A),op(A)') if job='S','C', or 'A'.
 * @param[out] rcond Estimated reciprocal condition number if job='C' or 'A'.
 * @param[out] ferr Estimated forward error bound if job='E' or 'A'.
 * @param[out] wr N-element array of real parts of eigenvalues (if fact='N').
 * @param[out] wi N-element array of imaginary parts of eigenvalues (if fact='N').
 * @param[out] iwork Integer workspace of dimension n*n. Not used if job='X'.
 * @param[out] dwork Double workspace.
 * @param[in] ldwork Workspace size. Requirements depend on job and fact.
 * @param[out] info 0 on success, -i if argument i invalid,
 *                  i (1<=i<=n) if Schur decomposition failed,
 *                  n+1 if near-reciprocal eigenvalues (perturbed values used).
 */
void sb03ud(const char* job, const char* fact, const char* trana,
            const char* uplo, const char* lyapun, i32 n, f64* scale,
            f64* a, i32 lda, f64* t, i32 ldt, f64* u, i32 ldu,
            f64* c, i32 ldc, f64* x, i32 ldx, f64* sepd, f64* rcond,
            f64* ferr, f64* wr, f64* wi, i32* iwork, f64* dwork,
            i32 ldwork, i32* info);

/**
 * @brief Solve continuous-time Sylvester equation AX + XB = C (Hessenberg-Schur).
 *
 * Solves the continuous-time Sylvester equation AX + XB = C where A, B, C, X are
 * N-by-N, M-by-M, N-by-M, and N-by-M matrices respectively. Uses Hessenberg-Schur
 * method: A is reduced to upper Hessenberg form H = U'AU, B' is reduced to real
 * Schur form S = Z'B'Z. The transformed system HY + YS' = F is solved by back
 * substitution, then X = UYZ'.
 *
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in] m Order of matrix B (m >= 0)
 * @param[in,out] a On entry: N-by-N coefficient matrix A
 *                  On exit: upper Hessenberg form H and orthogonal U (factored)
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] b On entry: M-by-M coefficient matrix B
 *                  On exit: quasi-triangular Schur factor S of B'
 * @param[in] ldb Leading dimension of B (ldb >= max(1,m))
 * @param[in,out] c On entry: N-by-M coefficient matrix C
 *                  On exit: N-by-M solution matrix X
 * @param[in] ldc Leading dimension of C (ldc >= max(1,n))
 * @param[out] z M-by-M orthogonal matrix Z transforming B' to Schur form
 * @param[in] ldz Leading dimension of Z (ldz >= max(1,m))
 * @param[out] iwork Integer workspace, dimension (4*n)
 * @param[out] dwork Double workspace, dimension (ldwork)
 *                   On exit: dwork[0] = optimal ldwork, dwork[1..n-1] = tau from DGEHRD
 * @param[in] ldwork Workspace size: max(1, 2*n*n + 8*n, 5*m, n + m)
 *                   If ldwork=-1, workspace query (returns optimal in dwork[0])
 * @param[out] info Exit code:
 *                  0 = success
 *                  < 0 = -i means i-th argument invalid
 *                  1..m = QR algorithm failed in DGEES
 *                  > m = singular matrix in column (info-m)
 */
void sb04md(i32 n, i32 m, f64* a, i32 lda, f64* b, i32 ldb,
            f64* c, i32 ldc, f64* z, i32 ldz,
            i32* iwork, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Solve linear system with compact storage (second subdiagonal zeros).
 *
 * Solves a linear algebraic system of order M whose coefficient matrix
 * has zeros below the second subdiagonal. The matrix is stored compactly,
 * row-wise. Uses Gaussian elimination with partial pivoting.
 *
 * @param[in] m Order of the system (m >= 0).
 *               Note: m should have twice the value in the original problem.
 * @param[in,out] d Compact array, dimension (m*(m+1)/2 + 3*m).
 *                  On entry: first m*(m+1)/2+2*m elements are coefficient matrix,
 *                  next m elements are RHS.
 *                  On exit: last m elements contain solution (indices in ipr).
 * @param[out] ipr Integer array, dimension (2*m).
 *                 First m elements: solution component indices.
 * @param[out] info Exit code: 0 = success, 1 = singular matrix.
 */
void sb04mr(const i32 m, f64* d, i32* ipr, i32* info);

/**
 * @brief Construct and solve linear system for 2x2 blocks (continuous-time).
 *
 * Constructs and solves a linear algebraic system of order 2*M whose coefficient
 * matrix has zeros below the second subdiagonal. Such systems appear when
 * solving continuous-time Sylvester equations using the Hessenberg-Schur method.
 *
 * @param[in] n Order of matrix B (n >= 0)
 * @param[in] m Order of matrix A (m >= 0)
 * @param[in] ind IND and IND-1 specify column indices in C to compute (ind > 1)
 * @param[in] a Upper Hessenberg matrix, dimension (lda,m)
 * @param[in] lda Leading dimension of A (lda >= max(1,m))
 * @param[in] b Real Schur matrix, dimension (ldb,n)
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in,out] c On entry: coefficient matrix, dimension (ldc,n)
 *                  On exit: columns IND-1 and IND updated
 * @param[in] ldc Leading dimension of C (ldc >= max(1,m))
 * @param[out] d Workspace, dimension (2*m*m + 7*m)
 * @param[out] ipr Integer workspace, dimension (4*m)
 * @param[out] info 0 = success, IND = singular matrix
 */
void sb04mu(i32 n, i32 m, i32 ind, const f64* a, i32 lda,
            const f64* b, i32 ldb, f64* c, i32 ldc,
            f64* d, i32* ipr, i32* info);

/**
 * @brief Solve linear system with compact storage (upper Hessenberg).
 *
 * Solves a linear algebraic system of order M whose coefficient matrix
 * is in upper Hessenberg form, stored compactly, row-wise.
 * Uses Gaussian elimination with partial pivoting.
 *
 * @param[in] m Order of the system (m >= 0).
 * @param[in,out] d Compact array, dimension (m*(m+1)/2 + 2*m).
 *                  On entry: first m*(m+1)/2+m elements are coefficient matrix,
 *                  next m elements are RHS.
 *                  On exit: last m elements contain solution (indices in ipr).
 * @param[out] ipr Integer array, dimension (2*m).
 *                 First m elements: solution component indices.
 * @param[out] info Exit code: 0 = success, 1 = singular matrix.
 */
void sb04mw(const i32 m, f64* d, i32* ipr, i32* info);

/**
 * @brief Construct and solve linear system for 1x1 blocks (continuous-time).
 *
 * Constructs and solves a linear algebraic system of order M whose coefficient
 * matrix is in upper Hessenberg form. Such systems appear when solving
 * Sylvester equations using the Hessenberg-Schur method.
 *
 * @param[in] n Order of matrix B (n >= 0)
 * @param[in] m Order of matrix A (m >= 0)
 * @param[in] ind Index of column in C to compute (ind >= 1)
 * @param[in] a Upper Hessenberg matrix, dimension (lda,m)
 * @param[in] lda Leading dimension of A (lda >= max(1,m))
 * @param[in] b Real Schur matrix, dimension (ldb,n)
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in,out] c On entry: coefficient matrix, dimension (ldc,n)
 *                  On exit: column IND updated
 * @param[in] ldc Leading dimension of C (ldc >= max(1,m))
 * @param[out] d Workspace, dimension (m*(m+1)/2 + 2*m)
 * @param[out] ipr Integer workspace, dimension (2*m)
 * @param[out] info 0 = success, IND = singular matrix
 */
void sb04my(i32 n, i32 m, i32 ind, const f64* a, i32 lda,
            const f64* b, i32 ldb, f64* c, i32 ldc,
            f64* d, i32* ipr, i32* info);

/**
 * @brief Solve continuous-time Sylvester equation AX + XB = C.
 *
 * Solves the continuous-time Sylvester equation AX + XB = C, where at least
 * one of the matrices A or B is in Schur form and the other is in Hessenberg
 * or Schur form (both either upper or lower).
 *
 * Uses the Hessenberg-Schur back substitution method proposed by
 * Golub, Nash and Van Loan.
 *
 * @param[in] abschu Specifies which matrix is in Schur form:
 *                   'A' = A is in Schur form, B is in Hessenberg form
 *                   'B' = B is in Schur form, A is in Hessenberg form
 *                   'S' = Both A and B are in Schur form
 * @param[in] ula 'U' for upper Hessenberg/Schur, 'L' for lower
 * @param[in] ulb 'U' for upper Hessenberg/Schur, 'L' for lower
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in] m Order of matrix B (m >= 0)
 * @param[in] a N-by-N coefficient matrix A, dimension (lda,n)
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in] b M-by-M coefficient matrix B, dimension (ldb,m)
 * @param[in] ldb Leading dimension of B (ldb >= max(1,m))
 * @param[in,out] c On entry: N-by-M coefficient matrix C
 *                  On exit: N-by-M solution matrix X
 * @param[in] ldc Leading dimension of C (ldc >= max(1,n))
 * @param[in] tol Tolerance for near-singularity test.
 *                If tol <= 0, default EPS is used.
 *                Not referenced if abschu='S', ula='U', ulb='U'.
 * @param[out] iwork Integer workspace, dimension (2*max(m,n)).
 *                   Not referenced if abschu='S', ula='U', ulb='U'.
 * @param[out] dwork Double workspace, dimension (ldwork).
 *                   Not referenced if abschu='S', ula='U', ulb='U'.
 * @param[in] ldwork Workspace size.
 *                   If abschu='S', ula='U', ulb='U': ldwork = 0
 *                   Otherwise: ldwork = 2*max(m,n)*(4 + 2*max(m,n))
 * @param[out] info Exit code:
 *                  0 = success
 *                  < 0 = -i means i-th argument invalid
 *                  1 = numerically singular matrix encountered
 */
void sb04nd(
    const char* abschu,
    const char* ula,
    const char* ulb,
    const i32 n,
    const i32 m,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    const f64 tol,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info);

/**
 * @brief Construct RHS for Sylvester equation solver (2 RHS case).
 *
 * Constructs the right-hand sides D for a system of equations in
 * Hessenberg form solved via SB04NX.
 *
 * @param[in] abschr 'A' if AB contains A, 'B' if AB contains B
 * @param[in] ul 'U' if AB is upper Hessenberg, 'L' if lower
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in] m Order of matrix B (m >= 0)
 * @param[in] c N-by-M matrix containing coefficient matrix C and
 *              currently computed part of solution
 * @param[in] ldc Leading dimension of C (ldc >= max(1,n))
 * @param[in] indx Position of first column/row of C to use (1-based)
 * @param[in] ab N-by-N or M-by-M matrix (A or B)
 * @param[in] ldab Leading dimension of AB
 * @param[out] d Right-hand side stored as matrix with two rows,
 *               dimension 2*N or 2*M
 */
void sb04nv(
    const char* abschr,
    const char* ul,
    const i32 n,
    const i32 m,
    const f64* c,
    const i32 ldc,
    const i32 indx,
    const f64* ab,
    const i32 ldab,
    f64* d);

/**
 * @brief Construct RHS for Sylvester equation solver (1 RHS case).
 *
 * Constructs the right-hand side D for a system of equations in
 * Hessenberg form solved via SB04NY.
 *
 * @param[in] abschr 'A' if AB contains A, 'B' if AB contains B
 * @param[in] ul 'U' if AB is upper Hessenberg, 'L' if lower
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in] m Order of matrix B (m >= 0)
 * @param[in] c N-by-M matrix containing coefficient matrix C and
 *              currently computed part of solution
 * @param[in] ldc Leading dimension of C (ldc >= max(1,n))
 * @param[in] indx Position of column/row of C to use (1-based)
 * @param[in] ab N-by-N or M-by-M matrix (A or B)
 * @param[in] ldab Leading dimension of AB
 * @param[out] d Right-hand side vector, dimension N or M
 */
void sb04nw(
    const char* abschr,
    const char* ul,
    const i32 n,
    const i32 m,
    const f64* c,
    const i32 ldc,
    const i32 indx,
    const f64* ab,
    const i32 ldab,
    f64* d);

/**
 * @brief Solve Hessenberg system with two offdiagonals and two RHS.
 *
 * Solves a system of equations in Hessenberg form with two consecutive
 * offdiagonals and two right-hand sides.
 *
 * @param[in] rc 'R' for row transformations, 'C' for column
 * @param[in] ul 'U' if matrix is upper Hessenberg, 'L' if lower
 * @param[in] m Order of matrix A (m >= 0)
 * @param[in] a M-by-M Hessenberg matrix, dimension (lda,m)
 * @param[in] lda Leading dimension of A (lda >= max(1,m))
 * @param[in] lambd1 Element (1,1) of 2x2 block to add to diagonal
 * @param[in] lambd2 Element (1,2) of 2x2 block to add to diagonal
 * @param[in] lambd3 Element (2,1) of 2x2 block to add to diagonal
 * @param[in] lambd4 Element (2,2) of 2x2 block to add to diagonal
 * @param[in,out] d On entry: two RHS stored row-wise.
 *                  On exit: two solution vectors stored row-wise
 * @param[in] tol Tolerance for near-singularity test
 * @param[out] iwork Integer workspace, dimension (2*m)
 * @param[out] dwork Double workspace, dimension (lddwor,2*m+3)
 * @param[in] lddwor Leading dimension of dwork (lddwor >= max(1,2*m))
 * @param[out] info 0 = success, 1 = numerically singular
 */
void sb04nx(
    const char* rc,
    const char* ul,
    const i32 m,
    const f64* a,
    const i32 lda,
    const f64 lambd1,
    const f64 lambd2,
    const f64 lambd3,
    const f64 lambd4,
    f64* d,
    const f64 tol,
    i32* iwork,
    f64* dwork,
    const i32 lddwor,
    i32* info);

/**
 * @brief Solve Hessenberg system with one offdiagonal and one RHS.
 *
 * Solves a system of equations in Hessenberg form with one offdiagonal
 * and one right-hand side.
 *
 * @param[in] rc 'R' for row transformations, 'C' for column
 * @param[in] ul 'U' if matrix is upper Hessenberg, 'L' if lower
 * @param[in] m Order of matrix A (m >= 0)
 * @param[in] a M-by-M Hessenberg matrix, dimension (lda,m)
 * @param[in] lda Leading dimension of A (lda >= max(1,m))
 * @param[in] lambda Value to add to diagonal elements
 * @param[in,out] d On entry: RHS vector. On exit: solution vector
 * @param[in] tol Tolerance for near-singularity test
 * @param[out] iwork Integer workspace, dimension (m)
 * @param[out] dwork Double workspace, dimension (lddwor,m+3)
 * @param[in] lddwor Leading dimension of dwork (lddwor >= max(1,m))
 * @param[out] info 0 = success, 1 = numerically singular
 */
void sb04ny(
    const char* rc,
    const char* ul,
    const i32 m,
    const f64* a,
    const i32 lda,
    const f64 lambda,
    f64* d,
    const f64 tol,
    i32* iwork,
    f64* dwork,
    const i32 lddwor,
    i32* info);

/**
 * @brief Solve small (N1xN2, 1<=N1,N2<=2) Sylvester equation.
 *
 * Solves for the N1-by-N2 matrix X (1 <= N1,N2 <= 2) in:
 *
 *     op(TL)*X*op(TR) + ISGN*X = SCALE*B
 *
 * where TL is N1-by-N1, TR is N2-by-N2, B is N1-by-N2, ISGN = 1 or -1,
 * and op(T) = T or T' (transpose).
 *
 * Uses Gaussian elimination with complete pivoting.
 *
 * @param[in] ltranl If true, use TL', otherwise use TL
 * @param[in] ltranr If true, use TR', otherwise use TR
 * @param[in] isgn Sign of equation: +1 or -1
 * @param[in] n1 Order of matrix TL (0, 1, or 2)
 * @param[in] n2 Order of matrix TR (0, 1, or 2)
 * @param[in] tl N1-by-N1 matrix TL, dimension (ldtl,n1)
 * @param[in] ldtl Leading dimension of TL (ldtl >= max(1,n1))
 * @param[in] tr N2-by-N2 matrix TR, dimension (ldtr,n2)
 * @param[in] ldtr Leading dimension of TR (ldtr >= max(1,n2))
 * @param[in] b N1-by-N2 right-hand side matrix B, dimension (ldb,n2)
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n1))
 * @param[out] scale Scale factor (0 < scale <= 1) to prevent overflow
 * @param[out] x N1-by-N2 solution matrix X, dimension (ldx,n2)
 *               Note: X may be identified with B in the calling statement
 * @param[in] ldx Leading dimension of X (ldx >= max(1,n1))
 * @param[out] xnorm Infinity-norm of the solution X
 * @param[out] info Exit code:
 *                  0 = success
 *                  1 = TL and -ISGN*TR have almost reciprocal eigenvalues,
 *                      so TL or TR is perturbed to get nonsingular equation
 *
 * @note This routine does not check inputs for errors (for speed).
 */
void sb04px(
    const bool ltranl,
    const bool ltranr,
    const i32 isgn,
    const i32 n1,
    const i32 n2,
    const f64* tl,
    const i32 ldtl,
    const f64* tr,
    const i32 ldtr,
    const f64* b,
    const i32 ldb,
    f64* scale,
    f64* x,
    const i32 ldx,
    f64* xnorm,
    i32* info);

/**
 * @brief Solve continuous-time or discrete-time Sylvester equations.
 *
 * Solves for X either the real continuous-time Sylvester equation:
 *   op(A)*X + ISGN*X*op(B) = scale*C
 *
 * or the real discrete-time Sylvester equation:
 *   op(A)*X*op(B) + ISGN*X = scale*C
 *
 * where op(M) = M or M**T, and ISGN = 1 or -1. A is M-by-M and B is N-by-N;
 * the right hand side C and the solution X are M-by-N; and scale is an
 * output scale factor, set less than or equal to 1 to avoid overflow in X.
 *
 * If A and/or B are not quasi-triangular, they are reduced to Schur canonical
 * form: A = U*S*U', B = V*T*V'.
 *
 * @param[in] dico 'C' for continuous-time, 'D' for discrete-time
 * @param[in] facta 'F' Schur form supplied, 'N' compute Schur, 'S' already Schur
 * @param[in] factb 'F' Schur form supplied, 'N' compute Schur, 'S' already Schur
 * @param[in] trana 'N' for A, 'T' or 'C' for A**T
 * @param[in] tranb 'N' for B, 'T' or 'C' for B**T
 * @param[in] isgn Sign of the equation (must be 1 or -1)
 * @param[in] m Order of matrix A (m >= 0)
 * @param[in] n Order of matrix B (n >= 0)
 * @param[in,out] a On entry: M-by-M matrix A
 *                  On exit: Schur form if facta='N'
 * @param[in] lda Leading dimension of A (lda >= max(1,m))
 * @param[in,out] u On entry: Schur vectors if facta='F'
 *                  On exit: Schur vectors if facta='N'
 * @param[in] ldu Leading dimension of U
 * @param[in,out] b On entry: N-by-N matrix B
 *                  On exit: Schur form if factb='N'
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in,out] v On entry: Schur vectors if factb='F'
 *                  On exit: Schur vectors if factb='N'
 * @param[in] ldv Leading dimension of V
 * @param[in,out] c On entry: M-by-N right-hand side C
 *                  On exit: M-by-N solution X
 * @param[in] ldc Leading dimension of C (ldc >= max(1,m))
 * @param[out] scale Scale factor (0 < scale <= 1)
 * @param[out] dwork Workspace array
 * @param[in] ldwork Workspace size
 * @param[out] info Exit code: 0=success, <0=argument error, >0=algorithm error
 */
void sb04pd(
    const char dico,
    const char facta,
    const char factb,
    const char trana,
    const char tranb,
    const i32 isgn,
    const i32 m,
    const i32 n,
    f64* a,
    const i32 lda,
    f64* u,
    const i32 ldu,
    f64* b,
    const i32 ldb,
    f64* v,
    const i32 ldv,
    f64* c,
    const i32 ldc,
    f64* scale,
    f64* dwork,
    const i32 ldwork,
    i32* info);

/**
 * @brief Solve discrete-time Sylvester equation with Schur matrices.
 *
 * Solves for X the discrete-time Sylvester equation:
 *   op(A)*X*op(B) + ISGN*X = scale*C
 *
 * where op(A) = A or A**T, A and B are both upper quasi-triangular,
 * and ISGN = 1 or -1. A is M-by-M and B is N-by-N; the right hand
 * side C and the solution X are M-by-N; and scale is an output scale
 * factor, set less than or equal to 1 to avoid overflow in X. The
 * solution matrix X is overwritten onto C.
 *
 * A and B must be in Schur canonical form (block upper triangular with
 * 1-by-1 and 2-by-2 diagonal blocks).
 *
 * @param[in] trana 'N' for A, 'T' or 'C' for A**T
 * @param[in] tranb 'N' for B, 'T' or 'C' for B**T
 * @param[in] isgn Sign of the equation (must be 1 or -1)
 * @param[in] m Order of matrix A and number of rows in C (m >= 0)
 * @param[in] n Order of matrix B and number of columns in C (n >= 0)
 * @param[in] a M-by-M upper quasi-triangular matrix A (Schur form)
 * @param[in] lda Leading dimension of A (lda >= max(1,m))
 * @param[in] b N-by-N upper quasi-triangular matrix B (Schur form)
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in,out] c On entry: M-by-N right-hand side matrix C
 *                  On exit: M-by-N solution matrix X
 * @param[in] ldc Leading dimension of C (ldc >= max(1,m))
 * @param[out] scale Scale factor (0 < scale <= 1) to prevent overflow
 * @param[out] dwork Double workspace, dimension (2*m)
 * @param[out] info Exit code:
 *                  0 = success
 *                  < 0 = -i means i-th argument invalid
 *                  1 = A and -ISGN*B have almost reciprocal eigenvalues
 */
void sb04py(
    const char trana,
    const char tranb,
    const i32 isgn,
    const i32 m,
    const i32 n,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* scale,
    f64* dwork,
    i32* info);

/**
 * @brief Solve discrete-time Sylvester equation X + AXB = C (Hessenberg-Schur).
 *
 * Solves the discrete-time Sylvester equation X + AXB = C where A, B, C, X are
 * N-by-N, M-by-M, N-by-M, and N-by-M matrices respectively. Uses Hessenberg-Schur
 * method: A is reduced to upper Hessenberg form H = U'AU, B' is reduced to real
 * Schur form S = Z'B'Z. The transformed system Y + HYS' = F is solved by back
 * substitution, then X = UYZ'.
 *
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in] m Order of matrix B (m >= 0)
 * @param[in,out] a On entry: N-by-N coefficient matrix A
 *                  On exit: upper Hessenberg form H and orthogonal U (factored)
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] b On entry: M-by-M coefficient matrix B
 *                  On exit: quasi-triangular Schur factor S of B'
 * @param[in] ldb Leading dimension of B (ldb >= max(1,m))
 * @param[in,out] c On entry: N-by-M coefficient matrix C
 *                  On exit: N-by-M solution matrix X
 * @param[in] ldc Leading dimension of C (ldc >= max(1,n))
 * @param[out] z M-by-M orthogonal matrix Z transforming B' to Schur form
 * @param[in] ldz Leading dimension of Z (ldz >= max(1,m))
 * @param[out] iwork Integer workspace, dimension (4*n)
 * @param[out] dwork Double workspace, dimension (ldwork)
 *                   On exit: dwork[0] = optimal ldwork, dwork[1..n-1] = tau from DGEHRD
 * @param[in] ldwork Workspace size: max(1, 2*n*n + 9*n, 5*m, n + m)
 *                   If ldwork=-1, workspace query (returns optimal in dwork[0])
 * @param[out] info Exit code:
 *                  0 = success
 *                  < 0 = -i means i-th argument invalid
 *                  1..m = QR algorithm failed in DGEES
 *                  > m = singular matrix in column (info-m)
 */
void sb04qd(i32 n, i32 m, f64* a, i32 lda, f64* b, i32 ldb,
            f64* c, i32 ldc, f64* z, i32 ldz,
            i32* iwork, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Solve linear system with compact storage (third subdiagonal pattern).
 *
 * Solves a linear algebraic system of order M whose coefficient matrix
 * has zeros below the third subdiagonal and zero elements on the third
 * subdiagonal with even column indices. Matrix stored compactly, row-wise.
 * Uses Gaussian elimination with partial pivoting.
 *
 * @param[in] m Order of the system (m >= 0, m even).
 *               Note: m should have twice the value in the original problem.
 * @param[in,out] d Compact array, dimension (m*m/2 + 4*m).
 *                  On entry: first m*m/2+3*m elements are coefficient matrix,
 *                  next m elements are RHS.
 *                  On exit: last m elements contain solution (indices in ipr).
 * @param[out] ipr Integer array, dimension (2*m).
 *                 First m elements: solution component indices.
 * @param[out] info Exit code: 0 = success, 1 = singular matrix.
 */
void sb04qr(const i32 m, f64* d, i32* ipr, i32* info);

/**
 * @brief Construct and solve linear system for 2x2 blocks (discrete-time).
 *
 * Constructs and solves a linear algebraic system of order 2*M whose coefficient
 * matrix has zeros below the third subdiagonal, and zero elements on the third
 * subdiagonal with even column indices. Such systems appear when solving
 * discrete-time Sylvester equations using the Hessenberg-Schur method.
 *
 * @param[in] n Order of matrix B (n >= 0)
 * @param[in] m Order of matrix A (m >= 0)
 * @param[in] ind IND and IND-1 specify column indices in C to compute (ind > 1)
 * @param[in] a Upper Hessenberg matrix, dimension (lda,m)
 * @param[in] lda Leading dimension of A (lda >= max(1,m))
 * @param[in] b Real Schur matrix, dimension (ldb,n)
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in,out] c On entry: coefficient matrix, dimension (ldc,n)
 *                  On exit: columns IND-1 and IND updated
 * @param[in] ldc Leading dimension of C (ldc >= max(1,m))
 * @param[out] d Workspace, dimension (2*m*m + 8*m)
 * @param[out] ipr Integer workspace, dimension (4*m)
 * @param[out] info 0 = success, IND = singular matrix
 */
void sb04qu(i32 n, i32 m, i32 ind, const f64* a, i32 lda,
            const f64* b, i32 ldb, f64* c, i32 ldc,
            f64* d, i32* ipr, i32* info);

/**
 * @brief Construct and solve linear system for 1x1 blocks (discrete-time).
 *
 * Constructs and solves a linear algebraic system of order M whose coefficient
 * matrix is in upper Hessenberg form. Such systems appear when solving
 * discrete-time Sylvester equations using the Hessenberg-Schur method.
 *
 * @param[in] n Order of matrix B (n >= 0)
 * @param[in] m Order of matrix A (m >= 0)
 * @param[in] ind Index of column in C to compute (ind >= 1)
 * @param[in] a Upper Hessenberg matrix, dimension (lda,m)
 * @param[in] lda Leading dimension of A (lda >= max(1,m))
 * @param[in] b Real Schur matrix, dimension (ldb,n)
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in,out] c On entry: coefficient matrix, dimension (ldc,n)
 *                  On exit: column IND updated
 * @param[in] ldc Leading dimension of C (ldc >= max(1,m))
 * @param[out] d Workspace, dimension (m*(m+1)/2 + 2*m)
 * @param[out] ipr Integer workspace, dimension (2*m)
 * @param[out] info 0 = success, IND = singular matrix
 */
void sb04qy(i32 n, i32 m, i32 ind, const f64* a, i32 lda,
            const f64* b, i32 ldb, f64* c, i32 ldc,
            f64* d, i32* ipr, i32* info);

/**
 * @brief Solve discrete-time Sylvester equation (Hessenberg-Schur method).
 *
 * Solves the discrete-time Sylvester equation:
 *     X + A*X*B = C
 *
 * where A is N-by-N, B is M-by-M, C and X are N-by-M. At least one of A or B
 * must be in Schur form, the other in Hessenberg or Schur form.
 *
 * @param[in] abschu 'A' = A in Schur, B in Hessenberg
 *                   'B' = B in Schur, A in Hessenberg
 *                   'S' = both A and B in Schur form
 * @param[in] ula 'U' = A is upper (Hessenberg if abschu='B', else Schur)
 *                'L' = A is lower
 * @param[in] ulb 'U' = B is upper (Hessenberg if abschu='A', else Schur)
 *                'L' = B is lower
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in] m Order of matrix B (m >= 0)
 * @param[in] a N-by-N coefficient matrix A
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in] b M-by-M coefficient matrix B
 * @param[in] ldb Leading dimension of B (ldb >= max(1,m))
 * @param[in,out] c On entry: N-by-M RHS matrix C
 *                  On exit: solution matrix X
 * @param[in] ldc Leading dimension of C (ldc >= max(1,n))
 * @param[in] tol Tolerance for singularity test; if <= 0, machine epsilon used
 * @param[out] iwork Integer workspace, dimension 2*max(m,n)
 *                   Not referenced if abschu='S', ula='U', ulb='U'
 * @param[out] dwork Real workspace, dimension ldwork
 * @param[in] ldwork Length of dwork:
 *                   2*n if abschu='S', ula='U', ulb='U'
 *                   2*max(m,n)*(4 + 2*max(m,n)) otherwise
 * @param[out] info 0 = success, 1 = nearly singular matrix encountered
 */
void sb04rd(
    const char* abschu,
    const char* ula,
    const char* ulb,
    const i32 n,
    const i32 m,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    const f64 tol,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info);

/**
 * @brief Left coprime factorization with inner denominator.
 *
 * Constructs, for a given system G = (A,B,C,D), an output injection matrix H,
 * an orthogonal transformation matrix Z, and a gain matrix V, such that
 * Q = (Z'*(A+H*C)*Z, Z'*(B+H*D), V*C*Z, V*D) and
 * R = (Z'*(A+H*C)*Z, Z'*H, V*C*Z, V) provide a stable left coprime
 * factorization of G in the form G = R^{-1} * Q, where R is co-inner.
 *
 * @param[in] dico 'C' for continuous-time, 'D' for discrete-time
 * @param[in] n State dimension (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] p Number of outputs (p >= 0)
 * @param[in,out] a On entry: N-by-N state matrix.
 *                  On exit: NQ-by-NQ numerator state matrix in real Schur form.
 * @param[in] lda Leading dimension of a. lda >= max(1,n).
 * @param[in,out] b On entry: N-by-M input matrix.
 *                  On exit: NQ-by-M numerator input matrix.
 * @param[in] ldb Leading dimension of b. ldb >= max(1,n).
 * @param[in,out] c On entry: P-by-N output matrix.
 *                  On exit: P-by-NQ numerator output matrix.
 * @param[in] ldc Leading dimension of c. ldc >= max(1,m,p) if n>0; else >= 1.
 * @param[in,out] d On entry: P-by-M feedthrough matrix.
 *                  On exit: P-by-M numerator feedthrough matrix.
 * @param[in] ldd Leading dimension of d. ldd >= max(1,m,p).
 * @param[out] nq Order of the resulting factors Q and R.
 * @param[out] nr Order of the minimal realization of factor R.
 * @param[out] br NQ-by-P output injection matrix Z'*H.
 * @param[in] ldbr Leading dimension of br. ldbr >= max(1,n).
 * @param[out] dr P-by-P lower triangular matrix V (feedthrough of R).
 * @param[in] lddr Leading dimension of dr. lddr >= max(1,p).
 * @param[in] tol Tolerance for observability test. If <= 0, default is used.
 * @param[out] dwork Workspace array.
 * @param[in] ldwork Length of dwork. ldwork >= max(1, p*n + max(n*(n+5),p*(p+2),4*p,4*m)).
 * @param[out] iwarn Warning: K violations of numerical stability condition.
 * @param[out] info 0=success, <0=parameter error, 1=Schur fail, 2=ordering fail,
 *                  3=observable eigenvalue on stability boundary.
 */
void sb08cd(
    const char* dico,
    const i32 n,
    const i32 m,
    const i32 p,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* d,
    const i32 ldd,
    i32* nq,
    i32* nr,
    f64* br,
    const i32 ldbr,
    f64* dr,
    const i32 lddr,
    const f64 tol,
    f64* dwork,
    const i32 ldwork,
    i32* iwarn,
    i32* info);

/**
 * @brief Right coprime factorization with inner denominator.
 *
 * Computes a right coprime factorization with inner denominator
 * G = N*M^(-1) of a given state-space representation (A,B,C,D).
 *
 * @param[in] dico 'C' for continuous-time, 'D' for discrete-time
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] p Number of outputs (p >= 0)
 * @param[in,out] a N-by-N state matrix, modified on exit
 * @param[in] lda Leading dimension of A
 * @param[in,out] b N-by-M input matrix, modified on exit
 * @param[in] ldb Leading dimension of B
 * @param[in,out] c P-by-N output matrix, modified on exit
 * @param[in] ldc Leading dimension of C
 * @param[in,out] d P-by-M feedthrough matrix, modified on exit
 * @param[in] ldd Leading dimension of D
 * @param[out] nq Order of computed stable coprime factorization
 * @param[out] nr Order of reduced antistable part
 * @param[out] cr M-by-N output matrix of denominator
 * @param[in] ldcr Leading dimension of CR
 * @param[out] dr M-by-M feedthrough of denominator
 * @param[in] lddr Leading dimension of DR
 * @param[in] tol Tolerance for rank determination
 * @param[out] dwork Workspace array
 * @param[in] ldwork Length of dwork
 * @param[out] iwarn Warning indicator
 * @param[out] info Exit code
 */
void sb08dd(
    const char* dico,
    const i32 n,
    const i32 m,
    const i32 p,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* d,
    const i32 ldd,
    i32* nq,
    i32* nr,
    f64* cr,
    const i32 ldcr,
    f64* dr,
    const i32 lddr,
    const f64 tol,
    f64* dwork,
    const i32 ldwork,
    i32* iwarn,
    i32* info);

/**
 * @brief Right coprime factorization with prescribed stability degree.
 *
 * Constructs, for a given system G = (A,B,C,D), a feedback matrix F and an
 * orthogonal transformation matrix Z, such that the systems
 *
 *     Q = (Z'*(A+B*F)*Z, Z'*B, (C+D*F)*Z, D)
 *     R = (Z'*(A+B*F)*Z, Z'*B, F*Z, I)
 *
 * provide a stable right coprime factorization G = Q * R^(-1).
 *
 * The resulting state dynamics matrix has eigenvalues lying inside a given
 * stability domain. Unstabilizable parts are automatically deflated.
 *
 * @param[in] dico 'C' for continuous-time, 'D' for discrete-time
 * @param[in] n Order of state matrix A (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] p Number of outputs (p >= 0)
 * @param[in] alpha Array of size 2: alpha[0] = desired stability degree,
 *                  alpha[1] = stability margin. For continuous-time, eigenvalues
 *                  outside alpha[1]-region get real parts = alpha[0] < 0.
 *                  For discrete-time, 0 <= alpha[i] < 1.
 * @param[in,out] a On entry: N-by-N state matrix A.
 *                  On exit: NQ-by-NQ numerator state matrix Z'*(A+B*F)*Z in Schur form.
 * @param[in] lda Leading dimension of a. lda >= max(1,n).
 * @param[in,out] b On entry: N-by-M input matrix B.
 *                  On exit: NQ-by-M numerator input matrix Z'*B.
 * @param[in] ldb Leading dimension of b. ldb >= max(1,n).
 * @param[in,out] c On entry: P-by-N output matrix C.
 *                  On exit: P-by-NQ numerator output matrix (C+D*F)*Z.
 * @param[in] ldc Leading dimension of c. ldc >= max(1,p).
 * @param[in] d P-by-M feedthrough matrix D (also numerator feedthrough).
 * @param[in] ldd Leading dimension of d. ldd >= max(1,p).
 * @param[out] nq Order of the resulting factors Q and R.
 * @param[out] nr Order of the minimal realization of factor R.
 * @param[out] cr M-by-NQ feedback matrix F*Z. Last NR columns are denominator output.
 * @param[in] ldcr Leading dimension of cr. ldcr >= max(1,m).
 * @param[out] dr M-by-M identity matrix (denominator feedthrough).
 * @param[in] lddr Leading dimension of dr. lddr >= max(1,m).
 * @param[in] tol Tolerance for controllability test. If <= 0, default is used.
 * @param[out] dwork Workspace array.
 * @param[in] ldwork Length of dwork. ldwork >= max(1, n*(n+5), 5*m, 4*p).
 * @param[out] iwarn Warning: K violations of numerical stability condition.
 * @param[out] info 0=success, <0=parameter error, 1=Schur fail, 2=ordering fail.
 */
void sb08fd(
    const char* dico,
    const i32 n,
    const i32 m,
    const i32 p,
    const f64* alpha,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    const f64* d,
    const i32 ldd,
    i32* nq,
    i32* nr,
    f64* cr,
    const i32 ldcr,
    f64* dr,
    const i32 lddr,
    const f64 tol,
    f64* dwork,
    const i32 ldwork,
    i32* iwarn,
    i32* info);

/**
 * @brief H-infinity optimal controller synthesis.
 *
 * Computes an H-infinity optimal n-state controller
 *
 *           | AK | BK |
 *       K = |----|----|
 *           | CK | DK |
 *
 * using modified Glover's and Doyle's 1988 formulas, for the system
 *
 *           | A  | B1  B2  |   | A | B |
 *       P = |----|---------| = |---|---|
 *           | C1 | D11 D12 |   | C | D |
 *           | C2 | D21 D22 |
 *
 * and the closed-loop system G = (AC, BC, CC, DC).
 *
 * @param[in] job Strategy for gamma reduction:
 *                1 = bisection, 2 = scan, 3 = bisection then scan, 4 = suboptimal only
 * @param[in] n Order of system (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] np Number of outputs (np >= 0)
 * @param[in] ncon Number of control inputs M2 (0 <= ncon <= m, ncon <= np-nmeas)
 * @param[in] nmeas Number of measurements NP2 (0 <= nmeas <= np, nmeas <= m-ncon)
 * @param[in,out] gamma Initial gamma on input; minimal estimated gamma on output
 * @param[in] a N-by-N state matrix A, dimension (lda,n)
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in] b N-by-M input matrix B, dimension (ldb,m)
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in] c NP-by-N output matrix C, dimension (ldc,n)
 * @param[in] ldc Leading dimension of C (ldc >= max(1,np))
 * @param[in] d NP-by-M matrix D, dimension (ldd,m)
 * @param[in] ldd Leading dimension of D (ldd >= max(1,np))
 * @param[out] ak N-by-N controller state matrix, dimension (ldak,n)
 * @param[in] ldak Leading dimension of AK (ldak >= max(1,n))
 * @param[out] bk N-by-NMEAS controller input matrix, dimension (ldbk,nmeas)
 * @param[in] ldbk Leading dimension of BK (ldbk >= max(1,n))
 * @param[out] ck NCON-by-N controller output matrix, dimension (ldck,n)
 * @param[in] ldck Leading dimension of CK (ldck >= max(1,ncon))
 * @param[out] dk NCON-by-NMEAS controller feedthrough, dimension (lddk,nmeas)
 * @param[in] lddk Leading dimension of DK (lddk >= max(1,ncon))
 * @param[out] ac 2N-by-2N closed-loop state matrix, dimension (ldac,2*n)
 * @param[in] ldac Leading dimension of AC (ldac >= max(1,2*n))
 * @param[out] bc 2N-by-(M-NCON) closed-loop input matrix, dimension (ldbc,m-ncon)
 * @param[in] ldbc Leading dimension of BC (ldbc >= max(1,2*n))
 * @param[out] cc (NP-NMEAS)-by-2N closed-loop output matrix, dimension (ldcc,2*n)
 * @param[in] ldcc Leading dimension of CC (ldcc >= max(1,np-nmeas))
 * @param[out] dc (NP-NMEAS)-by-(M-NCON) closed-loop feedthrough, dimension (lddc,m-ncon)
 * @param[in] lddc Leading dimension of DC (lddc >= max(1,np-nmeas))
 * @param[out] rcond Reciprocal condition numbers, dimension (4)
 * @param[in] gtol Gamma tolerance (if <= 0, sqrt(EPS) is used)
 * @param[in] actol Upper bound for closed-loop poles (actol <= 0 for stable)
 * @param iwork Integer workspace, dimension (liwork)
 * @param[in] liwork Size of iwork (>= max(2*max(n,m1,np1,m2,np2), n*n))
 * @param dwork Real workspace, dimension (ldwork)
 * @param[in] ldwork Size of dwork (see routine for formula)
 * @param bwork Logical workspace, dimension (lbwork)
 * @param[in] lbwork Size of bwork (>= 2*n)
 * @param[out] info Exit code:
 *                  0 = success, <0 = invalid parameter -info
 *                  1-5 = rank/SVD errors, 6 = gamma too small
 *                  7-8 = Riccati errors, 9-11 = numerical errors
 *                  12 = no stabilizing controller found
 */
void sb10ad(
    const i32 job,
    const i32 n,
    const i32 m,
    const i32 np,
    const i32 ncon,
    const i32 nmeas,
    f64* gamma,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    const f64* c,
    const i32 ldc,
    const f64* d,
    const i32 ldd,
    f64* ak,
    const i32 ldak,
    f64* bk,
    const i32 ldbk,
    f64* ck,
    const i32 ldck,
    f64* dk,
    const i32 lddk,
    f64* ac,
    const i32 ldac,
    f64* bc,
    const i32 ldbc,
    f64* cc,
    const i32 ldcc,
    f64* dc,
    const i32 lddc,
    f64* rcond,
    const f64 gtol,
    const f64 actol,
    i32* iwork,
    const i32 liwork,
    f64* dwork,
    const i32 ldwork,
    i32* bwork,
    const i32 lbwork,
    i32* info);

/**
 * @brief H-infinity (sub)optimal controller for discrete-time system.
 *
 * Computes the matrices of an H-infinity (sub)optimal n-state controller
 *
 *                     | AK | BK |
 *                 K = |----|----|
 *                     | CK | DK |
 *
 * for the discrete-time system
 *
 *             | A  | B1  B2  |   | A | B |
 *         P = |----|---------| = |---|---|
 *             | C1 | D11 D12 |   | C | D |
 *             | C2 | D21 D22 |
 *
 * where B2 has NCON columns (control inputs) and C2 has NMEAS rows (measurements).
 *
 * @param[in] n Order of the system (n >= 0)
 * @param[in] m Column size of B (m >= 0)
 * @param[in] np Row size of C (np >= 0)
 * @param[in] ncon Number of control inputs (0 <= ncon <= m, np-nmeas >= ncon)
 * @param[in] nmeas Number of measurements (0 <= nmeas <= np, m-ncon >= nmeas)
 * @param[in] gamma H-infinity norm bound (gamma > 0)
 * @param[in] a N-by-N state matrix A
 * @param[in] lda Leading dimension of A (>= max(1,n))
 * @param[in] b N-by-M input matrix B
 * @param[in] ldb Leading dimension of B (>= max(1,n))
 * @param[in] c NP-by-N output matrix C
 * @param[in] ldc Leading dimension of C (>= max(1,np))
 * @param[in] d NP-by-M feedthrough matrix D
 * @param[in] ldd Leading dimension of D (>= max(1,np))
 * @param[out] ak N-by-N controller state matrix
 * @param[in] ldak Leading dimension of AK (>= max(1,n))
 * @param[out] bk N-by-NMEAS controller input matrix
 * @param[in] ldbk Leading dimension of BK (>= max(1,n))
 * @param[out] ck NCON-by-N controller output matrix
 * @param[in] ldck Leading dimension of CK (>= max(1,ncon))
 * @param[out] dk NCON-by-NMEAS controller feedthrough matrix
 * @param[in] lddk Leading dimension of DK (>= max(1,ncon))
 * @param[out] x N-by-N solution of X-Riccati equation
 * @param[in] ldx Leading dimension of X (>= max(1,n))
 * @param[out] z N-by-N solution of Z-Riccati equation
 * @param[in] ldz Leading dimension of Z (>= max(1,n))
 * @param[out] rcond Array(8) of reciprocal condition numbers
 * @param[in] tol Tolerance for rank determination (if <= 0, uses 1000*eps)
 * @param[out] iwork Integer workspace
 * @param[out] dwork Double workspace
 * @param[in] ldwork Workspace size
 * @param[out] info 0=success, 1-4=rank deficiency, 5=inadmissible, 6-7=Riccati fail, 8=singular, 9=SVD fail
 */
void sb10dd(
    const i32 n,
    const i32 m,
    const i32 np,
    const i32 ncon,
    const i32 nmeas,
    const f64 gamma,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    const f64* c,
    const i32 ldc,
    const f64* d,
    const i32 ldd,
    f64* ak,
    const i32 ldak,
    f64* bk,
    const i32 ldbk,
    f64* ck,
    const i32 ldck,
    f64* dk,
    const i32 lddk,
    f64* x,
    const i32 ldx,
    f64* z,
    const i32 ldz,
    f64* rcond,
    const f64 tol,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info);

/**
 * @brief H2 optimal n-state controller for discrete-time systems.
 *
 * Computes the H2 optimal n-state controller:
 *           | AK | BK |
 *       K = |----|----|
 *           | CK | DK |
 *
 * for the discrete-time system:
 *           | A  | B1  B2  |   | A | B |
 *       P = |----|---------| = |---|---|
 *           | C1 |  0  D12 |   | C | D |
 *           | C2 | D21 D22 |
 *
 * where B2 has NCON columns (control inputs) and C2 has NMEAS rows
 * (measurements) provided to the controller.
 *
 * Assumptions:
 * (A1) (A,B2) is stabilizable and (C2,A) is detectable
 * (A2) D12 is full column rank and D21 is full row rank
 * (A3,A4) Discrete-time invariant zeros conditions
 *
 * @param[in] n Order of system (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] np Number of outputs (np >= 0)
 * @param[in] ncon Number of control inputs M2 (0 <= ncon <= m, ncon <= np-nmeas)
 * @param[in] nmeas Number of measurements NP2 (0 <= nmeas <= np, nmeas <= m-ncon)
 * @param[in,out] a N-by-N state matrix A (modified internally, restored on exit)
 * @param[in] lda Leading dimension of A (>= max(1,n))
 * @param[in] b N-by-M input matrix B
 * @param[in] ldb Leading dimension of B (>= max(1,n))
 * @param[in] c NP-by-N output matrix C
 * @param[in] ldc Leading dimension of C (>= max(1,np))
 * @param[in] d NP-by-M feedthrough matrix D
 * @param[in] ldd Leading dimension of D (>= max(1,np))
 * @param[out] ak N-by-N controller state matrix
 * @param[in] ldak Leading dimension of AK (>= max(1,n))
 * @param[out] bk N-by-NMEAS controller input matrix
 * @param[in] ldbk Leading dimension of BK (>= max(1,n))
 * @param[out] ck NCON-by-N controller output matrix
 * @param[in] ldck Leading dimension of CK (>= max(1,ncon))
 * @param[out] dk NCON-by-NMEAS controller feedthrough
 * @param[in] lddk Leading dimension of DK (>= max(1,ncon))
 * @param[out] rcond Reciprocal condition numbers, dimension (7):
 *                   rcond[0] = control transformation matrix TU
 *                   rcond[1] = measurement transformation matrix TY
 *                   rcond[2] = Im2 + B2'*X2*B2
 *                   rcond[3] = Ip2 + C2*Y2*C2'
 *                   rcond[4] = X-Riccati equation
 *                   rcond[5] = Y-Riccati equation
 *                   rcond[6] = Im2 + DKHAT*D22
 * @param[in] tol Tolerance for transformations. If <= 0, sqrt(EPS) is used.
 * @param[out] iwork Integer workspace, dimension max(2*ncon,2*n,n*n,nmeas)
 * @param[out] dwork Double workspace, dimension (ldwork). On exit, dwork[0] = optimal.
 * @param[in] ldwork Size of dwork (see routine documentation)
 * @param[out] bwork Logical workspace, dimension (2*n)
 * @param[out] info Exit code:
 *                  0 = success, <0 = invalid parameter -info,
 *                  1-2 = rank deficiency at discrete-time invariant zeros,
 *                  3-4 = D12/D21 not full rank,
 *                  5 = SVD did not converge,
 *                  6 = X-Riccati equation failed,
 *                  7 = Im2 + B2'*X2*B2 not positive definite,
 *                  8 = Y-Riccati equation failed,
 *                  9 = Ip2 + C2*Y2*C2' not positive definite,
 *                  10 = Im2 + DKHAT*D22 singular
 */
void sb10ed(
    const i32 n,
    const i32 m,
    const i32 np,
    const i32 ncon,
    const i32 nmeas,
    f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    const f64* c,
    const i32 ldc,
    const f64* d,
    const i32 ldd,
    f64* ak,
    const i32 ldak,
    f64* bk,
    const i32 ldbk,
    f64* ck,
    const i32 ldck,
    f64* dk,
    const i32 lddk,
    f64* rcond,
    const f64 tol,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* bwork,
    i32* info);

/**
 * @brief H-infinity (sub)optimal controller for fixed gamma.
 *
 * Computes an H-infinity (sub)optimal n-state controller
 *
 *           | AK | BK |
 *       K = |----|----|
 *           | CK | DK |
 *
 * using modified Glover's and Doyle's 1988 formulas, for the system
 *
 *           | A  | B1  B2  |   | A | B |
 *       P = |----|---------| = |---|---|
 *           | C1 | D11 D12 |   | C | D |
 *           | C2 | D21 D22 |
 *
 * and for a given value of gamma. Simpler interface than SB10AD when
 * the gamma value is known.
 *
 * Assumptions:
 * - (A,B2) is stabilizable and (C2,A) is detectable
 * - D12 is full column rank and D21 is full row rank
 * - Rank conditions for H-infinity solvability are satisfied
 *
 * @param[in] n Order of system (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] np Number of outputs (np >= 0)
 * @param[in] ncon Number of control inputs M2 (0 <= ncon <= m, ncon <= np-nmeas)
 * @param[in] nmeas Number of measurements NP2 (0 <= nmeas <= np, nmeas <= m-ncon)
 * @param[in] gamma The H-infinity norm bound (gamma >= 0)
 * @param[in] a N-by-N state matrix A, dimension (lda,n)
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in] b N-by-M input matrix B, dimension (ldb,m)
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in] c NP-by-N output matrix C, dimension (ldc,n)
 * @param[in] ldc Leading dimension of C (ldc >= max(1,np))
 * @param[in] d NP-by-M matrix D, dimension (ldd,m)
 * @param[in] ldd Leading dimension of D (ldd >= max(1,np))
 * @param[out] ak N-by-N controller state matrix, dimension (ldak,n)
 * @param[in] ldak Leading dimension of AK (ldak >= max(1,n))
 * @param[out] bk N-by-NMEAS controller input matrix, dimension (ldbk,nmeas)
 * @param[in] ldbk Leading dimension of BK (ldbk >= max(1,n))
 * @param[out] ck NCON-by-N controller output matrix, dimension (ldck,n)
 * @param[in] ldck Leading dimension of CK (ldck >= max(1,ncon))
 * @param[out] dk NCON-by-NMEAS controller feedthrough, dimension (lddk,nmeas)
 * @param[in] lddk Leading dimension of DK (lddk >= max(1,ncon))
 * @param[out] rcond Reciprocal condition numbers, dimension (4):
 *                   rcond[0] = rcond of control transformation
 *                   rcond[1] = rcond of measurement transformation
 *                   rcond[2] = rcond of X-Riccati equation
 *                   rcond[3] = rcond of Y-Riccati equation
 * @param[in] tol Tolerance for transformations. If <= 0, sqrt(EPS) is used.
 * @param[out] iwork Integer workspace, dimension max(2*max(n,m-ncon,np-nmeas,ncon), n*n)
 * @param[out] dwork Real workspace, dimension (ldwork). On exit, dwork[0] = optimal ldwork.
 * @param[in] ldwork Size of dwork (see routine for formula)
 * @param[out] bwork Logical workspace, dimension (2*n)
 * @param[out] info Exit code:
 *                  0 = success
 *                  <0 = invalid parameter -info
 *                  1-2 = rank deficiency detected
 *                  3-4 = D12/D21 not full rank
 *                  5 = SVD did not converge
 *                  6 = controller not admissible (gamma too small)
 *                  7 = X-Riccati equation failed
 *                  8 = Y-Riccati equation failed
 *                  9 = determinant of Im2 + Tu*D11HAT*Ty*D22 is zero
 */
void sb10fd(
    const i32 n,
    const i32 m,
    const i32 np,
    const i32 ncon,
    const i32 nmeas,
    const f64 gamma,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    const f64* c,
    const i32 ldc,
    const f64* d,
    const i32 ldd,
    f64* ak,
    const i32 ldak,
    f64* bk,
    const i32 ldbk,
    f64* ck,
    const i32 ldck,
    f64* dk,
    const i32 lddk,
    f64* rcond,
    const f64 tol,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* bwork,
    i32* info);

/**
 * @brief Convert descriptor state-space to regular state-space form.
 *
 * Converts the descriptor state-space system:
 *     E*dx/dt = A*x + B*u
 *          y = C*x + D*u
 *
 * into regular state-space form:
 *     dx/dt = Ad*x + Bd*u
 *         y = Cd*x + Dd*u
 *
 * Uses SVD decomposition of E for descriptor elimination. The order of the
 * resulting regular system (NSYS) may be less than N if E is rank deficient.
 *
 * Reference:
 * Chiang, R.Y. and Safonov, M.G.
 * Robust Control Toolbox User's Guide.
 * The MathWorks Inc., Natick, Mass., 1992.
 *
 * @param[in] n Order of descriptor system (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] np Number of outputs (np >= 0)
 * @param[in,out] a N-by-N state matrix A, dimension (lda,n)
 *                  On exit: NSYS-by-NSYS state matrix Ad
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] b N-by-M input matrix B, dimension (ldb,m)
 *                  On exit: NSYS-by-M input matrix Bd
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in,out] c NP-by-N output matrix C, dimension (ldc,n)
 *                  On exit: NP-by-NSYS output matrix Cd
 * @param[in] ldc Leading dimension of C (ldc >= max(1,np))
 * @param[in,out] d NP-by-M feedthrough matrix D, dimension (ldd,m)
 *                  On exit: NP-by-M feedthrough matrix Dd
 * @param[in] ldd Leading dimension of D (ldd >= max(1,np))
 * @param[in,out] e N-by-N descriptor matrix E, dimension (lde,n)
 *                  On exit: destroyed (no useful info)
 * @param[in] lde Leading dimension of E (lde >= max(1,n))
 * @param[out] nsys Order of converted regular state-space system
 * @param[out] dwork Double workspace, dimension (ldwork)
 *                   On exit: dwork[0] = optimal ldwork
 * @param[in] ldwork Workspace size
 *                   ldwork >= max(1, 2*n*n + 2*n + n*max(5, n+m+np))
 * @param[out] info Exit code:
 *                  0 = success
 *                  <0 = invalid parameter -info
 *                  1 = SVD iteration did not converge
 */

/**
 * @brief Compute H2 optimal n-state controller for continuous-time system.
 *
 * Computes the matrices of the H2 optimal n-state controller:
 *
 *         | AK | BK |
 *     K = |----|----|
 *         | CK | DK |
 *
 * for the system:
 *
 *             | A  | B1  B2  |   | A | B |
 *         P = |----|---------| = |---|---|
 *             | C1 |  0  D12 |   | C | D |
 *             | C2 | D21 D22 |
 *
 * where B2 has column size NCON (control inputs) and C2 has row size NMEAS (measurements).
 *
 * Assumptions:
 * - (A,B2) is stabilizable and (C2,A) is detectable
 * - D11 = 0
 * - D12 is full column rank and D21 is full row rank
 *
 * Reference:
 * Zhou, K., Doyle, J.C., and Glover, K.
 * Robust and Optimal Control. Prentice-Hall, 1996.
 *
 * @param[in] n System order (n >= 0)
 * @param[in] m Column size of B (m >= 0)
 * @param[in] np Row size of C (np >= 0)
 * @param[in] ncon Number of control inputs M2 (0 <= ncon <= m, ncon <= np-nmeas)
 * @param[in] nmeas Number of measurements NP2 (0 <= nmeas <= np, nmeas <= m-ncon)
 * @param[in] a System state matrix A, dimension (lda, n)
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in] b System input matrix B, dimension (ldb, m)
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in] c System output matrix C, dimension (ldc, n)
 * @param[in] ldc Leading dimension of C (ldc >= max(1,np))
 * @param[in] d System matrix D, dimension (ldd, m)
 * @param[in] ldd Leading dimension of D (ldd >= max(1,np))
 * @param[out] ak Controller state matrix AK, dimension (ldak, n)
 * @param[in] ldak Leading dimension of AK (ldak >= max(1,n))
 * @param[out] bk Controller input matrix BK, dimension (ldbk, nmeas)
 * @param[in] ldbk Leading dimension of BK (ldbk >= max(1,n))
 * @param[out] ck Controller output matrix CK, dimension (ldck, n)
 * @param[in] ldck Leading dimension of CK (ldck >= max(1,ncon))
 * @param[out] dk Controller matrix DK, dimension (lddk, nmeas)
 * @param[in] lddk Leading dimension of DK (lddk >= max(1,ncon))
 * @param[out] rcond Condition estimates, dimension (4):
 *                   rcond[0] = control transformation reciprocal condition
 *                   rcond[1] = measurement transformation reciprocal condition
 *                   rcond[2] = X-Riccati reciprocal condition estimate
 *                   rcond[3] = Y-Riccati reciprocal condition estimate
 * @param[in] tol Tolerance for transformation accuracy. If tol <= 0, sqrt(eps) is used.
 * @param[out] iwork Integer workspace, dimension (max(2*n, n*n))
 * @param[out] dwork Double workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size (see documentation for formula)
 * @param[out] bwork Logical workspace, dimension (2*n)
 * @param[out] info Exit code:
 *                  0 = success
 *                  <0 = invalid parameter -info
 *                  1 = D12 not full column rank
 *                  2 = D21 not full row rank
 *                  3 = SVD did not converge
 *                  4 = X-Riccati equation not solved
 *                  5 = Y-Riccati equation not solved
 */
void sb10hd(
    const i32 n,
    const i32 m,
    const i32 np,
    const i32 ncon,
    const i32 nmeas,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    const f64* c,
    const i32 ldc,
    const f64* d,
    const i32 ldd,
    f64* ak,
    const i32 ldak,
    f64* bk,
    const i32 ldbk,
    f64* ck,
    const i32 ldck,
    f64* dk,
    const i32 lddk,
    f64* rcond,
    const f64 tol,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* bwork,
    i32* info);

/**
 * @brief Positive feedback controller for loop shaping design.
 *
 * Computes the matrices of the positive feedback controller
 * K = [Ak, Bk; Ck, Dk] for the shaped plant G = [A, B; C, D]
 * in the McFarlane/Glover Loop Shaping Design Procedure.
 *
 * @param[in] n Order of the plant (N >= 0)
 * @param[in] m Number of plant inputs (M >= 0)
 * @param[in] np Number of plant outputs (NP >= 0)
 * @param[in] a State matrix A (N x N)
 * @param[in] lda Leading dimension of A
 * @param[in] b Input matrix B (N x M)
 * @param[in] ldb Leading dimension of B
 * @param[in] c Output matrix C (NP x N)
 * @param[in] ldc Leading dimension of C
 * @param[in] d Feedthrough matrix D (NP x M)
 * @param[in] ldd Leading dimension of D
 * @param[in] factor =1: optimal controller; >1: suboptimal controller
 * @param[out] nk Controller order (NK <= N)
 * @param[out] ak Controller state matrix (NK x NK)
 * @param[in] ldak Leading dimension of AK
 * @param[out] bk Controller input matrix (NK x NP)
 * @param[in] ldbk Leading dimension of BK
 * @param[out] ck Controller output matrix (M x NK)
 * @param[in] ldck Leading dimension of CK
 * @param[out] dk Controller feedthrough matrix (M x NP)
 * @param[in] lddk Leading dimension of DK
 * @param[out] rcond Reciprocal condition numbers: [X-Riccati, Z-Riccati]
 * @param[out] iwork Integer workspace (max(2*N, N*N, M, NP))
 * @param[out] dwork Double workspace
 * @param[in] ldwork Workspace size
 * @param[out] bwork Boolean workspace (2*N)
 * @param[out] info 0: success
 *                  1: X-Riccati equation failed
 *                  2: Z-Riccati equation failed
 *                  3: eigenvalue/singular value iteration failed
 *                  4: Ip - D*Dk is singular
 *                  5: Im - Dk*D is singular
 *                  6: closed-loop system is unstable
 */
void sb10id(
    const i32 n,
    const i32 m,
    const i32 np,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    const f64* c,
    const i32 ldc,
    const f64* d,
    const i32 ldd,
    const f64 factor,
    i32* nk,
    f64* ak,
    const i32 ldak,
    f64* bk,
    const i32 ldbk,
    f64* ck,
    const i32 ldck,
    f64* dk,
    const i32 lddk,
    f64* rcond,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* bwork,
    i32* info);

/**
 * @brief Convert descriptor system to regular state-space form.
 *
 * Converts E*dx/dt = A*x + B*u, y = C*x + D*u to regular state-space
 * dx/dt = Ad*x + Bd*u, y = Cd*x + Dd*u by eliminating non-dynamic modes.
 *
 * @param[in] n Order of descriptor system (N >= 0)
 * @param[in] m Number of inputs (M >= 0)
 * @param[in] np Number of outputs (NP >= 0)
 * @param[in,out] a State matrix (N x N). On exit, regular form (NSYS x NSYS).
 * @param[in] lda Leading dimension of A
 * @param[in,out] b Input matrix (N x M). On exit, regular form (NSYS x M).
 * @param[in] ldb Leading dimension of B
 * @param[in,out] c Output matrix (NP x N). On exit, regular form (NP x NSYS).
 * @param[in] ldc Leading dimension of C
 * @param[in,out] d Feedthrough matrix (NP x M). On exit, updated.
 * @param[in] ldd Leading dimension of D
 * @param[in,out] e Descriptor matrix (N x N). Modified on exit.
 * @param[in] lde Leading dimension of E
 * @param[out] nsys Order of resulting regular system
 * @param[out] dwork Workspace
 * @param[in] ldwork Workspace size
 * @param[out] info 0=success, <0=arg error, 1=singular pencil
 */
void sb10jd(
    const i32 n,
    const i32 m,
    const i32 np,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* d,
    const i32 ldd,
    f64* e,
    const i32 lde,
    i32* nsys,
    f64* dwork,
    const i32 ldwork,
    i32* info);

/**
 * @brief Compute closed-loop system matrices.
 *
 * Computes the matrices of the closed-loop system G = (AC, BC, CC, DC) from
 * the open-loop plant P = (A, B, C, D) and controller K = (AK, BK, CK, DK).
 *
 * The plant matrix partition is:
 *   P = [A   B1  B2 ]    where B1: N-by-M1, B2: N-by-M2, M1=M-NCON, M2=NCON
 *       [C1  D11 D12]          C1: NP1-by-N, C2: NP2-by-N, NP1=NP-NMEAS, NP2=NMEAS
 *       [C2  D21 D22]
 *
 * The controller K has dimensions N-by-N (AK), N-by-NP2 (BK), M2-by-N (CK),
 * M2-by-NP2 (DK).
 *
 * @param[in] n System order (n >= 0)
 * @param[in] m Total number of inputs (m >= 0)
 * @param[in] np Total number of outputs (np >= 0)
 * @param[in] ncon Number of control inputs M2 (0 <= ncon <= m, ncon <= np-nmeas)
 * @param[in] nmeas Number of measurements NP2 (0 <= nmeas <= np, nmeas <= m-ncon)
 * @param[in] a Plant state matrix A, dimension (lda, n)
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in] b Plant input matrix B, dimension (ldb, m)
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in] c Plant output matrix C, dimension (ldc, n)
 * @param[in] ldc Leading dimension of C (ldc >= max(1,np))
 * @param[in] d Plant feedthrough matrix D, dimension (ldd, m)
 * @param[in] ldd Leading dimension of D (ldd >= max(1,np))
 * @param[in] ak Controller state matrix AK, dimension (ldak, n)
 * @param[in] ldak Leading dimension of AK (ldak >= max(1,n))
 * @param[in] bk Controller input matrix BK, dimension (ldbk, nmeas)
 * @param[in] ldbk Leading dimension of BK (ldbk >= max(1,n))
 * @param[in] ck Controller output matrix CK, dimension (ldck, n)
 * @param[in] ldck Leading dimension of CK (ldck >= max(1,ncon))
 * @param[in] dk Controller feedthrough matrix DK, dimension (lddk, nmeas)
 * @param[in] lddk Leading dimension of DK (lddk >= max(1,ncon))
 * @param[out] ac Closed-loop state matrix AC, dimension (ldac, 2*n)
 * @param[in] ldac Leading dimension of AC (ldac >= max(1,2*n))
 * @param[out] bc Closed-loop input matrix BC, dimension (ldbc, m-ncon)
 * @param[in] ldbc Leading dimension of BC (ldbc >= max(1,2*n))
 * @param[out] cc Closed-loop output matrix CC, dimension (ldcc, 2*n)
 * @param[in] ldcc Leading dimension of CC (ldcc >= max(1,np-nmeas))
 * @param[out] dc Closed-loop feedthrough DC, dimension (lddc, m-ncon)
 * @param[in] lddc Leading dimension of DC (lddc >= max(1,np-nmeas))
 * @param[out] iwork Integer workspace, dimension (2*max(ncon,nmeas))
 * @param[out] dwork Double workspace, dimension (ldwork)
 *                   On exit: dwork[0] = optimal ldwork
 * @param[in] ldwork Workspace size (ldwork >= 2*m*m + np*np + 2*m*n + m*np + 2*n*np)
 * @param[out] info Exit code:
 *                  0 = success
 *                  <0 = invalid parameter -info
 *                  1 = I - D22*DK is singular
 *                  2 = I - DK*D22 is singular
 */
void sb10ld(
    const i32 n,
    const i32 m,
    const i32 np,
    const i32 ncon,
    const i32 nmeas,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    const f64* c,
    const i32 ldc,
    const f64* d,
    const i32 ldd,
    const f64* ak,
    const i32 ldak,
    const f64* bk,
    const i32 ldbk,
    const f64* ck,
    const i32 ldck,
    const f64* dk,
    const i32 lddk,
    f64* ac,
    const i32 ldac,
    f64* bc,
    const i32 ldbc,
    f64* cc,
    const i32 ldcc,
    f64* dc,
    const i32 lddc,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info);

/**
 * @brief Normalize system for H-infinity controller design.
 *
 * Reduces D12 and D21 matrices of the partitioned system:
 *
 *           | A  | B1  B2  |   | A | B |
 *       P = |----|---------| = |---|---|
 *           | C1 | D11 D12 |   | C | D |
 *           | C2 | D21 D22 |
 *
 * to unit diagonal form and transforms B, C, D11 for H2/H-infinity
 * controller computation. Checks rank conditions using SVD.
 *
 * Partitioning: M1 = M - NCON, M2 = NCON, NP1 = NP - NMEAS, NP2 = NMEAS
 *
 * @param[in] n Order of system (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] np Number of outputs (np >= 0)
 * @param[in] ncon Number of control inputs M2 (0 <= ncon <= m, ncon <= np-nmeas)
 * @param[in] nmeas Number of measurements NP2 (0 <= nmeas <= np, nmeas <= m-ncon)
 * @param[in] a N-by-N state matrix A, dimension (lda,n)
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] b N-by-M input matrix B, dimension (ldb,m)
 *                  On exit: transformed B
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in,out] c NP-by-N output matrix C, dimension (ldc,n)
 *                  On exit: transformed C
 * @param[in] ldc Leading dimension of C (ldc >= max(1,np))
 * @param[in,out] d NP-by-M matrix D, dimension (ldd,m)
 *                  On exit: (NP1-by-M1) contains transformed D11
 * @param[in] ldd Leading dimension of D (ldd >= max(1,np))
 * @param[out] tu M2-by-M2 control transformation matrix, dimension (ldtu,m2)
 * @param[in] ldtu Leading dimension of TU (ldtu >= max(1,m2))
 * @param[out] ty NP2-by-NP2 measurement transformation matrix, dimension (ldty,np2)
 * @param[in] ldty Leading dimension of TY (ldty >= max(1,np2))
 * @param[out] rcond Reciprocal condition numbers, dimension (2)
 *                   rcond[0] = rcond of TU
 *                   rcond[1] = rcond of TY
 * @param[in] tol Tolerance for condition tests. If <= 0, sqrt(EPS) is used.
 * @param[out] dwork Workspace, dimension (ldwork)
 *                   On exit: dwork[0] = optimal ldwork
 * @param[in] ldwork Workspace size (see routine for formula)
 * @param[out] info Exit code:
 *                  0 = success
 *                  <0 = invalid parameter -info
 *                  1 = |A B2; C1 D12| not full column rank
 *                  2 = |A B1; C2 D21| not full row rank
 *                  3 = D12 not full column rank (rcond < tol)
 *                  4 = D21 not full row rank (rcond < tol)
 *                  5 = SVD did not converge
 */
void sb10pd(
    const i32 n,
    const i32 m,
    const i32 np,
    const i32 ncon,
    const i32 nmeas,
    const f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* d,
    const i32 ldd,
    f64* tu,
    const i32 ldtu,
    f64* ty,
    const i32 ldty,
    f64* rcond,
    const f64 tol,
    f64* dwork,
    const i32 ldwork,
    i32* info);

/**
 * @brief State feedback and output injection matrices for H-infinity controller.
 *
 * Computes the state feedback matrix F and output injection matrix H
 * for an H-infinity (sub)optimal n-state controller using Glover's and
 * Doyle's 1988 formulas. Solves two Riccati equations (X and Y) with
 * condition and accuracy estimates.
 *
 * System partitioning:
 *           | A  | B1  B2  |   | A | B |
 *       P = |----|---------| = |---|---|
 *           | C1 | D11 D12 |   | C | D |
 *           | C2 | D21 D22 |
 *
 * Assumptions (as obtained from SB10PD):
 * - (A,B2) is stabilizable and (C2,A) is detectable
 * - D12 = [0; I] (full column rank) and D21 = [0 I] (full row rank)
 * - Full rank conditions on (A,B2,C1,D12) and (A,B1,C2,D21)
 *
 * @param[in] n Order of system (n >= 0)
 * @param[in] m Column size of B (m >= 0)
 * @param[in] np Row size of C (np >= 0)
 * @param[in] ncon Number of control inputs M2 (m >= ncon >= 0, np-nmeas >= ncon)
 * @param[in] nmeas Number of measurements NP2 (np >= nmeas >= 0, m-ncon >= nmeas)
 * @param[in] gamma H-infinity norm bound (gamma >= 0)
 * @param[in] a N-by-N state matrix A, dimension (lda,n)
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in] b N-by-M input matrix B, dimension (ldb,m)
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in] c NP-by-N output matrix C, dimension (ldc,n)
 * @param[in] ldc Leading dimension of C (ldc >= max(1,np))
 * @param[in] d NP-by-M feedthrough matrix D, dimension (ldd,m)
 * @param[in] ldd Leading dimension of D (ldd >= max(1,np))
 * @param[out] f M-by-N state feedback matrix F, dimension (ldf,n)
 * @param[in] ldf Leading dimension of F (ldf >= max(1,m))
 * @param[out] h N-by-NP output injection matrix H, dimension (ldh,np)
 * @param[in] ldh Leading dimension of H (ldh >= max(1,n))
 * @param[out] x N-by-N X-Riccati solution, dimension (ldx,n)
 * @param[in] ldx Leading dimension of X (ldx >= max(1,n))
 * @param[out] y N-by-N Y-Riccati solution, dimension (ldy,n)
 * @param[in] ldy Leading dimension of Y (ldy >= max(1,n))
 * @param[out] xycond Array of dimension 2:
 *                    xycond[0] = reciprocal condition number of X-Riccati eq
 *                    xycond[1] = reciprocal condition number of Y-Riccati eq
 * @param[out] iwork Integer workspace, dimension max(2*max(n,m-ncon,np-nmeas),n*n)
 * @param[out] dwork Double workspace, dimension ldwork
 *                   On exit: dwork[0] = optimal ldwork
 * @param[in] ldwork Workspace size.
 *                   Required: max(1, M*M + max(2*M1, 3*N*N + max(N*M, 10*N*N+12*N+5)),
 *                                    NP*NP + max(2*NP1, 3*N*N + max(N*NP, 10*N*N+12*N+5)))
 *                   where M1 = M-NCON, NP1 = NP-NMEAS
 * @param[out] bwork Logical workspace, dimension 2*n
 * @param[out] info Exit code:
 *                  0 = success
 *                  <0 = invalid parameter -info
 *                  1 = controller not admissible (gamma too small)
 *                  2 = X-Riccati equation failed
 *                  3 = Y-Riccati equation failed
 */
void sb10qd(
    const i32 n,
    const i32 m,
    const i32 np,
    const i32 ncon,
    const i32 nmeas,
    const f64 gamma,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    const f64* c,
    const i32 ldc,
    const f64* d,
    const i32 ldd,
    f64* f,
    const i32 ldf,
    f64* h,
    const i32 ldh,
    f64* x,
    const i32 ldx,
    f64* y,
    const i32 ldy,
    f64* xycond,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* bwork,
    i32* info);

/**
 * @brief H-infinity controller from state feedback and output injection matrices.
 *
 * Computes the matrices of an H-infinity (sub)optimal controller:
 *
 *          | AK | BK |
 *      K = |----|----|
 *          | CK | DK |
 *
 * from the state feedback matrix F and output injection matrix H as
 * determined by SB10QD. Implements Glover-Doyle formulas.
 *
 * @param[in] n Order of system (n >= 0)
 * @param[in] m Column size of B (m >= 0)
 * @param[in] np Row size of C (np >= 0)
 * @param[in] ncon Number of control inputs M2 (m >= ncon >= 0, np-nmeas >= ncon)
 * @param[in] nmeas Number of measurements NP2 (np >= nmeas >= 0, m-ncon >= nmeas)
 * @param[in] gamma H-infinity norm bound (gamma >= 0)
 * @param[in] a N-by-N state matrix A, dimension (lda,n)
 * @param[in] lda Leading dimension of A
 * @param[in] b N-by-M input matrix B, dimension (ldb,m)
 * @param[in] ldb Leading dimension of B
 * @param[in] c NP-by-N output matrix C, dimension (ldc,n)
 * @param[in] ldc Leading dimension of C
 * @param[in] d NP-by-M feedthrough matrix D, dimension (ldd,m)
 * @param[in] ldd Leading dimension of D
 * @param[in] f M-by-N state feedback matrix F, dimension (ldf,n)
 * @param[in] ldf Leading dimension of F
 * @param[in] h N-by-NP output injection matrix H, dimension (ldh,np)
 * @param[in] ldh Leading dimension of H
 * @param[in] tu M2-by-M2 control transformation TU, dimension (ldtu,m2)
 * @param[in] ldtu Leading dimension of TU
 * @param[in] ty NP2-by-NP2 measurement transformation TY, dimension (ldty,np2)
 * @param[in] ldty Leading dimension of TY
 * @param[in] x N-by-N X-Riccati solution, dimension (ldx,n)
 * @param[in] ldx Leading dimension of X
 * @param[in] y N-by-N Y-Riccati solution, dimension (ldy,n)
 * @param[in] ldy Leading dimension of Y
 * @param[out] ak N-by-N controller state matrix AK, dimension (ldak,n)
 * @param[in] ldak Leading dimension of AK
 * @param[out] bk N-by-NMEAS controller input matrix BK, dimension (ldbk,nmeas)
 * @param[in] ldbk Leading dimension of BK
 * @param[out] ck NCON-by-N controller output matrix CK, dimension (ldck,n)
 * @param[in] ldck Leading dimension of CK
 * @param[out] dk NCON-by-NMEAS controller feedthrough DK, dimension (lddk,nmeas)
 * @param[in] lddk Leading dimension of DK
 * @param[out] iwork Integer workspace
 * @param[out] dwork Double workspace, dwork[0] = optimal ldwork on exit
 * @param[in] ldwork Workspace size
 * @param[out] info Exit code:
 *                  0 = success
 *                  <0 = invalid parameter -info
 *                  1 = controller not admissible (gamma too small)
 *                  2 = Im2 + Tu*D11HAT*Ty*D22 is singular
 */
void sb10rd(
    const i32 n,
    const i32 m,
    const i32 np,
    const i32 ncon,
    const i32 nmeas,
    const f64 gamma,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    const f64* c,
    const i32 ldc,
    const f64* d,
    const i32 ldd,
    const f64* f,
    const i32 ldf,
    const f64* h,
    const i32 ldh,
    const f64* tu,
    const i32 ldtu,
    const f64* ty,
    const i32 ldty,
    const f64* x,
    const i32 ldx,
    const f64* y,
    const i32 ldy,
    f64* ak,
    const i32 ldak,
    f64* bk,
    const i32 ldbk,
    f64* ck,
    const i32 ldck,
    f64* dk,
    const i32 lddk,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info);

/**
 * @brief H2 optimal controller for normalized discrete-time systems.
 *
 * Computes the H2 optimal controller matrices:
 *     K = | AK | BK |
 *         |----|----|
 *         | CK | DK |
 *
 * for the normalized discrete-time system:
 *     P = | A  | B1  B2  |
 *         |----|---------|
 *         | C1 | D11 D12 |
 *         | C2 | D21  0  |
 *
 * where B2 has column size NCON (control inputs) and C2 has row size
 * NMEAS (measurements). The system should be normalized using SB10PD.
 *
 * Assumptions:
 * - (A,B2) is stabilizable and (C2,A) is detectable
 * - D12 = [0; I] (full column rank), D21 = [0 I] (full row rank)
 * - Full rank conditions on (A,B2,C1,D12) and (A,B1,C2,D21)
 *
 * Solves two discrete Riccati equations with condition estimates.
 *
 * Reference: Zhou, K., Doyle, J.C., and Glover, K.
 * Robust and Optimal Control. Prentice-Hall, 1996.
 *
 * @param[in] n System order (n >= 0)
 * @param[in] m Column size of B (m >= 0)
 * @param[in] np Row size of C (np >= 0)
 * @param[in] ncon Number of control inputs M2 (0 <= ncon <= m, ncon <= np-nmeas)
 * @param[in] nmeas Number of measurements NP2 (0 <= nmeas <= np, nmeas <= m-ncon)
 * @param[in] a N-by-N state matrix A, dimension (lda, n)
 * @param[in] lda Leading dimension of A (>= max(1,n))
 * @param[in] b N-by-M input matrix B, dimension (ldb, m)
 * @param[in] ldb Leading dimension of B (>= max(1,n))
 * @param[in] c NP-by-N output matrix C, dimension (ldc, n)
 * @param[in] ldc Leading dimension of C (>= max(1,np))
 * @param[in] d NP-by-M feedthrough D (only D11 used), dimension (ldd, m)
 * @param[in] ldd Leading dimension of D (>= max(1,np))
 * @param[out] ak N-by-N controller state matrix, dimension (ldak, n)
 * @param[in] ldak Leading dimension of AK (>= max(1,n))
 * @param[out] bk N-by-NMEAS controller input matrix, dimension (ldbk, nmeas)
 * @param[in] ldbk Leading dimension of BK (>= max(1,n))
 * @param[out] ck NCON-by-N controller output matrix, dimension (ldck, n)
 * @param[in] ldck Leading dimension of CK (>= max(1,ncon))
 * @param[out] dk NCON-by-NMEAS controller feedthrough, dimension (lddk, nmeas)
 * @param[in] lddk Leading dimension of DK (>= max(1,ncon))
 * @param[out] x N-by-N X-Riccati solution, dimension (ldx, n)
 * @param[in] ldx Leading dimension of X (>= max(1,n))
 * @param[out] y N-by-N Y-Riccati solution, dimension (ldy, n)
 * @param[in] ldy Leading dimension of Y (>= max(1,n))
 * @param[out] rcond Array of dimension 4:
 *                   rcond[0] = reciprocal condition of Im2 + B2'*X*B2
 *                   rcond[1] = reciprocal condition of Ip2 + C2*Y*C2'
 *                   rcond[2] = reciprocal condition of X-Riccati
 *                   rcond[3] = reciprocal condition of Y-Riccati
 * @param[in] tol Tolerance for nonsingularity (if <= 0, sqrt(eps) used)
 * @param[out] iwork Integer workspace, dimension (max(m2,2*n,n*n,np2))
 * @param[out] dwork Double workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size:
 *                   >= max(1, 14*n*n+6*n+max(14*n+23,16*n),
 *                             m2*(n+m2+max(3,m1)), np2*(n+np2+3))
 *                   where m1=m-ncon, m2=ncon, np2=nmeas.
 * @param[out] bwork Logical workspace, dimension (2*n)
 * @param[out] info Exit code:
 *                  0 = success
 *                  <0 = invalid parameter -info
 *                  1 = X-Riccati not solved successfully
 *                  2 = Im2 + B2'*X*B2 not positive definite or singular
 *                  3 = Y-Riccati not solved successfully
 *                  4 = Ip2 + C2*Y*C2' not positive definite or singular
 */
void sb10sd(
    const i32 n,
    const i32 m,
    const i32 np,
    const i32 ncon,
    const i32 nmeas,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    const f64* c,
    const i32 ldc,
    const f64* d,
    const i32 ldd,
    f64* ak,
    const i32 ldak,
    f64* bk,
    const i32 ldbk,
    f64* ck,
    const i32 ldck,
    f64* dk,
    const i32 lddk,
    f64* x,
    const i32 ldx,
    f64* y,
    const i32 ldy,
    f64* rcond,
    const f64 tol,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* bwork,
    i32* info);

/**
 * @brief Normalize D12 and D21 matrices for H2 controller design.
 *
 * Reduces the matrices D12 and D21 of a partitioned system:
 *
 *           | A  | B1  B2  |   | A | B |
 *       P = |----|---------| = |---|---|
 *           | C1 |  0  D12 |   | C | D |
 *           | C2 | D21 D22 |
 *
 * to unit diagonal form and transforms B and C to satisfy the formulas
 * for H2 optimal controller computation. Uses SVD decomposition.
 *
 * @param[in] n System order (n >= 0)
 * @param[in] m Column size of B (m >= 0)
 * @param[in] np Row size of C (np >= 0)
 * @param[in] ncon Number of control inputs M2 (0 <= ncon <= m, ncon <= np-nmeas)
 * @param[in] nmeas Number of measurements NP2 (0 <= nmeas <= np, nmeas <= m-ncon)
 * @param[in,out] b Input/output matrix B, dimension (ldb, m).
 *                  On exit, contains transformed B.
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in,out] c Input/output matrix C, dimension (ldc, n).
 *                  On exit, contains transformed C.
 * @param[in] ldc Leading dimension of C (ldc >= max(1,np))
 * @param[in,out] d Input/output matrix D, dimension (ldd, m).
 *                  The (NP-NMEAS)-by-(M-NCON) submatrix D11 is not referenced.
 *                  On exit, the trailing NMEAS-by-NCON part contains transformed D22.
 * @param[in] ldd Leading dimension of D (ldd >= max(1,np))
 * @param[out] tu Control transformation matrix, dimension (ldtu, ncon)
 * @param[in] ldtu Leading dimension of TU (ldtu >= max(1,ncon))
 * @param[out] ty Measurement transformation matrix, dimension (ldty, nmeas)
 * @param[in] ldty Leading dimension of TY (ldty >= max(1,nmeas))
 * @param[out] rcond Array of dimension 2.
 *                   rcond[0] = reciprocal condition number of TU
 *                   rcond[1] = reciprocal condition number of TY
 * @param[in] tol Tolerance for transformation accuracy. If tol <= 0, sqrt(eps) is used.
 * @param[out] dwork Workspace, dimension (ldwork). On exit, dwork[0] = optimal ldwork.
 * @param[in] ldwork Workspace size.
 *                   Required: MAX(M2+NP1*NP1+MAX(NP1*N,3*M2+NP1,5*M2),
 *                                 NP2+M1*M1+MAX(M1*N,3*NP2+M1,5*NP2),
 *                                 N*M2,NP2*N,NP2*M2,1)
 *                   where M1=M-NCON, M2=NCON, NP1=NP-NMEAS, NP2=NMEAS.
 * @param[out] info Exit code:
 *                  0 = success
 *                  <0 = invalid parameter -info
 *                  1 = D12 not full column rank (rcond[0] <= tol)
 *                  2 = D21 not full row rank (rcond[1] <= tol)
 *                  3 = SVD algorithm did not converge
 */
void sb10ud(
    const i32 n,
    const i32 m,
    const i32 np,
    const i32 ncon,
    const i32 nmeas,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* d,
    const i32 ldd,
    f64* tu,
    const i32 ldtu,
    f64* ty,
    const i32 ldty,
    f64* rcond,
    const f64 tol,
    f64* dwork,
    const i32 ldwork,
    i32* info);

/**
 * @brief Compute state feedback and output injection matrices for H2 optimal controller.
 *
 * Computes the state feedback matrix F and output injection matrix H
 * for an H2 optimal n-state controller for the system:
 *
 *             | A  | B1  B2  |   | A | B |
 *         P = |----|---------| = |---|---|
 *             | C1 |  0  D12 |   | C | D |
 *             | C2 | D21 D22 |
 *
 * where B2 has column size NCON (control inputs) and C2 has row size NMEAS (measurements).
 *
 * Assumptions:
 * - (A,B2) is stabilizable and (C2,A) is detectable
 * - D12 = [0; I] (full column rank) and D21 = [0 I] (full row rank)
 *   as obtained by SB10UD
 *
 * Solves two Riccati equations:
 * - X-Riccati: Ax'*X + X*Ax + Cx - X*Dx*X = 0 for state feedback F
 * - Y-Riccati: Ay*Y + Y*Ay' + Cy - Y*Dy*Y = 0 for output injection H
 *
 * Reference:
 * Zhou, K., Doyle, J.C., and Glover, K.
 * Robust and Optimal Control. Prentice-Hall, 1996.
 *
 * @param[in] n System order (n >= 0)
 * @param[in] m Column size of B (m >= 0)
 * @param[in] np Row size of C (np >= 0)
 * @param[in] ncon Number of control inputs M2 (0 <= ncon <= m, ncon <= np-nmeas)
 * @param[in] nmeas Number of measurements NP2 (0 <= nmeas <= np, nmeas <= m-ncon)
 * @param[in] a System state matrix A, dimension (lda, n)
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in] b System input matrix B, dimension (ldb, m)
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in] c System output matrix C, dimension (ldc, n)
 * @param[in] ldc Leading dimension of C (ldc >= max(1,np))
 * @param[out] f State feedback matrix F, dimension (ldf, n)
 * @param[in] ldf Leading dimension of F (ldf >= max(1,ncon))
 * @param[out] h Output injection matrix H, dimension (ldh, nmeas)
 * @param[in] ldh Leading dimension of H (ldh >= max(1,n))
 * @param[out] x Solution of X-Riccati equation, dimension (ldx, n)
 * @param[in] ldx Leading dimension of X (ldx >= max(1,n))
 * @param[out] y Solution of Y-Riccati equation, dimension (ldy, n)
 * @param[in] ldy Leading dimension of Y (ldy >= max(1,n))
 * @param[out] xycond Condition estimates: xycond[0]=X-Riccati, xycond[1]=Y-Riccati
 * @param[out] iwork Integer workspace, dimension (max(2*n, n*n))
 * @param[out] dwork Double workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size (>= 13*n*n + 12*n + 5)
 * @param[out] bwork Logical workspace, dimension (2*n)
 * @param[out] info 0=success, <0=invalid parameter -info, 1=X-Riccati failed, 2=Y-Riccati failed
 */
void sb10vd(
    const i32 n,
    const i32 m,
    const i32 np,
    const i32 ncon,
    const i32 nmeas,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    const f64* c,
    const i32 ldc,
    f64* f,
    const i32 ldf,
    f64* h,
    const i32 ldh,
    f64* x,
    const i32 ldx,
    f64* y,
    const i32 ldy,
    f64* xycond,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* bwork,
    i32* info);

/**
 * @brief Compute H2 optimal controller matrices from state feedback and output injection.
 *
 * Computes the matrices of the H2 optimal controller:
 *
 *         | AK | BK |
 *     K = |----|----|
 *         | CK | DK |
 *
 * from the state feedback matrix F and output injection matrix H as
 * determined by SB10VD.
 *
 * Controller formulas:
 *   AK = A + H*C2 + B2*F + H*D22*F
 *   BK = -H*TY
 *   CK = TU*F
 *   DK = 0
 *
 * where B2 = B(:,M-M2+1:M), C2 = C(NP-NP2+1:NP,:), D22 = D(NP-NP2+1:NP,M-M2+1:M).
 *
 * Reference:
 * Zhou, K., Doyle, J.C., and Glover, K.
 * Robust and Optimal Control.
 * Prentice-Hall, Upper Saddle River, NJ, 1996.
 *
 * @param[in] n System order (n >= 0)
 * @param[in] m Total number of inputs (m >= 0)
 * @param[in] np Total number of outputs (np >= 0)
 * @param[in] ncon Number of control inputs M2 (0 <= ncon <= m, ncon <= np-nmeas)
 * @param[in] nmeas Number of measurements NP2 (0 <= nmeas <= np, nmeas <= m-ncon)
 * @param[in] a System state matrix A, dimension (lda, n)
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in] b System input matrix B, dimension (ldb, m)
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in] c System output matrix C, dimension (ldc, n)
 * @param[in] ldc Leading dimension of C (ldc >= max(1,np))
 * @param[in] d System feedthrough matrix D, dimension (ldd, m)
 * @param[in] ldd Leading dimension of D (ldd >= max(1,np))
 * @param[in] f State feedback matrix F, dimension (ldf, n)
 * @param[in] ldf Leading dimension of F (ldf >= max(1,ncon))
 * @param[in] h Output injection matrix H, dimension (ldh, nmeas)
 * @param[in] ldh Leading dimension of H (ldh >= max(1,n))
 * @param[in] tu Control transformation matrix TU, dimension (ldtu, ncon)
 * @param[in] ldtu Leading dimension of TU (ldtu >= max(1,ncon))
 * @param[in] ty Measurement transformation matrix TY, dimension (ldty, nmeas)
 * @param[in] ldty Leading dimension of TY (ldty >= max(1,nmeas))
 * @param[out] ak Controller state matrix AK, dimension (ldak, n)
 * @param[in] ldak Leading dimension of AK (ldak >= max(1,n))
 * @param[out] bk Controller input matrix BK, dimension (ldbk, nmeas)
 * @param[in] ldbk Leading dimension of BK (ldbk >= max(1,n))
 * @param[out] ck Controller output matrix CK, dimension (ldck, n)
 * @param[in] ldck Leading dimension of CK (ldck >= max(1,ncon))
 * @param[out] dk Controller feedthrough matrix DK, dimension (lddk, nmeas)
 * @param[in] lddk Leading dimension of DK (lddk >= max(1,ncon))
 * @param[out] info Exit code:
 *                  0 = success
 *                  <0 = invalid parameter -info
 */
void sb10wd(
    const i32 n,
    const i32 m,
    const i32 np,
    const i32 ncon,
    const i32 nmeas,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    const f64* c,
    const i32 ldc,
    const f64* d,
    const i32 ldd,
    const f64* f,
    const i32 ldf,
    const f64* h,
    const i32 ldh,
    const f64* tu,
    const i32 ldtu,
    const f64* ty,
    const i32 ldty,
    f64* ak,
    const i32 ldak,
    f64* bk,
    const i32 ldbk,
    f64* ck,
    const i32 ldck,
    f64* dk,
    const i32 lddk,
    i32* info);

/**
 * @brief Fit frequency response data with a stable, minimum phase SISO system.
 *
 * Fits a supplied frequency response data with a stable, minimum phase SISO
 * (single-input single-output) system represented by its matrices A, B, C, D.
 * It handles both discrete- and continuous-time cases.
 *
 * @param[in] discfl Type of system: 0 = continuous-time, 1 = discrete-time
 * @param[in] flag Constraint flag: 0 = no constraints, 1 = stable/min phase
 * @param[in] lendat Length of frequency data vectors (lendat >= 2)
 * @param[in] rfrdat Real part of frequency data, dimension (lendat)
 * @param[in] ifrdat Imaginary part of frequency data, dimension (lendat)
 * @param[in] omega Frequencies, dimension (lendat)
 * @param[in,out] n Desired/obtained order of system. n <= lendat-1.
 * @param[out] a State matrix, dimension (lda, n)
 * @param[in] lda Leading dimension of a (lda >= max(1,n))
 * @param[out] b Input vector, dimension (n)
 * @param[out] c Output vector, dimension (n)
 * @param[out] d Feedthrough scalar, dimension (1)
 * @param[in] tol Tolerance for rank determination. <= 0 uses default.
 * @param[out] iwork Integer workspace, dimension (max(2, 2*n+1))
 * @param[out] dwork Double workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size
 * @param[out] zwork Complex workspace, dimension (lzwork)
 * @param[in] lzwork Complex workspace size
 * @param[out] info Exit code
 */
void sb10yd(i32 discfl, i32 flag, i32 lendat, const f64* rfrdat, const f64* ifrdat,
            const f64* omega, i32* n, f64* a, i32 lda, f64* b, f64* c, f64* d,
            f64 tol, i32* iwork, f64* dwork, i32 ldwork, c128* zwork, i32 lzwork,
            i32* info);

/**
 * @brief Transform SISO system to stable and minimum phase.
 *
 * Transforms a SISO (single-input single-output) system [A,B;C,D] by
 * mirroring its unstable poles and zeros in the boundary of the
 * stability domain, preserving the frequency response but making the
 * system stable and minimum phase.
 *
 * For continuous-time systems, positive real parts of poles/zeros are
 * exchanged with their negatives. Discrete-time systems are first
 * converted to continuous-time using bilinear transformation.
 *
 * @param[in] discfl System type: 0=continuous-time, 1=discrete-time
 * @param[in,out] n On entry: order of original system (n >= 0).
 *                  On exit: order of transformed minimal system
 * @param[in,out] a N-by-N system matrix A, dimension (lda,n).
 *                  On exit: transformed A in upper Hessenberg form
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] b System vector B, dimension (n). On exit: transformed B
 * @param[in,out] c System vector C, dimension (n). On exit: transformed C
 *                  (first N-1 elements are zero for exit N)
 * @param[in,out] d System scalar D (1 element). On exit: transformed D
 * @param[out] iwork Integer workspace, dimension (max(2,n+1))
 * @param[out] dwork Double workspace, dimension (ldwork).
 *                   On exit: dwork[0] = optimal ldwork
 * @param[in] ldwork Workspace size (>= max(n*n+5*n, 6*n+1+min(1,n)))
 * @param[out] info Exit code:
 *                  0 = success
 *                  <0 = invalid parameter -info
 *                  1 = discrete->continuous transformation failed
 *                  2 = system poles cannot be found
 *                  3 = inverse system cannot be found (D close to zero)
 *                  4 = system zeros cannot be found
 *                  5 = state-space representation cannot be found
 *                  6 = continuous->discrete transformation failed
 */
void sb10zp(
    i32 discfl,
    i32* n,
    f64* a,
    i32 lda,
    f64* b,
    f64* c,
    f64* d,
    i32* iwork,
    f64* dwork,
    i32 ldwork,
    i32* info);

/**
 * @brief Single-input state feedback matrix for pole assignment.
 *
 * Determines the one-dimensional state feedback matrix G of the
 * linear time-invariant single-input system dX/dt = A*X + B*U,
 * such that the closed-loop system dX/dt = (A - B*G)*X has desired poles.
 * The system must be reduced to orthogonal canonical form via AB01MD first.
 *
 * @param[in] ncont Controllable order from AB01MD (ncont >= 0)
 * @param[in] n Order of matrix Z (n >= ncont)
 * @param[in,out] a On entry: canonical form of A. On exit: Schur form S of (A-B*G)
 * @param[in] lda Leading dimension of A (lda >= max(1, ncont))
 * @param[in,out] b On entry: canonical form of B. On exit: transformed Z*B
 * @param[in] wr Real parts of desired closed-loop poles
 * @param[in] wi Imaginary parts (complex conjugates consecutive)
 * @param[in,out] z On entry: orthogonal transform from AB01MD.
 *                  On exit: orthogonal Z reducing (A-B*G) to Schur form
 * @param[in] ldz Leading dimension of Z (ldz >= max(1, n))
 * @param[out] g State feedback matrix (ncont elements)
 * @param[out] dwork Workspace of size 3*ncont
 * @param[out] info Exit status: 0=success, <0=parameter -info invalid
 */
void sb01md(
    i32 ncont,
    i32 n,
    f64* a,
    i32 lda,
    f64* b,
    const f64* wr,
    const f64* wi,
    f64* z,
    i32 ldz,
    f64* g,
    f64* dwork,
    i32* info);

/**
 * @brief Select purely imaginary eigenvalues for H-infinity norm.
 *
 * Callback function for DGEES eigenvalue selection. Returns 1 (true) for
 * eigenvalues with |real part| < 100*eps, where eps is machine epsilon.
 *
 * @param[in] reig Pointer to real part of eigenvalue
 * @param[in] ieig Pointer to imaginary part of eigenvalue (unused)
 * @return 1 for purely imaginary, 0 otherwise
 */
int sb02cx(const f64* reig, const f64* ieig);

/**
 * @brief Select stable eigenvalues for continuous-time Riccati.
 *
 * Callback function for DGEES eigenvalue selection. Returns 1 (true) for
 * eigenvalues with real part < 0 (stable in continuous-time).
 *
 * @param[in] reig Pointer to real part of eigenvalue
 * @param[in] ieig Pointer to imaginary part of eigenvalue (unused)
 * @return 1 for stable (reig < 0), 0 otherwise
 */
int sb02mv(const f64* reig, const f64* ieig);

/**
 * @brief Select unstable eigenvalues for continuous-time Riccati.
 *
 * Callback function for DGEES eigenvalue selection. Returns 1 (true) for
 * eigenvalues with real part >= 0 (unstable in continuous-time).
 *
 * @param[in] reig Pointer to real part of eigenvalue
 * @param[in] ieig Pointer to imaginary part of eigenvalue (unused)
 * @return 1 for unstable (reig >= 0), 0 otherwise
 */
int sb02mr(const f64* reig, const f64* ieig);

/**
 * @brief Select unstable eigenvalues for discrete-time Riccati.
 *
 * Callback function for DGEES eigenvalue selection. Returns 1 (true) for
 * eigenvalues with modulus >= 1 (unstable in discrete-time).
 *
 * @param[in] reig Pointer to real part of eigenvalue
 * @param[in] ieig Pointer to imaginary part of eigenvalue
 * @return 1 for unstable (|lambda| >= 1), 0 otherwise
 */
int sb02ms(const f64* reig, const f64* ieig);

/**
 * @brief Select stable eigenvalues for discrete-time Riccati.
 *
 * Callback function for DGEES eigenvalue selection. Returns 1 (true) for
 * eigenvalues with modulus < 1 (stable in discrete-time).
 *
 * @param[in] reig Pointer to real part of eigenvalue
 * @param[in] ieig Pointer to imaginary part of eigenvalue
 * @return 1 for stable (|lambda| < 1), 0 otherwise
 */
int sb02mw(const f64* reig, const f64* ieig);

/**
 * @brief Select unstable generalized eigenvalues for continuous-time Riccati.
 *
 * Callback function for DGGES eigenvalue selection. Returns 1 (true) for
 * generalized eigenvalues with positive real part (unstable in continuous-time).
 *
 * Selection criterion: (ALPHAR < 0 AND BETA < 0) OR (ALPHAR > 0 AND BETA > 0)
 * This is equivalent to: Re(lambda) = ALPHAR/BETA > 0
 *
 * Used internally by SB02OD for continuous-time algebraic Riccati equation solver.
 *
 * @param[in] alphar Pointer to real part of eigenvalue numerator
 * @param[in] alphai Pointer to imaginary part of eigenvalue numerator (unused)
 * @param[in] beta Pointer to eigenvalue denominator
 * @return 1 for unstable (Re(lambda) > 0), 0 otherwise
 */
int sb02ou(const f64* alphar, const f64* alphai, const f64* beta);

/**
 * @brief Select unstable generalized eigenvalues for discrete-time Riccati.
 *
 * Callback function for DGGES eigenvalue selection. Returns 1 (true) for
 * generalized eigenvalues with modulus >= 1 (unstable in discrete-time).
 *
 * Selection criterion: sqrt(ALPHAR^2 + ALPHAI^2) >= abs(BETA)
 * This is equivalent to: |lambda| = |ALPHAR + i*ALPHAI|/|BETA| >= 1
 *
 * Used internally by discrete-time algebraic Riccati equation solvers.
 *
 * @param[in] alphar Pointer to real part of eigenvalue numerator
 * @param[in] alphai Pointer to imaginary part of eigenvalue numerator
 * @param[in] beta Pointer to eigenvalue denominator
 * @return 1 for unstable (|lambda| >= 1), 0 otherwise
 */
int sb02ov(const f64* alphar, const f64* alphai, const f64* beta);

/**
 * @brief Select stable generalized eigenvalues for continuous-time Riccati.
 *
 * Callback function for DGGES eigenvalue selection. Returns 1 (true) for
 * generalized eigenvalues with negative real part (stable in continuous-time).
 *
 * Selection criterion: (ALPHAR < 0 AND BETA > 0) OR (ALPHAR > 0 AND BETA < 0)
 * This is equivalent to: Re(lambda) = ALPHAR/BETA < 0
 *
 * Used internally by SB02OD for continuous-time algebraic Riccati equation solver.
 *
 * @param[in] alphar Pointer to real part of eigenvalue numerator
 * @param[in] alphai Pointer to imaginary part of eigenvalue numerator (unused)
 * @param[in] beta Pointer to eigenvalue denominator
 * @return 1 for stable (Re(lambda) < 0), 0 otherwise
 */
int sb02ow(const f64* alphar, const f64* alphai, const f64* beta);

/**
 * @brief Select stable generalized eigenvalues for discrete-time Riccati.
 *
 * Callback function for DGGES eigenvalue selection. Returns 1 (true) for
 * generalized eigenvalues with modulus less than 1 (stable in discrete-time).
 *
 * Selection criterion: sqrt(ALPHAR^2 + ALPHAI^2) < abs(BETA)
 * This is equivalent to: |lambda| = |ALPHAR + i*ALPHAI|/|BETA| < 1
 *
 * Used internally by discrete-time algebraic Riccati equation solvers.
 *
 * @param[in] alphar Pointer to real part of eigenvalue numerator
 * @param[in] alphai Pointer to imaginary part of eigenvalue numerator
 * @param[in] beta Pointer to eigenvalue denominator
 * @return 1 for stable (|lambda| < 1), 0 otherwise
 */
int sb02ox(const f64* alphar, const f64* alphai, const f64* beta);

/**
 * @brief Solve complex Lyapunov equation for Cholesky factor of solution.
 *
 * Computes upper triangular complex matrix U such that X = op(U)^H * op(U) satisfies:
 *   Continuous: op(A)^H * X + X*op(A) = -scale^2 * op(B)^H * op(B)
 *   Discrete:   op(A)^H * X*op(A) - X = -scale^2 * op(B)^H * op(B)
 *
 * @param[in] dico 'C' for continuous-time, 'D' for discrete-time
 * @param[in] fact 'F' if Schur factorization provided, 'N' to compute it
 * @param[in] trans 'N' for op(K)=K, 'C' for op(K)=K^H
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in] m Number of rows of op(B) (m >= 0)
 * @param[in,out] a On entry: N-by-N complex matrix A
 *                  On exit: Upper triangular Schur form S
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] q On entry if FACT='F': Unitary matrix Q of Schur factorization
 *                  On exit: Unitary matrix Q of Schur factorization
 * @param[in] ldq Leading dimension of Q (ldq >= max(1,n))
 * @param[in,out] b On entry: Coefficient matrix B (M-by-N or N-by-M)
 *                  On exit: Upper triangular Cholesky factor U
 * @param[in] ldb Leading dimension of B
 * @param[out] scale Scale factor (0 < scale <= 1) to prevent overflow
 * @param[out] w Complex eigenvalues of A
 * @param[out] dwork Real workspace, dimension (n)
 * @param[out] zwork Complex workspace, dimension (lzwork)
 * @param[in] lzwork Length of zwork
 * @param[out] info Exit code:
 *                  0 = success
 *                  1 = near-singular (warning)
 *                  2 = A not stable/convergent (FACT='N')
 *                  3 = S not stable/convergent (FACT='F')
 *                  6 = ZGEES failed to converge
 */
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
    i32* info);

/**
 * @brief Solve continuous-time Lyapunov equation with separation estimation.
 *
 * Solves: op(A)' * X + X * op(A) = scale * C
 * where op(A) = A or A^T, C is symmetric.
 * Optionally estimates sep(op(A), -op(A)').
 *
 * @param[in] job 'X' solution only, 'S' separation only, 'B' both
 * @param[in] fact 'F' Schur form provided, 'N' compute Schur
 * @param[in] trana 'N' op(A)=A, 'T'/'C' op(A)=A^T
 * @param[in] n Order of A (n >= 0)
 * @param[in,out] a N-by-N matrix A. If FACT='N', overwritten with Schur form
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] u If FACT='F', input orthogonal Schur factor.
 *                  If FACT='N', output orthogonal Schur factor
 * @param[in] ldu Leading dimension of U (ldu >= max(1,n))
 * @param[in,out] c Symmetric N-by-N matrix C. Overwritten with solution X
 * @param[in] ldc Leading dimension of C
 * @param[out] scale Scale factor (0 < scale <= 1)
 * @param[out] sep Separation estimate (JOB='S' or 'B')
 * @param[out] ferr Forward error bound (JOB='B')
 * @param[out] wr Real parts of eigenvalues (FACT='N')
 * @param[out] wi Imaginary parts of eigenvalues (FACT='N')
 * @param[out] iwork Integer workspace (N*N) for JOB='S' or 'B'
 * @param[out] dwork Real workspace
 * @param[in] ldwork Length of dwork
 * @param[out] info 0 success, -i param i invalid, 1..n DGEES fail, n+1 singular
 */
void sb03rd(
    const char* job, const char* fact, const char* trana,
    const i32 n,
    f64* a, const i32 lda,
    f64* u, const i32 ldu,
    f64* c, const i32 ldc,
    f64* scale,
    f64* sep, f64* ferr,
    f64* wr, f64* wi,
    i32* iwork,
    f64* dwork, const i32 ldwork,
    i32* info);

/**
 * @brief Estimate conditioning and forward error bound for discrete-time Lyapunov.
 *
 * Estimates the conditioning and computes an error bound on the solution of the
 * real discrete-time Lyapunov matrix equation:
 *   op(A)' * X * op(A) - X = scale * C
 * where op(A) = A or A^T, C is symmetric (C = C^T).
 *
 * @param[in] job 'C': reciprocal condition number only, 'E': error bound only,
 *                'B': compute both
 * @param[in] fact 'F': Schur factorization provided in T and U,
 *                 'N': compute Schur factorization
 * @param[in] trana 'N': op(A) = A, 'T'/'C': op(A) = A^T
 * @param[in] uplo 'U': upper triangle of C used, 'L': lower triangle used
 * @param[in] lyapun 'O': solve original Lyapunov equations updating with U,
 *                   'R': solve reduced equations only
 * @param[in] n Order of matrices A, X, C. n >= 0.
 * @param[in] scale Scale factor from Lyapunov solver, 0 <= scale <= 1.
 * @param[in] a If FACT='N' or LYAPUN='O': N-by-N original matrix A.
 *              Not referenced if FACT='F' and LYAPUN='R'.
 * @param[in] lda Leading dimension of A.
 * @param[in,out] t If FACT='F': input upper quasi-triangular Schur form of A.
 *                  If FACT='N': output Schur form.
 * @param[in] ldt Leading dimension of T. ldt >= max(1,n).
 * @param[in,out] u If LYAPUN='O' and FACT='F': input orthogonal Schur factor.
 *                  If LYAPUN='O' and FACT='N': output orthogonal Schur factor.
 *                  Not referenced if LYAPUN='R'.
 * @param[in] ldu Leading dimension of U.
 * @param[in] c N-by-N symmetric matrix C (only triangle specified by UPLO used).
 * @param[in] ldc Leading dimension of C. ldc >= max(1,n).
 * @param[in,out] x N-by-N symmetric solution matrix. Modified internally, restored on exit.
 * @param[in] ldx Leading dimension of X. ldx >= max(1,n).
 * @param[out] sepd If JOB='C' or 'B': estimated sepd(op(A),op(A)').
 * @param[out] rcond If JOB='C' or 'B': reciprocal condition number estimate.
 * @param[out] ferr If JOB='E' or 'B': forward error bound estimate.
 * @param[out] iwork Integer workspace of dimension N*N.
 * @param[out] dwork Real workspace of dimension LDWORK.
 * @param[in] ldwork Workspace size. See documentation for requirements.
 *                   If LDWORK=-1, workspace query performed.
 * @param[out] info 0 on success, -i if argument i invalid,
 *                  1..n if DGEES failed, n+1 if near-reciprocal eigenvalues.
 */
void sb03sd(const char* job, const char* fact, const char* trana,
            const char* uplo, const char* lyapun, i32 n, f64 scale,
            const f64* a, i32 lda, f64* t, i32 ldt, f64* u, i32 ldu,
            const f64* c, i32 ldc, f64* x, i32 ldx, f64* sepd,
            f64* rcond, f64* ferr, i32* iwork, f64* dwork, i32 ldwork,
            i32* info);

/**
 * @brief Solve generalized Sylvester equations with separation estimation.
 *
 * Solves for R and L one of the generalized Sylvester equations:
 *   A * R - L * B = scale * C, D * R - L * E = scale * F    (equation 1)
 * or
 *   A' * R + D' * L = scale * C, R * B' + L * E' = scale * (-F)  (equation 2)
 *
 * The solution (R, L) overwrites (C, F). Optionally computes a Dif estimate
 * measuring the separation of (A,D) from (B,E).
 *
 * @param[in] reduce 'R': reduce both (A,D) and (B,E) to generalized Schur form,
 *                   'A': reduce (A,D) only, (B,E) already in Schur form,
 *                   'B': reduce (B,E) only, (A,D) already in Schur form,
 *                   'N': both already in generalized Schur form.
 * @param[in] trans 'N': solve equation (1), 'T': solve equation (2).
 * @param[in] jobd '1': one-norm Dif only, '2': Frobenius Dif only,
 *                 'D': solve (1) + one-norm Dif, 'F': solve (1) + Frobenius Dif,
 *                 'N': no Dif computation. Ignored if trans='T'.
 * @param[in] m Order of A, D; rows of C, F, R, L. m >= 0.
 * @param[in] n Order of B, E; cols of C, F, R, L. n >= 0.
 * @param[in,out] a M-by-M matrix A. Output: upper quasi-triangular form.
 * @param[in] lda Leading dimension of A. lda >= max(1,m).
 * @param[in,out] b N-by-N matrix B. Output: upper quasi-triangular form.
 * @param[in] ldb Leading dimension of B. ldb >= max(1,n).
 * @param[in,out] c M-by-N matrix C. Output: solution R.
 * @param[in] ldc Leading dimension of C. ldc >= max(1,m).
 * @param[in,out] d M-by-M matrix D. Output: upper triangular form.
 * @param[in] ldd Leading dimension of D. ldd >= max(1,m).
 * @param[in,out] e N-by-N matrix E. Output: upper triangular form.
 * @param[in] lde Leading dimension of E. lde >= max(1,n).
 * @param[in,out] f M-by-N matrix F. Output: solution L.
 * @param[in] ldf Leading dimension of F. ldf >= max(1,m).
 * @param[out] scale Scaling factor (0 <= scale <= 1).
 * @param[out] dif Dif estimate if jobd != 'N' and trans = 'N'.
 * @param[out] p M-by-M left transformation for (A,D) if reduce='R' or 'A'.
 * @param[in] ldp Leading dimension of P.
 * @param[out] q M-by-M right transformation for (A,D) if reduce='R' or 'A'.
 * @param[in] ldq Leading dimension of Q.
 * @param[out] u N-by-N left transformation for (B,E) if reduce='R' or 'B'.
 * @param[in] ldu Leading dimension of U.
 * @param[out] v N-by-N right transformation for (B,E) if reduce='R' or 'B'.
 * @param[in] ldv Leading dimension of V.
 * @param[out] iwork Integer workspace of dimension m+n+6.
 * @param[out] dwork Real workspace. On exit, dwork[0] = optimal ldwork.
 * @param[in] ldwork Workspace size. -1 for query.
 * @param[out] info 0=success, -i=argument i invalid, 1=Schur fail,
 *                  2=not quasi-triangular, 3=singular in solve.
 */
void sb04od(const char* reduce, const char* trans, const char* jobd,
            i32 m, i32 n,
            f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            f64* d, i32 ldd, f64* e, i32 lde, f64* f, i32 ldf,
            f64* scale, f64* dif,
            f64* p, i32 ldp, f64* q, i32 ldq,
            f64* u, i32 ldu, f64* v, i32 ldv,
            i32* iwork, f64* dwork, i32 ldwork,
            i32* info);

/**
 * @brief Solve periodic Sylvester equation with matrices in periodic Schur form.
 *
 * Solves a periodic Sylvester equation:
 *     A * R - L * B = scale * C
 *     D * L - R * E = scale * F
 *
 * where R and L are unknown M-by-N matrices, (A, D), (B, E) and (C, F) are
 * given matrix pairs of size M-by-M, N-by-N and M-by-N, respectively.
 * (A, D) and (B, E) must be in periodic Schur form, i.e., A, B are upper
 * quasi triangular and D, E are upper triangular. The solution (R, L)
 * overwrites (C, F). 0 <= scale <= 1 is an output scaling factor chosen
 * to avoid overflow.
 *
 * @param[in] m Order of A and D, row dimension of C, F, R, L. m > 0.
 * @param[in] n Order of B and E, column dimension of C, F, R, L. n > 0.
 * @param[in] a M-by-M upper quasi triangular matrix A.
 * @param[in] lda Leading dimension of A. lda >= max(1,m).
 * @param[in] b N-by-N upper quasi triangular matrix B.
 * @param[in] ldb Leading dimension of B. ldb >= max(1,n).
 * @param[in,out] c M-by-N matrix C. On exit, contains solution R.
 * @param[in] ldc Leading dimension of C. ldc >= max(1,m).
 * @param[in] d M-by-M upper triangular matrix D.
 * @param[in] ldd Leading dimension of D. ldd >= max(1,m).
 * @param[in] e N-by-N upper triangular matrix E.
 * @param[in] lde Leading dimension of E. lde >= max(1,n).
 * @param[in,out] f M-by-N matrix F. On exit, contains solution L.
 * @param[in] ldf Leading dimension of F. ldf >= max(1,m).
 * @param[out] scale Scaling factor (0 <= scale <= 1).
 * @param[out] iwork Integer workspace of dimension m+n+2.
 * @param[out] info 0=success, -i=argument i invalid,
 *                  >0=common eigenvalues in A*D and B*E.
 */
void sb04ow(i32 m, i32 n,
            const f64* a, i32 lda, const f64* b, i32 ldb,
            f64* c, i32 ldc, const f64* d, i32 ldd,
            const f64* e, i32 lde, f64* f, i32 ldf,
            f64* scale, i32* iwork, i32* info);

/**
 * @brief Construct right-hand sides for quasi-Hessenberg Sylvester solver (2 RHS).
 *
 * Constructs right-hand sides D for a system of equations in quasi-Hessenberg
 * form solved via SB04RX (case with 2 right-hand sides).
 *
 * For the Sylvester equation X + AXB = C:
 * - If abschr='B': ab contains B (m-by-m), ba contains A (n-by-n), D has 2*n elements
 * - If abschr='A': ab contains A (n-by-n), ba contains B (m-by-m), D has 2*m elements
 *
 * @param[in] abschr 'A' if ab contains A, 'B' if ab contains B.
 * @param[in] ul 'U' if ab is upper Hessenberg, 'L' if lower Hessenberg.
 * @param[in] n Order of matrix A. n >= 0.
 * @param[in] m Order of matrix B. m >= 0.
 * @param[in] c N-by-M coefficient/solution matrix.
 * @param[in] ldc Leading dimension of c. ldc >= max(1,n).
 * @param[in] indx Position of first column/row of c to use (1-based).
 * @param[in] ab N-by-N or M-by-M Hessenberg matrix (A or B).
 * @param[in] ldab Leading dimension of ab.
 * @param[in] ba N-by-N or M-by-M matrix (B or A, the one not in ab).
 * @param[in] ldba Leading dimension of ba.
 * @param[out] d Output right-hand side vector (2*n or 2*m elements, interleaved).
 * @param[out] dwork Workspace of dimension 2*n or 2*m.
 */
void sb04rv(const char* abschr, const char* ul,
            i32 n, i32 m, const f64* c, i32 ldc, i32 indx,
            const f64* ab, i32 ldab, const f64* ba, i32 ldba,
            f64* d, f64* dwork);

/**
 * @brief Construct right-hand side for Hessenberg Sylvester solver (1 RHS).
 *
 * Constructs right-hand side D for a system of equations in Hessenberg
 * form solved via SB04RY (case with 1 right-hand side).
 *
 * For the Sylvester equation X + AXB = C:
 * - If abschr='B': ab contains B (m-by-m), ba contains A (n-by-n), D has n elements
 * - If abschr='A': ab contains A (n-by-n), ba contains B (m-by-m), D has m elements
 *
 * @param[in] abschr 'A' if ab contains A, 'B' if ab contains B.
 * @param[in] ul 'U' if ab is upper Hessenberg, 'L' if lower Hessenberg.
 * @param[in] n Order of matrix A. n >= 0.
 * @param[in] m Order of matrix B. m >= 0.
 * @param[in] c N-by-M coefficient/solution matrix.
 * @param[in] ldc Leading dimension of c. ldc >= max(1,n).
 * @param[in] indx Position of column/row of c to use (1-based).
 * @param[in] ab N-by-N or M-by-M Hessenberg matrix (A or B).
 * @param[in] ldab Leading dimension of ab.
 * @param[in] ba N-by-N or M-by-M matrix (B or A, the one not in ab).
 * @param[in] ldba Leading dimension of ba.
 * @param[out] d Output right-hand side vector (n or m elements).
 * @param[out] dwork Workspace of dimension n or m.
 */
void sb04rw(const char* abschr, const char* ul,
            i32 n, i32 m, const f64* c, i32 ldc, i32 indx,
            const f64* ab, i32 ldab, const f64* ba, i32 ldba,
            f64* d, f64* dwork);

/**
 * @brief Solve quasi-Hessenberg system with two right-hand sides.
 *
 * Solves a system of equations in quasi-Hessenberg form (Hessenberg form
 * plus two consecutive offdiagonals) with two right-hand sides via QR
 * decomposition with Givens rotations.
 *
 * @param[in] rc 'R' for row transformations, 'C' for column transformations.
 * @param[in] ul 'U' if A is upper Hessenberg, 'L' if lower Hessenberg.
 * @param[in] m Order of matrix A. m >= 0.
 * @param[in] a M-by-M Hessenberg matrix.
 * @param[in] lda Leading dimension of a. lda >= max(1,m).
 * @param[in] lambd1 Element (1,1) of the 2x2 block.
 * @param[in] lambd2 Element (1,2) of the 2x2 block.
 * @param[in] lambd3 Element (2,1) of the 2x2 block.
 * @param[in] lambd4 Element (2,2) of the 2x2 block.
 * @param[in,out] d On entry: 2*M RHS vector, stored row-wise.
 *                  On exit: solution vector if info=0.
 * @param[in] tol Tolerance for near-singularity test. Matrix is singular
 *                if estimated rcond <= tol.
 * @param[out] iwork Integer workspace of dimension 2*M.
 * @param[out] dwork Real workspace of dimension (lddwor, 2*M+3).
 * @param[in] lddwor Leading dimension of dwork. lddwor >= max(1, 2*M).
 * @param[out] info 0 = success, 1 = system is (numerically) singular.
 */
void sb04rx(const char* rc, const char* ul,
            i32 m, const f64* a, i32 lda,
            f64 lambd1, f64 lambd2, f64 lambd3, f64 lambd4,
            f64* d, f64 tol, i32* iwork, f64* dwork, i32 lddwor,
            i32* info);

/**
 * @brief Solve Hessenberg system with one right-hand side.
 *
 * Solves a system of equations (I + LAMBDA * A) * x = d in Hessenberg form
 * with one right-hand side via QR decomposition with Givens rotations.
 *
 * @param[in] rc 'R' for row transformations, 'C' for column transformations.
 * @param[in] ul 'U' if A is upper Hessenberg, 'L' if lower Hessenberg.
 * @param[in] m Order of matrix A. m >= 0.
 * @param[in] a M-by-M Hessenberg matrix.
 * @param[in] lda Leading dimension of a. lda >= max(1,m).
 * @param[in] lambda Scalar multiplier for A.
 * @param[in,out] d On entry: M-element RHS vector.
 *                  On exit: solution vector if info=0.
 * @param[in] tol Tolerance for near-singularity test. Matrix is singular
 *                if estimated rcond <= tol.
 * @param[out] iwork Integer workspace of dimension M.
 * @param[out] dwork Real workspace of dimension (lddwor, M+3).
 * @param[in] lddwor Leading dimension of dwork. lddwor >= max(1, M).
 * @param[out] info 0 = success, 1 = system is (numerically) singular.
 */
void sb04ry(const char* rc, const char* ul,
            i32 m, const f64* a, i32 lda,
            f64 lambda, f64* d, f64 tol,
            i32* iwork, f64* dwork, i32 lddwor,
            i32* info);

/**
 * @brief Minimum norm feedback matrix for deadbeat control.
 *
 * Constructs the minimum norm feedback matrix F to perform "deadbeat control"
 * on a (A,B)-pair of a state-space model (which must be preliminarily reduced
 * to upper "staircase" form using AB01OD) such that R = A + BFU' is nilpotent.
 *
 * @param[in] n State dimension, order of matrix A. n >= 0.
 * @param[in] m Input dimension. m >= 0.
 * @param[in] kmax Number of "stairs" in staircase form from AB01OD. 0 <= kmax <= n.
 * @param[in,out] a On entry: N-by-N transformed state matrix in staircase form.
 *                  On exit: U'AU + U'BF.
 * @param[in] lda Leading dimension of a. lda >= max(1,n).
 * @param[in,out] b On entry: N-by-M transformed input matrix in triangular form.
 *                  On exit: U'B.
 * @param[in] ldb Leading dimension of b. ldb >= max(1,n).
 * @param[in] kstair KMAX-element array of stair dimensions from AB01OD.
 * @param[in,out] u On entry: N-by-N transformation matrix or identity.
 *                  On exit: U times the transformation reducing A+BFU' to Schur form.
 * @param[in] ldu Leading dimension of u. ldu >= max(1,n).
 * @param[out] f M-by-N deadbeat feedback matrix.
 * @param[in] ldf Leading dimension of f. ldf >= max(1,m).
 * @param[out] dwork Workspace of dimension 2*N.
 * @param[out] info 0 = success, < 0 = -i means i-th argument illegal.
 */
void sb06nd(i32 n, i32 m, i32 kmax,
            f64* a, i32 lda,
            f64* b, i32 ldb,
            const i32* kstair,
            f64* u, i32 ldu,
            f64* f, i32 ldf,
            f64* dwork, i32* info);

/**
 * @brief State-space representation from left coprime factorization.
 *
 * Constructs the state-space representation G = (A,B,C,D) from the factors
 * Q = (AQR,BQ,CQR,DQ) and R = (AQR,BR,CQR,DR) of its left coprime factorization
 * G = R^{-1} * Q.
 *
 * The formulas used are:
 *   A = AQR - BR * DR^{-1} * CQR
 *   B = BQ  - BR * DR^{-1} * DQ
 *   C = DR^{-1} * CQR
 *   D = DR^{-1} * DQ
 *
 * @param[in] n Order of matrix A. Also rows of B,BR and columns of C. n >= 0.
 * @param[in] m Input dimension, columns of B and D. m >= 0.
 * @param[in] p Output dimension, rows of C,D,DR and columns of BR,DR. p >= 0.
 * @param[in,out] a On entry: N-by-N state matrix AQR. On exit: state matrix of G.
 * @param[in] lda Leading dimension of a. lda >= max(1,n).
 * @param[in,out] b On entry: N-by-M input matrix BQ. On exit: input matrix of G.
 * @param[in] ldb Leading dimension of b. ldb >= max(1,n).
 * @param[in,out] c On entry: P-by-N output matrix CQR. On exit: output matrix of G.
 * @param[in] ldc Leading dimension of c. ldc >= max(1,p).
 * @param[in,out] d On entry: P-by-M feedthrough matrix DQ. On exit: feedthrough of G.
 * @param[in] ldd Leading dimension of d. ldd >= max(1,p).
 * @param[in] br N-by-P input matrix BR of system R.
 * @param[in] ldbr Leading dimension of br. ldbr >= max(1,n).
 * @param[in,out] dr On entry: P-by-P matrix DR. On exit: LU factorization of DR.
 * @param[in] lddr Leading dimension of dr. lddr >= max(1,p).
 * @param[out] dwork Workspace of dimension max(1, 4*p). dwork[0] = rcond on exit.
 * @param[out] iwork Integer workspace of dimension p.
 * @param[out] info 0 = success, 1 = DR singular, 2 = DR numerically singular (warning).
 */
void sb08gd(i32 n, i32 m, i32 p,
            f64* a, i32 lda,
            f64* b, i32 ldb,
            f64* c, i32 ldc,
            f64* d, i32 ldd,
            const f64* br, i32 ldbr,
            f64* dr, i32 lddr,
            f64* dwork, i32* iwork, i32* info);

/**
 * @brief Left coprime factorization with prescribed stability degree.
 *
 * Constructs, for a given system G = (A,B,C,D), an output injection matrix H
 * and an orthogonal transformation matrix Z, such that the systems
 *
 *      Q = (Z'*(A+H*C)*Z, Z'*(B+H*D), C*Z, D)
 * and
 *      R = (Z'*(A+H*C)*Z, Z'*H, C*Z, I)
 *
 * provide a stable left coprime factorization of G in the form G = R^{-1} * Q.
 * The eigenvalues of the resulting state dynamics matrix lie inside a given
 * stability domain.
 *
 * @param[in] dico 'C' for continuous-time, 'D' for discrete-time system.
 * @param[in] n State dimension (n >= 0).
 * @param[in] m Input dimension (m >= 0).
 * @param[in] p Output dimension (p >= 0).
 * @param[in] alpha Array of 2 elements: alpha[0] = stability degree,
 *                  alpha[1] = stability margin. For continuous: alpha < 0.
 *                  For discrete: 0 <= alpha < 1.
 * @param[in,out] a On entry: N-by-N state matrix. On exit: NQ-by-NQ state
 *                  matrix of Q in real Schur form. Leading NR-by-NR is AR.
 * @param[in] lda Leading dimension of a (>= max(1,n)).
 * @param[in,out] b On entry: N-by-M input matrix. On exit: NQ-by-M input
 *                  matrix BQ of numerator factor Q.
 * @param[in] ldb Leading dimension of b (>= max(1,n)).
 * @param[in,out] c On entry: P-by-N output matrix. On exit: P-by-NQ output
 *                  matrix CQ. First NR columns are CR.
 * @param[in] ldc Leading dimension of c (>= max(1,m,p) if n>0, else >= 1).
 * @param[in] d P-by-M feedthrough matrix. Modified internally but restored.
 * @param[in] ldd Leading dimension of d (>= max(1,m,p)).
 * @param[out] nq Order of resulting factors Q and R.
 * @param[out] nr Order of minimal realization of factor R.
 * @param[out] br NQ-by-P output injection matrix Z'*H. First NR rows form BR.
 * @param[in] ldbr Leading dimension of br (>= max(1,n)).
 * @param[out] dr P-by-P identity matrix (input/output matrix of R).
 * @param[in] lddr Leading dimension of dr (>= max(1,p)).
 * @param[in] tol Tolerance for observability tests. If <= 0, computed default.
 * @param[out] dwork Workspace of dimension ldwork.
 * @param[in] ldwork Workspace size (>= max(1, n*p + max(n*(n+5), 5*p, 4*m))).
 * @param[out] iwarn Warning: K violations of numerical stability condition.
 * @param[out] info 0=success, <0=parameter error, 1=Schur fail, 2=ordering fail.
 */
void sb08ed(
    const char* dico,
    const i32 n,
    const i32 m,
    const i32 p,
    const f64* alpha,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* d,
    const i32 ldd,
    i32* nq,
    i32* nr,
    f64* br,
    const i32 ldbr,
    f64* dr,
    const i32 lddr,
    const f64 tol,
    f64* dwork,
    const i32 ldwork,
    i32* iwarn,
    i32* info);

/**
 * @brief State-space representation from right coprime factorization.
 *
 * Constructs the state-space representation G = (A,B,C,D) from the factors
 * Q = (AQR,BQR,CQ,DQ) and R = (AQR,BQR,CR,DR) of its right coprime
 * factorization G = Q * R^{-1}.
 *
 * Method:
 *     A = AQR - BQR * DR^{-1} * CR
 *     B = BQR * DR^{-1}
 *     C = CQ - DQ * DR^{-1} * CR
 *     D = DQ * DR^{-1}
 *
 * @param[in] n Order of systems Q and R (N >= 0).
 * @param[in] m Number of inputs (M >= 0).
 * @param[in] p Number of outputs (P >= 0).
 * @param[in,out] a On entry: N-by-N state matrix AQR. On exit: state matrix of G.
 * @param[in] lda Leading dimension of a (>= max(1,N)).
 * @param[in,out] b On entry: N-by-M input matrix BQR. On exit: input matrix of G.
 * @param[in] ldb Leading dimension of b (>= max(1,N)).
 * @param[in,out] c On entry: P-by-N output matrix CQ. On exit: output matrix of G.
 * @param[in] ldc Leading dimension of c (>= max(1,P)).
 * @param[in,out] d On entry: P-by-M feedthrough DQ. On exit: feedthrough of G.
 * @param[in] ldd Leading dimension of d (>= max(1,P)).
 * @param[in] cr M-by-N output matrix CR of system R.
 * @param[in] ldcr Leading dimension of cr (>= max(1,M)).
 * @param[in,out] dr On entry: M-by-M feedthrough DR. On exit: LU factorization.
 * @param[in] lddr Leading dimension of dr (>= max(1,M)).
 * @param[out] dwork Workspace, dimension max(1, 4*M). dwork[0] = rcond(DR).
 * @param[out] info 0=success, 1=DR singular, 2=DR numerically singular (warning).
 */
void sb08hd(
    const i32 n,
    const i32 m,
    const i32 p,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* d,
    const i32 ldd,
    const f64* cr,
    const i32 ldcr,
    f64* dr,
    const i32 lddr,
    f64* dwork,
    i32* info);

/**
 * @brief Compute B(s) = A(s) * A(-s) and accuracy norm
 *
 * Computes the coefficients of B(s) = A(s) * A(-s) where A(s) is a
 * polynomial given in increasing powers of s. B(s) is returned in
 * increasing powers of s**2.
 *
 * @param[in] da Degree of polynomials A(s) and B(s). DA >= 0.
 * @param[in] a Array of dimension DA+1 containing coefficients of A(s)
 *              in increasing powers of s.
 * @param[out] b Array of dimension DA+1 containing coefficients of B(s)
 *               in increasing powers of s**2.
 * @param[in,out] epsb On entry: machine precision (DLAMCH("E")).
 *                     On exit: updated accuracy norm 3*maxsa*epsb.
 */
void sb08my(
    const i32 da,
    const f64* a,
    f64* b,
    f64* epsb);

/**
 * @brief Compute B(z) = A(1/z) * A(z) and accuracy norm for discrete-time
 *
 * Computes the coefficients of B(z) = A(1/z) * A(z) where A(z) is a
 * polynomial given in increasing powers of z. The output B contains
 * the autocorrelation coefficients of A.
 *
 * @param[in] da Degree of polynomials A(z) and B(z). DA >= 0.
 * @param[in] a Array of dimension DA+1 containing coefficients of A(z)
 *              in increasing powers of z.
 * @param[out] b Array of dimension DA+1 containing coefficients of B(z).
 *               b[i] = sum_{k=0}^{da-i} a[k] * a[k+i] (autocorrelation at lag i).
 * @param[out] epsb Accuracy norm: 3 * machine_epsilon * b[0].
 */
void sb08ny(
    const i32 da,
    const f64* a,
    f64* b,
    f64* epsb);

/**
 * @brief Spectral factorization of polynomials (continuous-time case)
 *
 * Computes a real polynomial E(s) such that:
 *   (a) E(-s) * E(s) = A(-s) * A(s)
 *   (b) E(s) is stable (all zeros have non-positive real parts)
 *
 * The input polynomial may be supplied as A(s) or as B(s) = A(-s) * A(s).
 *
 * @param[in] acona 'A': coefficients of A(s) supplied
 *                  'B': coefficients of B(s) = A(-s)*A(s) supplied
 * @param[in] da Degree of polynomials A(s) and E(s). DA >= 0.
 * @param[in,out] a On entry: coefficients in increasing powers of s (if ACONA='A')
 *                  or s**2 (if ACONA='B'). On exit: coefficients of B(s)
 *                  in increasing powers of s**2.
 * @param[out] res Accuracy estimate for E(s) coefficients.
 * @param[out] e Coefficients of spectral factor E(s) in increasing powers of s.
 * @param[out] dwork Workspace array of dimension LDWORK.
 * @param[in] ldwork Workspace size. LDWORK >= 5*DA+5.
 * @param[out] info 0: success
 *                  1: all A(i) = 0
 *                  2: ACONA='B' but B(s) not valid for spectral factorization
 *                  3: no convergence in 30 iterations
 *                  4: last iterate is unstable
 */
void sb08md(
    const char* acona,
    const i32 da,
    f64* a,
    f64* res,
    f64* e,
    f64* dwork,
    const i32 ldwork,
    i32* info);

/**
 * @brief Spectral factorization of polynomials (discrete-time case)
 *
 * Computes a real polynomial E(z) such that:
 *   (a) E(1/z) * E(z) = A(1/z) * A(z)
 *   (b) E(z) is stable (all zeros have modulus <= 1)
 *
 * The input polynomial may be supplied as A(z) or as B(z) = A(1/z) * A(z).
 *
 * @param[in] acona 'A': coefficients of A(z) supplied
 *                  'B': coefficients of B(z) = A(1/z)*A(z) supplied
 * @param[in] da Degree of polynomials A(z) and E(z). DA >= 0.
 * @param[in,out] a On entry: coefficients in increasing powers of z.
 *                  On exit: coefficients of B(z) in equation (1).
 * @param[out] res Accuracy estimate for E(z) coefficients.
 * @param[out] e Coefficients of spectral factor E(z) in increasing powers of z.
 * @param[out] dwork Workspace array of dimension LDWORK.
 * @param[in] ldwork Workspace size. LDWORK >= 5*DA+5.
 * @param[out] info 0: success
 *                  2: ACONA='B' but B(z) not valid for spectral factorization
 *                  3: no convergence in 30 iterations
 *                  4: last iterate is unstable
 */
void sb08nd(
    const char* acona,
    const i32 da,
    f64* a,
    f64* res,
    f64* e,
    f64* dwork,
    const i32 ldwork,
    i32* info);

/**
 * @brief Evaluation of closeness of two multivariable sequences.
 *
 * Compares two multivariable sequences M1(k) and M2(k) for k = 1,2,...,N
 * and evaluates their closeness. Each M1(k) and M2(k) is an NC by NB matrix.
 *
 * Computes:
 * - SS(i,j) = sum_{k=1}^{N} M1(i,j,k)^2             (sum of squares)
 * - SE(i,j) = sum_{k=1}^{N} (M1(i,j,k) - M2(i,j,k))^2  (quadratic error)
 * - PRE(i,j) = 100 * sqrt(SE(i,j) / SS(i,j))       (percentage relative error)
 *
 * @param[in] n Number of parameters (N >= 0).
 * @param[in] nc Number of rows in M1(k) and M2(k) (NC >= 0).
 * @param[in] nb Number of columns in M1(k) and M2(k) (NB >= 0).
 * @param[in] h1 NC-by-(N*NB) array containing M1(k) sequence.
 *               Element M1(i,j,k) stored at h1[i + ((k-1)*nb + j)*ldh1].
 * @param[in] ldh1 Leading dimension of h1 (>= max(1, NC)).
 * @param[in] h2 NC-by-(N*NB) array containing M2(k) sequence.
 *               Element M2(i,j,k) stored at h2[i + ((k-1)*nb + j)*ldh2].
 * @param[in] ldh2 Leading dimension of h2 (>= max(1, NC)).
 * @param[out] ss NC-by-NB sum-of-squares matrix.
 * @param[in] ldss Leading dimension of ss (>= max(1, NC)).
 * @param[out] se NC-by-NB quadratic error matrix.
 * @param[in] ldse Leading dimension of se (>= max(1, NC)).
 * @param[out] pre NC-by-NB percentage relative error matrix.
 * @param[in] ldpre Leading dimension of pre (>= max(1, NC)).
 * @param[in] tol Tolerance for computations. If TOL < EPS, EPS is used.
 * @param[out] info 0: success, <0: parameter -info had illegal value.
 */
void sb09md(
    const i32 n,
    const i32 nc,
    const i32 nb,
    const f64* h1,
    const i32 ldh1,
    const f64* h2,
    const i32 ldh2,
    f64* ss,
    const i32 ldss,
    f64* se,
    const i32 ldse,
    f64* pre,
    const i32 ldpre,
    const f64 tol,
    i32* info);

/**
 * @brief D-step in D-K iteration for continuous-time systems.
 *
 * Performs the D-step in the D-K iteration for mu-synthesis. Estimates
 * the structured singular value mu(jw) at given frequencies and optionally
 * fits state-space realizations for the D-scaling matrices.
 *
 * @param[in] nc Order of matrix A (NC >= 0).
 * @param[in] mp Order of matrix D (MP >= 0).
 * @param[in] lendat Length of frequency vector OMEGA (LENDAT >= 2).
 * @param[in] f Size of identity block I_f in D-scaling system (F >= 0).
 * @param[in,out] ord Max order per block in fitting. On exit, MAX(1, ORD).
 * @param[in] mnb Number of diagonal blocks (1 <= MNB <= MP).
 * @param[in] nblock Array(MNB) of block sizes.
 * @param[in] itype Array(MNB) of block types (1=real, 2=complex).
 * @param[in] qutol Tolerance for fit. If < 0, only mu estimated.
 * @param[in,out] a NC-by-NC closed-loop A matrix. On exit, Hessenberg form.
 * @param[in] lda Leading dimension of A (>= max(1,NC)).
 * @param[in,out] b NC-by-MP closed-loop B matrix. Transformed on exit.
 * @param[in] ldb Leading dimension of B (>= max(1,NC)).
 * @param[in,out] c MP-by-NC closed-loop C matrix. Transformed on exit.
 * @param[in] ldc Leading dimension of C (>= max(1,MP)).
 * @param[in] d MP-by-MP closed-loop D matrix.
 * @param[in] ldd Leading dimension of D (>= max(1,MP)).
 * @param[in] omega Array(LENDAT) of frequencies.
 * @param[out] totord Total order of D-scaling system.
 * @param[out] ad TOTORD-by-TOTORD D-scaling A matrix.
 * @param[in] ldad Leading dimension of AD.
 * @param[out] bd TOTORD-by-(MP+F) D-scaling B matrix.
 * @param[in] ldbd Leading dimension of BD.
 * @param[out] cd (MP+F)-by-TOTORD D-scaling C matrix.
 * @param[in] ldcd Leading dimension of CD.
 * @param[out] dd (MP+F)-by-(MP+F) D-scaling D matrix.
 * @param[in] lddd Leading dimension of DD.
 * @param[out] mju Array(LENDAT) of structured singular values.
 * @param[out] iwork Integer workspace.
 * @param[in] liwork Length of IWORK.
 * @param[out] dwork Double workspace.
 * @param[in] ldwork Length of DWORK.
 * @param[out] zwork Complex workspace.
 * @param[in] lzwork Length of ZWORK.
 * @param[out] info 0: success, <0: param error, 1: singular jw*I-A,
 *             2-5: block structure errors, 6: linear algebra error,
 *             7: eigenvalue/SVD error, 1i: SB10YD error i.
 */
void sb10md(
    i32 nc,
    i32 mp,
    i32 lendat,
    i32 f,
    i32* ord,
    i32 mnb,
    const i32* nblock,
    const i32* itype,
    f64 qutol,
    f64* a,
    i32 lda,
    f64* b,
    i32 ldb,
    f64* c,
    i32 ldc,
    const f64* d,
    i32 ldd,
    const f64* omega,
    i32* totord,
    f64* ad,
    i32 ldad,
    f64* bd,
    i32 ldbd,
    f64* cd,
    i32 ldcd,
    f64* dd,
    i32 lddd,
    f64* mju,
    i32* iwork,
    i32 liwork,
    f64* dwork,
    i32 ldwork,
    c128* zwork,
    i32 lzwork,
    i32* info);

/**
 * @brief Discrete-time loop shaping controller design.
 *
 * Computes the positive feedback controller K = [Ak, Bk; Ck, Dk]
 * for the shaped plant G = [A, B; C, 0] using the McFarlane-Glover method.
 *
 * @param[in] n Order of the plant (N >= 0).
 * @param[in] m Column size of matrix B (M >= 0).
 * @param[in] np Row size of matrix C (NP >= 0).
 * @param[in] a N-by-N system state matrix of shaped plant.
 * @param[in] lda Leading dimension of A (>= max(1,N)).
 * @param[in] b N-by-M system input matrix of shaped plant.
 * @param[in] ldb Leading dimension of B (>= max(1,N)).
 * @param[in] c NP-by-N system output matrix of shaped plant.
 * @param[in] ldc Leading dimension of C (>= max(1,NP)).
 * @param[in] factor = 1: optimal controller; > 1: suboptimal (FACTOR >= 1).
 * @param[out] ak N-by-N controller state matrix.
 * @param[in] ldak Leading dimension of AK (>= max(1,N)).
 * @param[out] bk N-by-NP controller input matrix.
 * @param[in] ldbk Leading dimension of BK (>= max(1,N)).
 * @param[out] ck M-by-N controller output matrix.
 * @param[in] ldck Leading dimension of CK (>= max(1,M)).
 * @param[out] dk M-by-NP controller feedthrough matrix.
 * @param[in] lddk Leading dimension of DK (>= max(1,M)).
 * @param[out] rcond Array(4) of reciprocal condition numbers:
 *             rcond[0]: P-Riccati, rcond[1]: Q-Riccati,
 *             rcond[2]: X-Riccati, rcond[3]: Rx + Bx'*X*Bx.
 * @param[out] iwork Integer workspace (2*max(N,NP+M)).
 * @param[out] dwork Double workspace.
 * @param[in] ldwork Workspace size (see routine for formula).
 * @param[out] bwork Logical workspace (2*N).
 * @param[out] info 0: success, <0: param -info invalid,
 *             1: P-Riccati failed, 2: Q-Riccati failed, 3: X-Riccati failed,
 *             4: eigenvalue computation failed, 5: singular matrix,
 *             6: closed-loop unstable.
 */
void sb10kd(
    const i32 n,
    const i32 m,
    const i32 np,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    const f64* c,
    const i32 ldc,
    const f64 factor,
    f64* ak,
    const i32 ldak,
    f64* bk,
    const i32 ldbk,
    f64* ck,
    const i32 ldck,
    f64* dk,
    const i32 lddk,
    f64* rcond,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* bwork,
    i32* info);

/**
 * @brief Compute H2 optimal discrete-time controller from normalized system.
 *
 * Transforms the controller matrices from the normalized system (as computed
 * by SB10SD) to the controller for the original system.
 *
 * @param[in] n Order of the system (>= 0).
 * @param[in] m Column size of matrix B (>= 0).
 * @param[in] np Row size of matrix C (>= 0).
 * @param[in] ncon Number of control inputs M2 (M >= NCON >= 0, NP-NMEAS >= NCON).
 * @param[in] nmeas Number of measurements NP2 (NP >= NMEAS >= 0, M-NCON >= NMEAS).
 * @param[in] d NP-by-M system matrix D. Only D22 submatrix is used.
 * @param[in] ldd Leading dimension of D (>= max(1,NP)).
 * @param[in] tu M2-by-M2 control transformation matrix from SB10PD.
 * @param[in] ldtu Leading dimension of TU (>= max(1,M2)).
 * @param[in] ty NP2-by-NP2 measurement transformation matrix from SB10PD.
 * @param[in] ldty Leading dimension of TY (>= max(1,NP2)).
 * @param[in,out] ak On entry: N-by-N controller state matrix for normalized system.
 *                   On exit: transformed controller state matrix.
 * @param[in] ldak Leading dimension of AK (>= max(1,N)).
 * @param[in,out] bk On entry: N-by-NMEAS controller input matrix for normalized system.
 *                   On exit: transformed controller input matrix.
 * @param[in] ldbk Leading dimension of BK (>= max(1,N)).
 * @param[in,out] ck On entry: NCON-by-N controller output matrix for normalized system.
 *                   On exit: transformed controller output matrix.
 * @param[in] ldck Leading dimension of CK (>= max(1,NCON)).
 * @param[in,out] dk On entry: NCON-by-NMEAS controller matrix for normalized system.
 *                   On exit: transformed controller feedthrough matrix.
 * @param[in] lddk Leading dimension of DK (>= max(1,NCON)).
 * @param[out] rcond Reciprocal condition number of (Im2 + DKHAT*D22).
 * @param[in] tol Tolerance for singularity test. If <= 0, sqrt(eps) is used.
 * @param[out] iwork Integer workspace (2*M2).
 * @param[out] dwork Double workspace.
 * @param[in] ldwork Workspace size (>= max(N*M2, N*NP2, M2*NP2, M2*M2+4*M2)).
 * @param[out] info 0: success, <0: param -info invalid,
 *             1: (Im2 + DKHAT*D22) singular or ill-conditioned.
 */
void sb10td(
    const i32 n,
    const i32 m,
    const i32 np,
    const i32 ncon,
    const i32 nmeas,
    const f64* d,
    const i32 ldd,
    const f64* tu,
    const i32 ldtu,
    const f64* ty,
    const i32 ldty,
    f64* ak,
    const i32 ldak,
    f64* bk,
    const i32 ldbk,
    f64* ck,
    const i32 ldck,
    f64* dk,
    const i32 lddk,
    f64* rcond,
    const f64 tol,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info);

/**
 * @brief Positive feedback controller for discrete-time system (D != 0).
 *
 * Computes the matrices of the positive feedback controller
 *   K = [Ak Bk; Ck Dk]
 * for the shaped plant G = [A B; C D] in the Discrete-Time Loop Shaping
 * Design Procedure.
 *
 * @param[in] n Order of the plant (n >= 0).
 * @param[in] m Column size of the matrix B (m >= 0).
 * @param[in] np Row size of the matrix C (np >= 0).
 * @param[in] a N-by-N system state matrix A of the shaped plant.
 * @param[in] lda Leading dimension of A (>= max(1,N)).
 * @param[in] b N-by-M system input matrix B of the shaped plant.
 * @param[in] ldb Leading dimension of B (>= max(1,N)).
 * @param[in] c NP-by-N system output matrix C of the shaped plant.
 * @param[in] ldc Leading dimension of C (>= max(1,NP)).
 * @param[in] d NP-by-M system input/output matrix D of the shaped plant.
 * @param[in] ldd Leading dimension of D (>= max(1,NP)).
 * @param[in] factor Performance factor (>= 1). factor=1 for optimal (not
 *                   recommended), factor>1 for suboptimal controller.
 * @param[out] ak N-by-N controller state matrix.
 * @param[in] ldak Leading dimension of AK (>= max(1,N)).
 * @param[out] bk N-by-NP controller input matrix.
 * @param[in] ldbk Leading dimension of BK (>= max(1,N)).
 * @param[out] ck M-by-N controller output matrix.
 * @param[in] ldck Leading dimension of CK (>= max(1,M)).
 * @param[out] dk M-by-NP controller feedthrough matrix.
 * @param[in] lddk Leading dimension of DK (>= max(1,M)).
 * @param[out] rcond Array of 6 reciprocal condition numbers:
 *             RCOND(1): P-Riccati equation system
 *             RCOND(2): Q-Riccati equation system
 *             RCOND(3): (gamma^2-1)*I - P*Q
 *             RCOND(4): Rx + Bx'*X*Bx
 *             RCOND(5): Ip + D*Dk_hat
 *             RCOND(6): Im + Dk_hat*D
 * @param[in] tol Tolerance for singularity tests. If <= 0, sqrt(eps) is used.
 *                Must be < 1.
 * @param[out] iwork Integer workspace (2*max(N, M+NP)).
 * @param[out] dwork Double workspace.
 * @param[in] ldwork Workspace size (>= 16*N*N + 5*M*M + 7*NP*NP + 6*M*N +
 *                   7*M*NP + 7*N*NP + 6*N + 2*(M+NP) + max(14*N+23,16*N,
 *                   2*M-1,2*NP-1)).
 * @param[out] bwork Logical workspace (2*N).
 * @param[out] info Exit code:
 *             0: success
 *             <0: if -i, parameter i had illegal value
 *             1: P-Riccati equation not solved successfully
 *             2: Q-Riccati equation not solved successfully
 *             3: eigenvalue/singular value iteration failed
 *             4: (gamma^2-1)*I - P*Q is singular
 *             5: Rx + Bx'*X*Bx is singular
 *             6: Ip + D*Dk_hat is singular
 *             7: Im + Dk_hat*D is singular
 *             8: Ip - D*Dk is singular
 *             9: Im - Dk*D is singular
 *             10: closed-loop system is unstable
 */
void sb10zd(
    const i32 n,
    const i32 m,
    const i32 np,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    const f64* c,
    const i32 ldc,
    const f64* d,
    const i32 ldd,
    const f64 factor,
    f64* ak,
    const i32 ldak,
    f64* bk,
    const i32 ldbk,
    f64* ck,
    const i32 ldck,
    f64* dk,
    const i32 lddk,
    f64* rcond,
    const f64 tol,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* bwork,
    i32* info);

/**
 * @brief Frequency-weighted controller reduction via balancing.
 *
 * Computes a reduced order controller (Acr,Bcr,Ccr,Dcr) for an original
 * state-space controller representation (Ac,Bc,Cc,Dc) using frequency-weighted
 * square-root or balancing-free square-root Balance & Truncate (B&T) or
 * Singular Perturbation Approximation (SPA) model reduction methods.
 *
 * The algorithm minimizes the norm of the frequency-weighted error ||V*(K-Kr)*W||
 * where K and Kr are the transfer-function matrices of the original and reduced
 * order controllers. V and W are special frequency-weighting transfer-function
 * matrices constructed to enforce closed-loop stability and/or performance.
 *
 * @param[in] dico Type of system: 'C' = continuous-time, 'D' = discrete-time
 * @param[in] jobc Controllability Grammian: 'S' = Enns, 'E' = enhanced
 * @param[in] jobo Observability Grammian: 'S' = Enns, 'E' = enhanced
 * @param[in] jobmr Model reduction method: 'B' = sqrt B&T, 'F' = balancing-free B&T,
 *            'S' = sqrt SPA, 'P' = balancing-free SPA
 * @param[in] weight Frequency-weighting type: 'N' = none, 'O' = output,
 *            'I' = input, 'P' = performance
 * @param[in] equil Equilibration: 'S' = scale, 'N' = no scaling
 * @param[in] ordsel Order selection: 'F' = fixed NCR, 'A' = automatic (via TOL1)
 * @param[in] n Order of open-loop system A matrix (n >= 0)
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] p Number of system outputs (p >= 0)
 * @param[in] nc Order of controller Ac matrix (nc >= 0)
 * @param[in,out] ncr On entry: desired order (ORDSEL='F'); on exit: actual order
 * @param[in] alpha Stability boundary for eigenvalues (<=0 for 'C', 0<=alpha<=1 for 'D')
 * @param[in,out] a Open-loop state matrix A, dimension (lda, n)
 * @param[in] lda Leading dimension of a (lda >= max(1, n))
 * @param[in,out] b Open-loop input matrix B, dimension (ldb, m)
 * @param[in] ldb Leading dimension of b (ldb >= max(1, n))
 * @param[in,out] c Open-loop output matrix C, dimension (ldc, n)
 * @param[in] ldc Leading dimension of c (ldc >= max(1, p))
 * @param[in] d Open-loop feedthrough matrix D, dimension (ldd, m)
 * @param[in] ldd Leading dimension of d (ldd >= max(1, p))
 * @param[in,out] ac Controller state matrix, dimension (ldac, nc); on exit: Acr
 * @param[in] ldac Leading dimension of ac (ldac >= max(1, nc))
 * @param[in,out] bc Controller input matrix, dimension (ldbc, p); on exit: Bcr
 * @param[in] ldbc Leading dimension of bc (ldbc >= max(1, nc))
 * @param[in,out] cc Controller output matrix, dimension (ldcc, nc); on exit: Ccr
 * @param[in] ldcc Leading dimension of cc (ldcc >= max(1, m))
 * @param[in,out] dc Controller feedthrough matrix, dimension (lddc, p); on exit: Dcr
 * @param[in] lddc Leading dimension of dc (lddc >= max(1, m))
 * @param[out] ncs Dimension of alpha-stable part of controller
 * @param[out] hsvc Frequency-weighted Hankel singular values, dimension (nc)
 * @param[in] tol1 Tolerance for order determination (ORDSEL='A'); if <=0, uses default
 * @param[in] tol2 Tolerance for minimal realization; if <=0, uses default
 * @param[out] iwork Integer workspace, dimension max(LIWRK1, LIWRK2)
 * @param[out] dwork Double workspace, dimension (ldwork)
 * @param[in] ldwork Size of dwork
 * @param[out] iwarn Warning indicator: 0=OK, 1=NCR>NSMIN, 2=repeated SVs, 3=NCR<NCU
 * @param[out] info Exit code: 0=success, <0=parameter error, 1-7=algorithm error
 */
void sb16ad(
    const char* dico,
    const char* jobc,
    const char* jobo,
    const char* jobmr,
    const char* weight,
    const char* equil,
    const char* ordsel,
    const i32 n,
    const i32 m,
    const i32 p,
    const i32 nc,
    i32* ncr,
    const f64 alpha,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    const f64* d,
    const i32 ldd,
    f64* ac,
    const i32 ldac,
    f64* bc,
    const i32 ldbc,
    f64* cc,
    const i32 ldcc,
    f64* dc,
    const i32 lddc,
    i32* ncs,
    f64* hsvc,
    const f64 tol1,
    const f64 tol2,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* iwarn,
    i32* info);

/**
 * @brief Cholesky factors of frequency-weighted controllability and
 *        observability Grammians for controller reduction.
 *
 * Computes for given state-space representations (A,B,C,D) of the open-loop
 * system G and (Ac,Bc,Cc,Dc) of the feedback controller K, the Cholesky
 * factors S and R of the frequency-weighted controllability Grammian
 * P = S*S' and observability Grammian Q = R'*R.
 *
 * The controller must stabilize the closed-loop system. The state matrix Ac
 * must be in block-diagonal real Schur form: Ac = diag(Ac1, Ac2), where Ac1
 * contains unstable eigenvalues and Ac2 contains stable eigenvalues.
 *
 * @param[in] dico Type of system:
 *            'C': continuous-time
 *            'D': discrete-time
 * @param[in] jobc Choice of controllability Grammian:
 *            'S': standard Enns' method
 *            'E': stability enhanced modified Enns' method
 * @param[in] jobo Choice of observability Grammian:
 *            'S': standard Enns' method
 *            'E': stability enhanced modified combination method
 * @param[in] weight Type of frequency-weighting:
 *            'N': no weightings (V=I, W=I)
 *            'O': stability enforcing left weighting V=(I-G*K)^(-1)*G
 *            'I': stability enforcing right weighting W=(I-G*K)^(-1)*G
 *            'P': both V=(I-G*K)^(-1)*G and W=(I-G*K)^(-1)
 * @param[in] n Order of open-loop system A matrix (n >= 0)
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] p Number of system outputs (p >= 0)
 * @param[in] nc Order of controller Ac matrix (nc >= 0)
 * @param[in] ncs Dimension of stable part Ac2 (0 <= ncs <= nc)
 * @param[in] a State matrix A of G, dimension (lda, n)
 * @param[in] lda Leading dimension of a (lda >= max(1, n))
 * @param[in] b Input matrix B, dimension (ldb, m)
 * @param[in] ldb Leading dimension of b (ldb >= max(1, n))
 * @param[in] c Output matrix C, dimension (ldc, n)
 * @param[in] ldc Leading dimension of c (ldc >= max(1, p))
 * @param[in] d Feedthrough matrix D, dimension (ldd, m)
 * @param[in] ldd Leading dimension of d (ldd >= max(1, p))
 * @param[in] ac Controller state matrix in block-diagonal Schur form,
 *            dimension (ldac, nc)
 * @param[in] ldac Leading dimension of ac (ldac >= max(1, nc))
 * @param[in] bc Controller input matrix, dimension (ldbc, p)
 * @param[in] ldbc Leading dimension of bc (ldbc >= max(1, nc))
 * @param[in] cc Controller output matrix, dimension (ldcc, nc)
 * @param[in] ldcc Leading dimension of cc (ldcc >= max(1, m))
 * @param[in] dc Controller feedthrough matrix, dimension (lddc, p)
 * @param[in] lddc Leading dimension of dc (lddc >= max(1, m))
 * @param[out] scalec Scaling factor for controllability Grammian
 * @param[out] scaleo Scaling factor for observability Grammian
 * @param[out] s Upper triangular Cholesky factor of P = S*S',
 *             dimension (lds, ncs)
 * @param[in] lds Leading dimension of s (lds >= max(1, ncs))
 * @param[out] r Upper triangular Cholesky factor of Q = R'*R,
 *             dimension (ldr, ncs)
 * @param[in] ldr Leading dimension of r (ldr >= max(1, ncs))
 * @param[out] iwork Integer workspace (0 if weight='N', else 2*(m+p))
 * @param[out] dwork Double workspace
 * @param[in] ldwork Size of dwork (>= max(1, LFREQ)) where:
 *            - if weight='I','O','P': LFREQ = (n+nc)*(n+nc+2*m+2*p) +
 *              max((n+nc)*(n+nc+max(n+nc,m,p)+7), (m+p)*(m+p+4))
 *            - if weight='N': LFREQ = ncs*(max(m,p)+5)
 * @param[out] info Exit code:
 *             0: success
 *             <0: if -i, parameter i had illegal value
 *             1: closed-loop system not well-posed (singular feedthrough)
 *             2: Schur form computation failed
 *             3: closed-loop state matrix not stable
 *             4: symmetric eigenproblem failed
 *             5: Ac2 not stable or not in Schur form
 */
void sb16ay(
    const char* dico,
    const char* jobc,
    const char* jobo,
    const char* weight,
    const i32 n,
    const i32 m,
    const i32 p,
    const i32 nc,
    const i32 ncs,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    const f64* c,
    const i32 ldc,
    const f64* d,
    const i32 ldd,
    const f64* ac,
    const i32 ldac,
    const f64* bc,
    const i32 ldbc,
    const f64* cc,
    const i32 ldcc,
    const f64* dc,
    const i32 lddc,
    f64* scalec,
    f64* scaleo,
    f64* s,
    const i32 lds,
    f64* r,
    const i32 ldr,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info);

/**
 * @brief Coprime factorization based state feedback controller reduction.
 *
 * Computes a reduced order controller model (Ac,Bc,Cc,Dc) for a given open-loop
 * model (A,B,C,D) with state feedback gain F and observer gain G using coprime
 * factorization based model reduction methods (B&T or SPA).
 *
 * @param[in] dico System type: 'C' = continuous-time, 'D' = discrete-time
 * @param[in] jobd Feedthrough: 'D' = D present, 'Z' = D is zero
 * @param[in] jobmr Reduction method: 'B' = sqrt B&T, 'F' = balancing-free B&T,
 *                  'S' = sqrt SPA, 'P' = balancing-free SPA
 * @param[in] jobcf Coprime factorization: 'L' = left, 'R' = right
 * @param[in] equil Equilibration: 'S' = scale, 'N' = no scaling
 * @param[in] ordsel Order selection: 'F' = fixed order NCR, 'A' = automatic
 * @param[in] n Original system order (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] p Number of outputs (p >= 0)
 * @param[in,out] ncr On entry with ORDSEL='F': desired controller order.
 *                    On exit: actual reduced controller order
 * @param[in,out] a On entry: N-by-N state matrix. On exit: NCR-by-NCR Ac
 * @param[in] lda Leading dimension of A (>= max(1,n))
 * @param[in] b N-by-M input matrix B
 * @param[in] ldb Leading dimension of B (>= max(1,n))
 * @param[in] c P-by-N output matrix C
 * @param[in] ldc Leading dimension of C (>= max(1,p))
 * @param[in] d P-by-M feedthrough matrix D (if JOBD='D')
 * @param[in] ldd Leading dimension of D (>= max(1,p) if JOBD='D', else >= 1)
 * @param[in,out] f On entry: M-by-N state feedback gain. On exit: M-by-NCR Cc
 * @param[in] ldf Leading dimension of F (>= max(1,m))
 * @param[in,out] g On entry: N-by-P observer gain. On exit: NCR-by-P Bc
 * @param[in] ldg Leading dimension of G (>= max(1,n))
 * @param[out] dc M-by-P controller feedthrough matrix Dc
 * @param[in] lddc Leading dimension of DC (>= max(1,m))
 * @param[out] hsv N Hankel singular values of extended system (decreasing)
 * @param[in] tol1 Tolerance for order selection (ORDSEL='A'), 0 = default
 * @param[in] tol2 Tolerance for minimal realization, 0 = default
 * @param[out] iwork Integer workspace
 * @param[out] dwork Double workspace
 * @param[in] ldwork Workspace size (see documentation for formula)
 * @param[out] iwarn Warning: 1 = NCR > minimal order with ORDSEL='F'
 * @param[out] info Error code: 0 = success, <0 = invalid parameter,
 *                  1 = Schur reduction of A+G*C failed,
 *                  2 = A+G*C not stable/convergent,
 *                  3 = Hankel singular value computation failed,
 *                  4 = Schur reduction of A+B*F failed,
 *                  5 = A+B*F not stable/convergent
 */
void sb16bd(
    const char* dico,
    const char* jobd,
    const char* jobmr,
    const char* jobcf,
    const char* equil,
    const char* ordsel,
    const i32 n,
    const i32 m,
    const i32 p,
    i32* ncr,
    f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    const f64* c,
    const i32 ldc,
    const f64* d,
    const i32 ldd,
    f64* f,
    const i32 ldf,
    f64* g,
    const i32 ldg,
    f64* dc,
    const i32 lddc,
    f64* hsv,
    const f64 tol1,
    const f64 tol2,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* iwarn,
    i32* info);

/**
 * @brief Cholesky factors of Grammians for coprime factors of state-feedback controller.
 *
 * Computes Cholesky factors Su and Ru of controllability Grammian P = Su*Su'
 * and observability Grammian Q = Ru'*Ru for frequency-weighted model reduction
 * of coprime factors of state-feedback controller.
 *
 * @param[in] dico 'C' for continuous-time, 'D' for discrete-time
 * @param[in] jobcf 'L' for left coprime factorization, 'R' for right
 * @param[in] n Order of state matrix A (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] p Number of outputs (p >= 0)
 * @param[in] a State matrix A, dimension (lda, n)
 * @param[in] lda Leading dimension of A (>= max(1,n))
 * @param[in] b Input matrix B, dimension (ldb, m)
 * @param[in] ldb Leading dimension of B (>= max(1,n))
 * @param[in] c Output matrix C, dimension (ldc, n)
 * @param[in] ldc Leading dimension of C (>= max(1,p))
 * @param[in] f State feedback gain F, dimension (ldf, n)
 * @param[in] ldf Leading dimension of F (>= max(1,m))
 * @param[in] g Observer gain G, dimension (ldg, p)
 * @param[in] ldg Leading dimension of G (>= max(1,n))
 * @param[out] scalec Scaling factor for controllability Grammian
 * @param[out] scaleo Scaling factor for observability Grammian
 * @param[out] s Cholesky factor Su of P = Su*Su', dimension (lds, n)
 * @param[in] lds Leading dimension of S (>= max(1,n))
 * @param[out] r Cholesky factor Ru of Q = Ru'*Ru, dimension (ldr, n)
 * @param[in] ldr Leading dimension of R (>= max(1,n))
 * @param[out] dwork Double workspace
 * @param[in] ldwork Size of dwork (see documentation)
 * @param[out] info Error code: 0 = success, <0 = invalid parameter,
 *             1 = eigenvalue computation failure,
 *             2 = A+G*C not stable,
 *             3 = A+B*F not stable,
 *             4 = observability Lyapunov equation singular,
 *             5 = controllability Lyapunov equation singular
 */
void sb16cy(
    const char* dico,
    const char* jobcf,
    const i32 n,
    const i32 m,
    const i32 p,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    const f64* c,
    const i32 ldc,
    const f64* f,
    const i32 ldf,
    const f64* g,
    const i32 ldg,
    f64* scalec,
    f64* scaleo,
    f64* s,
    const i32 lds,
    f64* r,
    const i32 ldr,
    f64* dwork,
    const i32 ldwork,
    i32* info);

/**
 * @brief Frequency-weighted coprime factorization controller reduction.
 *
 * Computes reduced order controller using coprime factorization with
 * frequency-weighted B&T (Balance & Truncate) model reduction.
 *
 * @param[in] dico System type: 'C'=continuous, 'D'=discrete
 * @param[in] jobd D matrix: 'D'=present, 'Z'=zero
 * @param[in] jobmr Reduction method: 'B'=sqrt B&T, 'F'=balancing-free sqrt B&T
 * @param[in] jobcf Coprime factorization: 'L'=left, 'R'=right
 * @param[in] ordsel Order selection: 'F'=fixed, 'A'=automatic
 * @param[in] n Order of original system (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] p Number of outputs (p >= 0)
 * @param[in,out] ncr On entry (ordsel='F'): desired order. On exit: actual order.
 * @param[in,out] a State matrix A, dimension (lda, n). On exit: reduced Ac.
 * @param[in] lda Leading dimension of A (>= max(1,n))
 * @param[in,out] b Input matrix B, dimension (ldb, m). On exit: overwritten.
 * @param[in] ldb Leading dimension of B (>= max(1,n))
 * @param[in,out] c Output matrix C, dimension (ldc, n). On exit: overwritten.
 * @param[in] ldc Leading dimension of C (>= max(1,p))
 * @param[in] d Feedthrough D, dimension (ldd, m). Not used if jobd='Z'.
 * @param[in] ldd Leading dimension of D
 * @param[in,out] f State feedback F, dimension (ldf, n). On exit: reduced Cc.
 * @param[in] ldf Leading dimension of F (>= max(1,m))
 * @param[in,out] g Observer gain G, dimension (ldg, p). On exit: reduced Bc.
 * @param[in] ldg Leading dimension of G (>= max(1,n))
 * @param[out] hsv Hankel singular values, dimension (n)
 * @param[in] tol Tolerance for automatic order selection
 * @param[out] iwork Integer workspace
 * @param[out] dwork Double workspace
 * @param[in] ldwork Size of dwork
 * @param[out] iwarn Warning: 0=ok, 1=NCR>minimal, 2=repeated singular values
 * @param[out] info Error code: 0=success, <0=invalid param, 1=eigenvalue failure,
 *             2=A+G*C unstable, 3=A+B*F unstable, 4=obs Lyapunov singular,
 *             5=ctrl Lyapunov singular, 6=HSV computation failed
 */
void sb16cd(
    const char* dico,
    const char* jobd,
    const char* jobmr,
    const char* jobcf,
    const char* ordsel,
    const i32 n,
    const i32 m,
    const i32 p,
    i32* ncr,
    f64* a, const i32 lda,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    const f64* d, const i32 ldd,
    f64* f, const i32 ldf,
    f64* g, const i32 ldg,
    f64* hsv,
    const f64 tol,
    i32* iwork,
    f64* dwork, const i32 ldwork,
    i32* iwarn,
    i32* info);

#ifdef __cplusplus
}
#endif

#endif /* SLICOT_SB_H */
