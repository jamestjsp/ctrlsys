/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 *
 * AUTO-GENERATED from docstrings.json - DO NOT EDIT DIRECTLY
 * Regenerate with: python tools/generate_docstrings.py
 */

#ifndef SLICOT_DOCSTRINGS_H
#define SLICOT_DOCSTRINGS_H

#define DOC_AB01MD "Controllable realization for single-input systems.\n" \
    "\n" \
    "Finds a controllable realization for the linear time-invariant\n" \
    "single-input system dX/dt = A*X + B*U, reducing to orthogonal\n" \
    "canonical form using orthogonal similarity transformations.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobz (str): 'N' (no Z), 'F' (factored form), 'I' (identity init)\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input vector B (n,), F-order\n" \
    "  tol (float): Tolerance (0 for default)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, ncont, z, tau, info): Transformed matrices, controllable order,\n" \
    "                               transformation matrix, reflectors, exit code"

#define DOC_AB01ND "Controllable realization for multi-input systems.\n" \
    "\n" \
    "Finds a controllable realization for the linear time-invariant\n" \
    "multi-input system dX/dt = A*X + B*U, reducing (A,B) to orthogonal\n" \
    "canonical form with upper block Hessenberg structure.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobz (str): 'N' (no Z), 'F' (factored form), 'I' (identity init)\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  tol (float): Tolerance (0 for default)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, ncont, indcon, nblk, z, tau, info):\n" \
    "    - a: Transformed A (block Hessenberg)\n" \
    "    - b: Transformed B\n" \
    "    - ncont: Order of controllable part\n" \
    "    - indcon: Controllability index\n" \
    "    - nblk: Block sizes (first indcon elements)\n" \
    "    - z: Transformation matrix\n" \
    "    - tau: Elementary reflector factors\n" \
    "    - info: Exit code (0 = success)"

#define DOC_AB01OD "Staircase form for multi-input systems.\n" \
    "\n" \
    "Reduces (A,B) to upper staircase form using orthogonal state-space\n" \
    "and input-space transformations U and V.\n" \
    "\n" \
    "Parameters:\n" \
    "  stages (str): 'F' (forward), 'B' (backward), 'A' (all)\n" \
    "  jobu (str): 'N' (no U), 'I' (return U)\n" \
    "  jobv (str): 'N' (no V), 'I' (return V)\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  tol (float): Tolerance (0 for default)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, u, v, ncont, indcon, kstair, info):\n" \
    "    - a: Transformed A (U'*A*U)\n" \
    "    - b: Transformed B (U'*B*V or U'*B)\n" \
    "    - u: State transformation matrix\n" \
    "    - v: Input transformation matrix\n" \
    "    - ncont: Order of controllable part\n" \
    "    - indcon: Controllability index\n" \
    "    - kstair: Block sizes (first indcon elements)\n" \
    "    - info: Exit code (0 = success)"

#define DOC_AB04MD "Bilinear transformation of state-space system.\n" \
    "\n" \
    "Performs discrete-time <-> continuous-time conversion via bilinear\n" \
    "transformation of the state-space matrices (A,B,C,D).\n" \
    "\n" \
    "Parameters:\n" \
    "  type (str): 'D' for discrete->continuous, 'C' for continuous->discrete\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (p x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (p x m, F-order)\n" \
    "  alpha (float, optional): Transformation parameter (default 1.0)\n" \
    "  beta (float, optional): Transformation parameter (default 1.0)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, c, d, info): Transformed matrices and exit code"

#define DOC_AB05MD "Cascade (series) inter-connection of two state-space systems.\n" \
    "\n" \
    "Computes the state-space model (A,B,C,D) for the cascaded connection\n" \
    "of two systems G1 and G2, where output of G1 feeds input of G2:\n" \
    "Y = G2(G1(U))\n" \
    "\n" \
    "Parameters:\n" \
    "  uplo (str): 'L' for lower block diagonal, 'U' for upper block diagonal\n" \
    "  over (str): 'N' no overlap, 'O' overlap arrays\n" \
    "  a1 (ndarray): State matrix of G1 (n1 x n1, F-order)\n" \
    "  b1 (ndarray): Input matrix of G1 (n1 x m1, F-order)\n" \
    "  c1 (ndarray): Output matrix of G1 (p1 x n1, F-order)\n" \
    "  d1 (ndarray): Feedthrough matrix of G1 (p1 x m1, F-order)\n" \
    "  a2 (ndarray): State matrix of G2 (n2 x n2, F-order)\n" \
    "  b2 (ndarray): Input matrix of G2 (n2 x p1, F-order)\n" \
    "  c2 (ndarray): Output matrix of G2 (p2 x n2, F-order)\n" \
    "  d2 (ndarray): Feedthrough matrix of G2 (p2 x p1, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, c, d, n, info): Cascaded system matrices, state order, exit code"

#define DOC_AB05ND "Feedback inter-connection of two state-space systems.\n" \
    "\n" \
    "Computes the state-space model (A,B,C,D) for the feedback connection\n" \
    "of two systems G1 and G2:\n" \
    "  U = U1 + alpha*Y2,  Y = Y1 = U2\n" \
    "  alpha = +1: positive feedback\n" \
    "  alpha = -1: negative feedback\n" \
    "\n" \
    "Parameters:\n" \
    "  over (str): 'N' no overlap, 'O' overlap arrays\n" \
    "  alpha (float): Feedback coefficient (+1 or -1)\n" \
    "  a1 (ndarray): State matrix of G1 (n1 x n1, F-order)\n" \
    "  b1 (ndarray): Input matrix of G1 (n1 x m1, F-order)\n" \
    "  c1 (ndarray): Output matrix of G1 (p1 x n1, F-order)\n" \
    "  d1 (ndarray): Feedthrough matrix of G1 (p1 x m1, F-order)\n" \
    "  a2 (ndarray): State matrix of G2 (n2 x n2, F-order)\n" \
    "  b2 (ndarray): Input matrix of G2 (n2 x p1, F-order)\n" \
    "  c2 (ndarray): Output matrix of G2 (m1 x n2, F-order)\n" \
    "  d2 (ndarray): Feedthrough matrix of G2 (m1 x p1, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, c, d, n, info): Feedback system matrices, state order, exit code"

#define DOC_AB05OD "Rowwise concatenation of two state-space systems.\n" \
    "\n" \
    "Computes the state-space model (A,B,C,D) for rowwise concatenation\n" \
    "(parallel inter-connection with separate inputs):\n" \
    "  Y = G1*U1 + alpha*G2*U2\n" \
    "\n" \
    "Combined system has:\n" \
    "  A = [[A1, 0], [0, A2]]  (block diagonal)\n" \
    "  B = [[B1, 0], [0, B2]]  (block diagonal)\n" \
    "  C = [C1, alpha*C2]      (rowwise concatenation)\n" \
    "  D = [D1, alpha*D2]      (rowwise concatenation)\n" \
    "\n" \
    "Parameters:\n" \
    "  over (str): 'N' no overlap, 'O' overlap arrays\n" \
    "  a1 (ndarray): State matrix of G1 (n1 x n1, F-order)\n" \
    "  b1 (ndarray): Input matrix of G1 (n1 x m1, F-order)\n" \
    "  c1 (ndarray): Output matrix of G1 (p1 x n1, F-order)\n" \
    "  d1 (ndarray): Feedthrough matrix of G1 (p1 x m1, F-order)\n" \
    "  a2 (ndarray): State matrix of G2 (n2 x n2, F-order)\n" \
    "  b2 (ndarray): Input matrix of G2 (n2 x m2, F-order)\n" \
    "  c2 (ndarray): Output matrix of G2 (p1 x n2, F-order)\n" \
    "  d2 (ndarray): Feedthrough matrix of G2 (p1 x m2, F-order)\n" \
    "  alpha (float): Coefficient multiplying second system\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, c, d, n, m, info): Combined system matrices, state order, input count, exit code"

#define DOC_AB05PD "Parallel inter-connection of two state-space systems (same inputs).\n" \
    "\n" \
    "Computes the state-space model (A,B,C,D) for the sum G = G1 + alpha*G2\n" \
    "where both systems share the same inputs:\n" \
    "  Y = G1*U + alpha*G2*U\n" \
    "\n" \
    "Combined system has:\n" \
    "  A = [[A1, 0], [0, A2]]    (block diagonal)\n" \
    "  B = [[B1], [B2]]          (stacked vertically)\n" \
    "  C = [C1, alpha*C2]        (concatenated horizontally)\n" \
    "  D = D1 + alpha*D2         (matrix sum)\n" \
    "\n" \
    "Parameters:\n" \
    "  n1 (int): Number of states in first system\n" \
    "  m (int): Number of inputs (same for both systems)\n" \
    "  p (int): Number of outputs (same for both systems)\n" \
    "  n2 (int): Number of states in second system\n" \
    "  alpha (float): Coefficient multiplying second system\n" \
    "  a1 (ndarray): State matrix of G1 (n1 x n1, F-order)\n" \
    "  b1 (ndarray): Input matrix of G1 (n1 x m, F-order)\n" \
    "  c1 (ndarray): Output matrix of G1 (p x n1, F-order)\n" \
    "  d1 (ndarray): Feedthrough matrix of G1 (p x m, F-order)\n" \
    "  a2 (ndarray): State matrix of G2 (n2 x n2, F-order)\n" \
    "  b2 (ndarray): Input matrix of G2 (n2 x m, F-order)\n" \
    "  c2 (ndarray): Output matrix of G2 (p x n2, F-order)\n" \
    "  d2 (ndarray): Feedthrough matrix of G2 (p x m, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (n, a, b, c, d, info): State order, combined system matrices, exit code"

#define DOC_AB05QD "Append two systems in state-space form (block diagonal).\n" \
    "\n" \
    "Constructs G = diag(G1, G2) with separate inputs and outputs:\n" \
    "  A = [[A1, 0], [0, A2]]   (block diagonal)\n" \
    "  B = [[B1, 0], [0, B2]]   (block diagonal)\n" \
    "  C = [[C1, 0], [0, C2]]   (block diagonal)\n" \
    "  D = [[D1, 0], [0, D2]]   (block diagonal)\n" \
    "\n" \
    "Parameters:\n" \
    "  over (str): 'N' no overlap, 'O' overlap arrays\n" \
    "  a1 (ndarray): State matrix of G1 (n1 x n1, F-order)\n" \
    "  b1 (ndarray): Input matrix of G1 (n1 x m1, F-order)\n" \
    "  c1 (ndarray): Output matrix of G1 (p1 x n1, F-order)\n" \
    "  d1 (ndarray): Feedthrough matrix of G1 (p1 x m1, F-order)\n" \
    "  a2 (ndarray): State matrix of G2 (n2 x n2, F-order)\n" \
    "  b2 (ndarray): Input matrix of G2 (n2 x m2, F-order)\n" \
    "  c2 (ndarray): Output matrix of G2 (p2 x n2, F-order)\n" \
    "  d2 (ndarray): Feedthrough matrix of G2 (p2 x m2, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, c, d, n, m, p, info): Combined system matrices,\n" \
    "    state order, input count, output count, exit code"

#define DOC_AB05RD "Closed-loop system for mixed output and state feedback control law.\n" \
    "\n" \
    "Constructs the closed-loop system (Ac,Bc,Cc,Dc) for the control law:\n" \
    "  u = alpha*F*y + beta*K*x + G*v\n" \
    "  z = H*y\n" \
    "\n" \
    "Closed-loop matrices:\n" \
    "  E = (I - alpha*D*F)^(-1)\n" \
    "  A1 = A + alpha*B*F*E*C,  B1 = B + alpha*B*F*E*D\n" \
    "  C1 = E*C,                D1 = E*D\n" \
    "  Ac = A1 + beta*B1*K,     Bc = B1*G\n" \
    "  Cc = H*(C1 + beta*D1*K), Dc = H*D1*G\n" \
    "\n" \
    "Parameters:\n" \
    "  fbtype (str): 'I' for identity feedback (F=I), 'O' for general feedback\n" \
    "  jobd (str): 'D' if D is present, 'Z' if D is zero matrix\n" \
    "  n (int): Order of state matrix A (n >= 0)\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  p (int): Number of outputs (p >= 0, p=m if fbtype='I')\n" \
    "  mv (int): Dimension of new input v (mv >= 0)\n" \
    "  pz (int): Dimension of new output z (pz >= 0)\n" \
    "  alpha (float): Coefficient in output feedback law\n" \
    "  beta (float): Coefficient in state feedback law\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (p x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (p x m, F-order)\n" \
    "  f (ndarray): Feedback matrix F (m x p, F-order)\n" \
    "  k (ndarray): State feedback matrix K (m x n, F-order)\n" \
    "  g (ndarray): Input scaling matrix G (m x mv, F-order)\n" \
    "  h (ndarray): Output scaling matrix H (pz x p, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (Ac, Bc, Cc, Dc, rcond, info): Closed-loop matrices,\n" \
    "    reciprocal condition number, exit code (0=success, 1=singular)"

#define DOC_AB05SD "Closed-loop system for output feedback control law.\n" \
    "\n" \
    "Constructs the closed-loop system (Ac,Bc,Cc,Dc) for the output\n" \
    "feedback control law u = alpha*F*y + v.\n" \
    "\n" \
    "Closed-loop matrices:\n" \
    "  E = (I - alpha*D*F)^(-1)\n" \
    "  Ac = A + alpha*B*F*E*C\n" \
    "  Bc = B + alpha*B*F*E*D\n" \
    "  Cc = E*C\n" \
    "  Dc = E*D\n" \
    "\n" \
    "Parameters:\n" \
    "  fbtype (str): 'I' for identity feedback (F=I), 'O' for general feedback\n" \
    "  jobd (str): 'D' if D is present, 'Z' if D is zero matrix\n" \
    "  n (int): Order of state matrix A (n >= 0)\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  p (int): Number of outputs (p >= 0, p=m if fbtype='I')\n" \
    "  alpha (float): Feedback gain coefficient\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (p x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (p x m, F-order)\n" \
    "  f (ndarray): Feedback matrix F (m x p, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (Ac, Bc, Cc, Dc, rcond, info): Closed-loop matrices,\n" \
    "    reciprocal condition number, exit code (0=success, 1=singular)"

#define DOC_AB07MD "Dual of a state-space representation.\n" \
    "\n" \
    "Finds the dual of a given state-space representation (A,B,C,D).\n" \
    "If M-input/P-output, its dual is P-input/M-output (A',C',B',D').\n" \
    "\n" \
    "Parameters:\n" \
    "  jobd (str): 'D' if D is present, 'Z' if D is zero matrix\n" \
    "  n (int): Order of state-space representation (n >= 0)\n" \
    "  m (int): Number of system inputs (m >= 0)\n" \
    "  p (int): Number of system outputs (p >= 0)\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x max(m,p), F-order)\n" \
    "  c (ndarray): Output matrix C (max(m,p) x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (max(m,p) x max(m,p), F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a_dual, b_dual, c_dual, d_dual, info): Dual system matrices and exit code"

#define DOC_AB07ND "Compute the inverse of a linear system.\n" \
    "\n" \
    "Computes the inverse (Ai,Bi,Ci,Di) of a given system (A,B,C,D):\n" \
    "  Ai = A - B*D^-1*C, Bi = -B*D^-1, Ci = D^-1*C, Di = D^-1.\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (m x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (m x m, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (ai, bi, ci, di, rcond, info): Inverse system matrices,\n" \
    "    reciprocal condition number, and exit code"

#define DOC_AB08MD "Compute normal rank of transfer-function matrix.\n" \
    "\n" \
    "Computes the normal rank of the transfer-function matrix of a\n" \
    "state-space model (A,B,C,D).\n" \
    "\n" \
    "Parameters:\n" \
    "  equil (str): 'S' to balance compound matrix, 'N' for no balancing\n" \
    "  n (int): Number of state variables (>= 0)\n" \
    "  m (int): Number of system inputs (>= 0)\n" \
    "  p (int): Number of system outputs (>= 0)\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (p x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (p x m, F-order)\n" \
    "  tol (float, optional): Tolerance for rank decisions (default 0)\n" \
    "\n" \
    "Returns:\n" \
    "  (rank, info): Normal rank and exit code (0 = success)"

#define DOC_AB08MZ "Compute normal rank of transfer-function matrix (complex case).\n" \
    "\n" \
    "Computes the normal rank of the transfer-function matrix of a\n" \
    "complex state-space model (A,B,C,D).\n" \
    "\n" \
    "Parameters:\n" \
    "  equil (str): 'S' to balance compound matrix, 'N' for no balancing\n" \
    "  n (int): Number of state variables (>= 0)\n" \
    "  m (int): Number of system inputs (>= 0)\n" \
    "  p (int): Number of system outputs (>= 0)\n" \
    "  a (ndarray): State matrix A (n x n, complex F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, complex F-order)\n" \
    "  c (ndarray): Output matrix C (p x n, complex F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (p x m, complex F-order)\n" \
    "  tol (float, optional): Tolerance for rank decisions (default 0)\n" \
    "\n" \
    "Returns:\n" \
    "  (rank, info): Normal rank and exit code (0 = success)"

#define DOC_AB08ND "Construct regular pencil for invariant zeros.\n" \
    "\n" \
    "Constructs a regular pencil (Af - lambda*Bf) for a state-space system\n" \
    "(A,B,C,D) whose generalized eigenvalues are the invariant zeros.\n" \
    "Also computes orders of infinite zeros and Kronecker indices.\n" \
    "\n" \
    "Parameters:\n" \
    "  equil (str): 'S' to balance compound matrix, 'N' for no balancing\n" \
    "  n (int): Number of state variables (>= 0)\n" \
    "  m (int): Number of system inputs (>= 0)\n" \
    "  p (int): Number of system outputs (>= 0)\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (p x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (p x m, F-order)\n" \
    "  tol (float, optional): Tolerance for rank decisions (default 0)\n" \
    "\n" \
    "Returns:\n" \
    "  (nu, rank, dinfz, nkror, nkrol, infz, kronr, kronl, af, bf, info):\n" \
    "    - nu: Number of finite invariant zeros\n" \
    "    - rank: Normal rank of transfer function matrix\n" \
    "    - dinfz: Maximum degree of infinite elementary divisors\n" \
    "    - nkror: Number of right Kronecker indices\n" \
    "    - nkrol: Number of left Kronecker indices\n" \
    "    - infz: Array of infinite zero degrees\n" \
    "    - kronr: Right Kronecker (column) indices\n" \
    "    - kronl: Left Kronecker (row) indices\n" \
    "    - af: Coefficient matrix Af of reduced pencil\n" \
    "    - bf: Coefficient matrix Bf of reduced pencil\n" \
    "    - info: Exit code (0 = success)"

#define DOC_AB08NW "Extract regular pencil for finite Smith zeros.\n" \
    "\n" \
    "Extracts from the system pencil S(lambda) = (A-lambda*I, B; C, D) a\n" \
    "regular pencil Af-lambda*Ef whose generalized eigenvalues are the\n" \
    "finite Smith zeros. Also computes orders of infinite Smith zeros and\n" \
    "Kronecker indices.\n" \
    "\n" \
    "Parameters:\n" \
    "  equil (str): 'S' to balance system matrix, 'N' for no balancing\n" \
    "  n (int): Number of state variables (>= 0)\n" \
    "  m (int): Number of system inputs (>= 0)\n" \
    "  p (int): Number of system outputs (>= 0)\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (p x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (p x m, F-order)\n" \
    "  tol (float, optional): Tolerance for rank decisions (default 0)\n" \
    "\n" \
    "Returns:\n" \
    "  (af, e, nfz, nrank, niz, dinfz, nkror, ninfe, nkrol, infz, kronr, infe, kronl, info):\n" \
    "    - af: Matrix Af of reduced pencil (n x n)\n" \
    "    - e: Matrix Ef of reduced pencil (n x n)\n" \
    "    - nfz: Number of finite zeros\n" \
    "    - nrank: Normal rank of system pencil\n" \
    "    - niz: Number of infinite zeros\n" \
    "    - dinfz: Maximal multiplicity of infinite Smith zeros\n" \
    "    - nkror: Number of right Kronecker indices\n" \
    "    - ninfe: Number of elementary infinite blocks\n" \
    "    - nkrol: Number of left Kronecker indices\n" \
    "    - infz: Array of infinite zero degrees\n" \
    "    - kronr: Right Kronecker (column) indices\n" \
    "    - infe: Multiplicities of infinite eigenvalues\n" \
    "    - kronl: Left Kronecker (row) indices\n" \
    "    - info: Exit code (0 = success)"

#define DOC_AB08NX "Extract reduced system with full row rank D.\n" \
    "\n" \
    "From (N+P)-by-(M+N) compound matrix [B A; D C], extracts reduced\n" \
    "system (NU+MU)-by-(M+NU) [B' A'; D' C'] with same transmission\n" \
    "zeros but D' of full row rank.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Number of state variables (>= 0)\n" \
    "  m (int): Number of system inputs (>= 0)\n" \
    "  p (int): Number of system outputs (>= 0)\n" \
    "  ro (int): P for original, max(P-M,0) for pertransposed\n" \
    "  sigma (int): 0 for original, M for pertransposed\n" \
    "  svlmax (float): Max singular value estimate (>= 0)\n" \
    "  abcd (ndarray): Compound matrix ((N+P) x (M+N), F-order)\n" \
    "  ninfz (int): Current number of infinite zeros (init 0)\n" \
    "  tol (float): Tolerance for rank decisions\n" \
    "\n" \
    "Returns:\n" \
    "  (abcd, ro, sigma, ninfz, mu, nu, nkrol, infz, kronl, info)"

#define DOC_AB08NY "Extract reduced system pencil with Dr of full row rank.\n" \
    "\n" \
    "From (N+P)-by-(M+N) system pencil (B, A-lambda*I; D, C), extracts\n" \
    "reduced (NR+PR)-by-(M+NR) pencil (Br, Ar-lambda*I; Dr, Cr) with\n" \
    "same transmission zeros but Dr of full row rank.\n" \
    "\n" \
    "Parameters:\n" \
    "  first (bool): True for first call, False for subsequent calls\n" \
    "  n (int): Number of state variables (>= 0)\n" \
    "  m (int): Number of system inputs (>= 0, <= p if first=False)\n" \
    "  p (int): Number of system outputs (>= 0)\n" \
    "  svlmax (float): Max singular value estimate (>= 0)\n" \
    "  abcd (ndarray): Compound matrix ((N+P) x (M+N), F-order)\n" \
    "  ninfz (int): Current number of infinite zeros (init 0 for first=True)\n" \
    "  tol (float): Tolerance for rank decisions (< 1)\n" \
    "\n" \
    "Returns:\n" \
    "  (abcd, ninfz, nr, pr, dinfz, nkronl, infz, kronl, info)"

#define DOC_AB08NZ "Construct regular pencil for invariant zeros of complex system.\n" \
    "\n" \
    "Computes (Af-lambda*Bf) whose generalized eigenvalues are the\n" \
    "invariant zeros of the system (A,B,C,D).\n" \
    "\n" \
    "Parameters:\n" \
    "  equil (str): 'S' scale, 'N' no scaling\n" \
    "  a (ndarray): N-by-N complex state matrix A (F-order)\n" \
    "  b (ndarray): N-by-M complex input matrix B (F-order)\n" \
    "  c (ndarray): P-by-N complex output matrix C (F-order)\n" \
    "  d (ndarray): P-by-M complex feedthrough D (F-order)\n" \
    "  tol (float): Tolerance for rank decisions\n" \
    "\n" \
    "Returns:\n" \
    "  (nu, rank, dinfz, nkror, nkrol, infz, kronr, kronl, af, bf, info)"

#define DOC_AB09AD "Balance & Truncate model reduction for stable systems.\n" \
    "\n" \
    "Computes a reduced order model (Ar,Br,Cr) using either square-root\n" \
    "or balancing-free square-root Balance & Truncate method.\n" \
    "The routine first reduces A to real Schur form, then performs\n" \
    "model reduction using AB09AX.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  job (str): 'B' for square-root B&T, 'N' for balancing-free B&T\n" \
    "  equil (str): 'S' to scale (A,B,C), 'N' for no scaling\n" \
    "  ordsel (str): 'F' for fixed order, 'A' for automatic selection\n" \
    "  n (int): Order of original system (n >= 0)\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  p (int): Number of outputs (p >= 0)\n" \
    "  nr (int): Desired order if ordsel='F' (0 <= nr <= n)\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (p x n, F-order)\n" \
    "  tol (float): Tolerance for order selection (used if ordsel='A')\n" \
    "\n" \
    "Returns:\n" \
    "  (ar, br, cr, hsv, nr, iwarn, info): Reduced system matrices,\n" \
    "    Hankel singular values, actual order, warning indicator, exit code"

#define DOC_AB09AX "Balance & Truncate model reduction for stable systems.\n" \
    "\n" \
    "Computes a reduced order model (Ar,Br,Cr) using either square-root\n" \
    "or balancing-free square-root Balance & Truncate method.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  job (str): 'B' for square-root B&T, 'N' for balancing-free B&T\n" \
    "  ordsel (str): 'F' for fixed order, 'A' for automatic selection\n" \
    "  n (int): Order of original system (n >= 0)\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  p (int): Number of outputs (p >= 0)\n" \
    "  nr (int): Desired order if ordsel='F' (0 <= nr <= n)\n" \
    "  a (ndarray): State matrix A in Schur form (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (p x n, F-order)\n" \
    "  tol (float): Tolerance for order selection (used if ordsel='A')\n" \
    "\n" \
    "Returns:\n" \
    "  (ar, br, cr, hsv, t, ti, nr, iwarn, info): Reduced system matrices,\n" \
    "    Hankel singular values, truncation matrices, actual order,\n" \
    "    warning indicator, and exit code"

#define DOC_AB09BD "SPA model reduction for stable systems.\n" \
    "\n" \
    "Computes a reduced order model (Ar,Br,Cr,Dr) for a stable original\n" \
    "state-space representation (A,B,C,D) using either square-root or\n" \
    "balancing-free square-root Singular Perturbation Approximation (SPA).\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  job (str): 'B' for square-root SPA, 'N' for balancing-free SPA\n" \
    "  equil (str): 'S' to equilibrate (A,B,C), 'N' for no scaling\n" \
    "  ordsel (str): 'F' for fixed order, 'A' for automatic selection\n" \
    "  n (int): Order of original system (n >= 0)\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  p (int): Number of outputs (p >= 0)\n" \
    "  nr (int): Desired order if ordsel='F' (0 <= nr <= n)\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (p x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (p x m, F-order)\n" \
    "  tol1 (float): Tolerance for order selection (used if ordsel='A')\n" \
    "  tol2 (float): Tolerance for minimal realization order\n" \
    "\n" \
    "Returns:\n" \
    "  (ar, br, cr, dr, nr, hsv, iwarn, info):\n" \
    "    Reduced system matrices, actual order, Hankel singular values,\n" \
    "    warning indicator, and exit code"

#define DOC_AB09BX "Compute reduced order model using square-root SPA method.\n" \
    "\n" \
    "Computes a reduced order model (Ar,Br,Cr,Dr) for a stable original\n" \
    "state-space representation (A,B,C,D) using either square-root or\n" \
    "balancing-free square-root Singular Perturbation Approximation (SPA).\n" \
    "\n" \
    "The state matrix A must be in real Schur canonical form.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  job (str): 'B' for square-root SPA, 'N' for balancing-free SPA\n" \
    "  ordsel (str): 'F' for fixed order, 'A' for automatic selection\n" \
    "  n (int): Order of original system (n >= 0)\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  p (int): Number of outputs (p >= 0)\n" \
    "  nr (int): Desired order if ordsel='F' (0 <= nr <= n)\n" \
    "  a (ndarray): State matrix A in Schur form (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (p x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (p x m, F-order)\n" \
    "  tol1 (float): Tolerance for order selection (used if ordsel='A')\n" \
    "  tol2 (float): Tolerance for minimal realization order\n" \
    "\n" \
    "Returns:\n" \
    "  (ar, br, cr, dr, nr, hsv, t, ti, nmin, iwarn, info):\n" \
    "    Reduced system matrices, actual order, Hankel singular values,\n" \
    "    truncation matrices, minimal order, warning, and exit code"

#define DOC_AB09CD "Optimal Hankel-norm approximation based model reduction for stable systems.\n" \
    "\n" \
    "Computes a reduced order model (Ar,Br,Cr,Dr) for a stable original\n" \
    "state-space representation (A,B,C,D) using the optimal Hankel-norm\n" \
    "approximation method in conjunction with square-root balancing.\n" \
    "\n" \
    "Unlike AB09CX, this routine accepts a general state matrix A (not\n" \
    "necessarily in Schur form) and optionally performs equilibration.\n" \
    "\n" \
    "Error bound: HSV(NR) <= ||G-Gr||_inf <= 2*sum(HSV(NR+1:N))\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  equil (str): 'S' to equilibrate (A,B,C), 'N' for no scaling\n" \
    "  ordsel (str): 'F' for fixed order, 'A' for automatic selection\n" \
    "  n (int): Order of original system (n >= 0)\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  p (int): Number of outputs (p >= 0)\n" \
    "  nr (int): Desired order if ordsel='F' (0 <= nr <= n)\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (p x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (p x m, F-order)\n" \
    "  tol1 (float): Tolerance for order selection (used if ordsel='A')\n" \
    "  tol2 (float): Tolerance for minimal realization order\n" \
    "\n" \
    "Returns:\n" \
    "  (ar, br, cr, dr, nr, hsv, nmin, iwarn, info):\n" \
    "    Reduced system matrices, actual order, Hankel singular values,\n" \
    "    minimal order, warning indicator, and exit code"

#define DOC_AB09CX "Optimal Hankel-norm approximation model reduction for stable systems.\n" \
    "\n" \
    "Computes a reduced order model (Ar,Br,Cr,Dr) for a stable original\n" \
    "state-space representation (A,B,C,D) using the optimal Hankel-norm\n" \
    "approximation method in conjunction with square-root balancing.\n" \
    "\n" \
    "The state matrix A must be in real Schur canonical form.\n" \
    "\n" \
    "Error bound: HSV(NR) <= ||G-Gr||_inf <= 2*sum(HSV(NR+1:N))\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  ordsel (str): 'F' for fixed order, 'A' for automatic selection\n" \
    "  n (int): Order of original system (n >= 0)\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  p (int): Number of outputs (p >= 0)\n" \
    "  nr (int): Desired order if ordsel='F' (0 <= nr <= n)\n" \
    "  a (ndarray): State matrix A in Schur form (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (p x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (p x m, F-order)\n" \
    "  tol1 (float): Tolerance for order selection (used if ordsel='A')\n" \
    "  tol2 (float): Tolerance for minimal realization order\n" \
    "\n" \
    "Returns:\n" \
    "  (ar, br, cr, dr, nr, hsv, nmin, iwarn, info):\n" \
    "    Reduced system matrices, actual order, Hankel singular values,\n" \
    "    minimal order, warning indicator, and exit code"

#define DOC_AB09DD "Compute reduced order model using singular perturbation approximation.\n" \
    "\n" \
    "Computes a reduced order model (Ar,Br,Cr,Dr) of order NR from a\n" \
    "given system (A,B,C,D) using residualization formulas.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  n (int): Order of original system (n >= 0)\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  p (int): Number of outputs (p >= 0)\n" \
    "  nr (int): Order of reduced system (0 <= nr <= n)\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (p x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (p x m, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (ar, br, cr, dr, rcond, info): Reduced system matrices,\n" \
    "    reciprocal condition number, and exit code"

#define DOC_AB09ED "Optimal Hankel-norm approximation model reduction for unstable systems.\n" \
    "\n" \
    "Computes a reduced order model (Ar,Br,Cr,Dr) for an original\n" \
    "state-space representation (A,B,C,D) using the optimal Hankel-norm\n" \
    "approximation method in conjunction with square-root balancing for\n" \
    "the ALPHA-stable part of the system.\n" \
    "\n" \
    "The system is decomposed as G = G1 + G2 where G1 has ALPHA-stable\n" \
    "poles and G2 has ALPHA-unstable poles. G1 is reduced using Hankel-norm\n" \
    "approximation while G2 is kept intact.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  equil (str): 'S' to equilibrate (A,B,C), 'N' for no scaling\n" \
    "  ordsel (str): 'F' for fixed order, 'A' for automatic selection\n" \
    "  n (int): Order of original system (n >= 0)\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  p (int): Number of outputs (p >= 0)\n" \
    "  nr (int): Desired order if ordsel='F' (0 <= nr <= n)\n" \
    "  alpha (float): Stability boundary. Continuous: alpha <= 0, Discrete: 0 <= alpha <= 1\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (p x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (p x m, F-order)\n" \
    "  tol1 (float): Tolerance for order selection (used if ordsel='A')\n" \
    "  tol2 (float): Tolerance for minimal realization order\n" \
    "\n" \
    "Returns:\n" \
    "  (ar, br, cr, dr, nr, ns, hsv, nmin, iwarn, info):\n" \
    "    Reduced system matrices, actual order, dimension of stable part,\n" \
    "    Hankel singular values, minimal order, warning, and exit code"

#define DOC_AB09FD "Balance & Truncate model reduction for unstable systems using coprime factorization.\n" \
    "\n" \
    "Computes a reduced order model (Ar,Br,Cr) for an original\n" \
    "state-space representation (A,B,C) by using either the square-root\n" \
    "or the balancing-free square-root Balance & Truncate (B&T)\n" \
    "model reduction method in conjunction with stable coprime\n" \
    "factorization techniques.\n" \
    "\n" \
    "The algorithm computes a stable coprime factorization of G:\n" \
    "  G = R^{-1}*Q (left) or G = Q*R^{-1} (right)\n" \
    "Then applies B&T model reduction to the extended system Ge = (Q,R)\n" \
    "or Ge = [Q;R], and recovers the reduced system from reduced factors.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  jobcf (str): 'L' for left coprime factorization, 'R' for right\n" \
    "  fact (str): 'S' for prescribed stability degree, 'I' for inner denominator\n" \
    "  jobmr (str): 'B' for square-root B&T, 'N' for balancing-free B&T\n" \
    "  equil (str): 'S' to equilibrate (A,B,C), 'N' for no scaling\n" \
    "  ordsel (str): 'F' for fixed order, 'A' for automatic selection\n" \
    "  n (int): Order of original system (n >= 0)\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  p (int): Number of outputs (p >= 0)\n" \
    "  nr (int): Desired order if ordsel='F' (0 <= nr <= n)\n" \
    "  alpha (float): Stability degree. Continuous: alpha < 0, Discrete: 0 <= alpha < 1.\n" \
    "                 Not used if fact='I'.\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (p x n, F-order)\n" \
    "  tol1 (float): Tolerance for order selection (used if ordsel='A')\n" \
    "  tol2 (float): Tolerance for controllability/observability tests\n" \
    "\n" \
    "Returns:\n" \
    "  (ar, br, cr, nr, nq, hsv, iwarn, info):\n" \
    "    Reduced system matrices (ar, br, cr), actual order (nr),\n" \
    "    order of coprime factorization (nq), Hankel singular values (hsv),\n" \
    "    warning indicator (iwarn), and exit code (info)"

#define DOC_AB09GD "SPA model reduction for unstable systems with coprime factorization.\n" \
    "\n" \
    "Computes a reduced order model (Ar,Br,Cr,Dr) for an original\n" \
    "state-space representation (A,B,C,D) by using either the square-root\n" \
    "or the balancing-free square-root Singular Perturbation Approximation\n" \
    "(SPA) model reduction method in conjunction with stable coprime\n" \
    "factorization techniques.\n" \
    "\n" \
    "The algorithm computes a stable coprime factorization of G:\n" \
    "  G = R^{-1}*Q (left) or G = Q*R^{-1} (right)\n" \
    "Then applies SPA model reduction to the extended system Ge = (Q,R)\n" \
    "or Ge = [Q;R], and recovers the reduced system from reduced factors.\n" \
    "\n" \
    "Error bound: HSV(NR) <= ||Ge-Ger||_inf <= 2*sum(HSV(NR+1:NQ))\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  jobcf (str): 'L' for left coprime factorization, 'R' for right\n" \
    "  fact (str): 'S' for prescribed stability degree, 'I' for inner denominator\n" \
    "  jobmr (str): 'B' for square-root B&T, 'N' for balancing-free B&T\n" \
    "  equil (str): 'S' to equilibrate (A,B,C), 'N' for no scaling\n" \
    "  ordsel (str): 'F' for fixed order, 'A' for automatic selection\n" \
    "  n (int): Order of original system (n >= 0)\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  p (int): Number of outputs (p >= 0)\n" \
    "  nr (int): Desired order if ordsel='F' (0 <= nr <= n)\n" \
    "  alpha (float): Stability degree. Continuous: alpha < 0, Discrete: 0 <= alpha < 1.\n" \
    "                 Not used if fact='I'.\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (p x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (p x m, F-order)\n" \
    "  tol1 (float): Tolerance for order selection (used if ordsel='A')\n" \
    "  tol2 (float): Tolerance for minimal realization order\n" \
    "  tol3 (float): Tolerance for controllability/observability tests\n" \
    "\n" \
    "Returns:\n" \
    "  (ar, br, cr, dr, nr, nq, hsv, iwarn, info):\n" \
    "    Reduced system matrices (ar, br, cr, dr), actual order (nr),\n" \
    "    order of extended system (nq), Hankel singular values (hsv),\n" \
    "    warning indicator (iwarn), and exit code (info)"

#define DOC_AB09HD "Stochastic balancing based model reduction.\n" \
    "\n" \
    "Computes a reduced order model (Ar,Br,Cr,Dr) for an original\n" \
    "state-space representation (A,B,C,D) using the stochastic balancing\n" \
    "approach in conjunction with Balance & Truncate or Singular Perturbation\n" \
    "Approximation methods for the ALPHA-stable part of the system.\n" \
    "\n" \
    "If BETA > 0, model reduction is performed on [G BETA*I] instead of G,\n" \
    "providing a balance between absolute and relative error minimization.\n" \
    "BETA = 0 is pure relative error (requires rank(D) = P).\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  job (str): 'B' square-root B&T, 'F' balancing-free B&T,\n" \
    "             'S' square-root SPA, 'P' balancing-free SPA\n" \
    "  equil (str): 'S' to equilibrate (A,B,C), 'N' for no scaling\n" \
    "  ordsel (str): 'F' for fixed order, 'A' for automatic selection\n" \
    "  n (int): Order of original system (n >= 0)\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  p (int): Number of outputs (p >= 0, p <= m if beta=0)\n" \
    "  nr (int): Desired order if ordsel='F' (0 <= nr <= n)\n" \
    "  alpha (float): Stability boundary. Continuous: alpha <= 0, Discrete: 0 <= alpha <= 1\n" \
    "  beta (float): Error weighting (beta >= 0). Large=absolute error, small=relative error\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (p x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (p x m, F-order)\n" \
    "  tol1 (float): Tolerance for order selection (used if ordsel='A')\n" \
    "  tol2 (float): Tolerance for minimal realization order\n" \
    "\n" \
    "Returns:\n" \
    "  (ar, br, cr, dr, nr, ns, hsv, iwarn, info):\n" \
    "    Reduced system matrices, actual order, dimension of stable part,\n" \
    "    Hankel singular values of phase system, warning, and exit code"

#define DOC_AB09HX "Stochastic balancing model reduction for stable systems.\n" \
    "\n" \
    "Computes a reduced order model (Ar,Br,Cr,Dr) for an original stable\n" \
    "state-space representation (A,B,C,D) using the stochastic balancing\n" \
    "approach with Balance & Truncate (B&T) or Singular Perturbation\n" \
    "Approximation (SPA) methods.\n" \
    "\n" \
    "The state dynamics matrix A must be in real Schur canonical form and\n" \
    "D must have full row rank.\n" \
    "\n" \
    "For B&T: Ar = TI * A * T, Br = TI * B, Cr = C * T\n" \
    "For SPA: Am = TI * A * T, Bm = TI * B, Cm = C * T, then SPA applied\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  job (str): 'B' square-root B&T, 'F' balancing-free B&T,\n" \
    "             'S' square-root SPA, 'P' balancing-free SPA\n" \
    "  ordsel (str): 'F' for fixed order, 'A' for automatic selection\n" \
    "  n (int): Order of original system (n >= 0)\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  p (int): Number of outputs (m >= p >= 0)\n" \
    "  nr (int): Desired order if ordsel='F' (0 <= nr <= n)\n" \
    "  a (ndarray): State matrix A in Schur form (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (p x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D with full row rank (p x m, F-order)\n" \
    "  tol1 (float): Tolerance for order selection (used if ordsel='A')\n" \
    "  tol2 (float): Tolerance for minimal realization order\n" \
    "\n" \
    "Returns:\n" \
    "  (ar, br, cr, dr, nr, hsv, t, ti, nmin, rcond, iwarn, info):\n" \
    "    - ar, br, cr, dr: Reduced system matrices\n" \
    "    - nr: Actual order of reduced model\n" \
    "    - hsv: Hankel singular values of phase system (decreasing order, <= 1)\n" \
    "    - t: N-by-NR right truncation matrix\n" \
    "    - ti: NR-by-N left truncation matrix\n" \
    "    - nmin: Order of minimal realization\n" \
    "    - rcond: Reciprocal condition number of Riccati solver\n" \
    "    - iwarn: Warning (0=ok, 1=nr > nmin)\n" \
    "    - info: Exit code (0=success, 1=A not stable/Schur, 2-5=Hamiltonian errors, 6=D rank deficient, 7=SVD failed)"

#define DOC_AB09HY "Cholesky factors of controllability and observability Grammians.\n" \
    "\n" \
    "Computes Cholesky factors Su and Ru of the controllability Grammian\n" \
    "P = Su*Su' and observability Grammian Q = Ru'*Ru satisfying:\n" \
    "  A*P + P*A' + scalec^2*B*B' = 0 (controllability Lyapunov)\n" \
    "  A'*Q + Q*A + scaleo^2*Cw'*Cw = 0 (observability Lyapunov)\n" \
    "where Cw = Hw - Bw'*X is modified output matrix.\n" \
    "\n" \
    "Matrix A must be stable (continuous-time) and in real Schur form.\n" \
    "Matrix D must have full row rank (requires m >= p).\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Order of state-space representation (n >= 0)\n" \
    "  m (int): Number of system inputs (m >= 0)\n" \
    "  p (int): Number of system outputs (0 <= p <= m)\n" \
    "  a (ndarray): N-by-N stable state matrix in real Schur form (F-order)\n" \
    "  b (ndarray): N-by-M input matrix (F-order)\n" \
    "  c (ndarray): P-by-N output matrix (F-order)\n" \
    "  d (ndarray): P-by-M feedthrough matrix with full row rank (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (s, r, scalec, scaleo, rcond, info):\n" \
    "    - s: N-by-N upper triangular Cholesky factor Su (P = Su*Su')\n" \
    "    - r: N-by-N upper triangular Cholesky factor Ru (Q = Ru'*Ru)\n" \
    "    - scalec: Scaling factor for controllability Grammian\n" \
    "    - scaleo: Scaling factor for observability Grammian\n" \
    "    - rcond: Reciprocal condition number of Riccati solver\n" \
    "    - info: Exit code (0=success, 1=A unstable, 2-5=Riccati failed, 6=D rank deficient)"

#define DOC_AB09ID "Frequency-weighted model reduction based on balancing techniques.\n" \
    "\n" \
    "Computes a reduced order model (Ar,Br,Cr,Dr) for an original\n" \
    "state-space representation (A,B,C,D) using frequency weighted\n" \
    "square-root or balancing-free square-root Balance & Truncate (B&T)\n" \
    "or Singular Perturbation Approximation (SPA) methods.\n" \
    "\n" \
    "The algorithm minimizes ||V*(G-Gr)*W|| where G and Gr are the\n" \
    "transfer-function matrices of original and reduced systems,\n" \
    "and V, W are frequency weighting transfer-function matrices.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' continuous-time, 'D' discrete-time\n" \
    "  jobc (str): 'S' standard combination, 'E' enhanced stability\n" \
    "  jobo (str): 'S' standard combination, 'E' enhanced stability\n" \
    "  job (str): 'B' sqrt B&T, 'F' balfree B&T, 'S' sqrt SPA, 'P' balfree SPA\n" \
    "  weight (str): 'N' no weight, 'L' left V, 'R' right W, 'B' both\n" \
    "  equil (str): 'S' scale (A,B,C), 'N' no scaling\n" \
    "  ordsel (str): 'F' fixed order, 'A' automatic\n" \
    "  n (int): Order of G (n >= 0)\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  p (int): Number of outputs (p >= 0)\n" \
    "  nv (int): Order of left weight V (nv >= 0)\n" \
    "  pv (int): Number of outputs of V (pv >= 0)\n" \
    "  nw (int): Order of right weight W (nw >= 0)\n" \
    "  mw (int): Number of inputs of W (mw >= 0)\n" \
    "  nr (int): Desired order if ordsel='F' (0 <= nr <= n)\n" \
    "  alpha (float): Stability boundary\n" \
    "  alphac (float): Combination parameter for controllability (|alphac|<=1)\n" \
    "  alphao (float): Combination parameter for observability (|alphao|<=1)\n" \
    "  a, b, c, d (ndarray): State-space matrices of G (F-order)\n" \
    "  av, bv, cv, dv (ndarray): State-space matrices of V (F-order)\n" \
    "  aw, bw, cw, dw (ndarray): State-space matrices of W (F-order)\n" \
    "  tol1 (float): Tolerance for order selection\n" \
    "  tol2 (float): Tolerance for minimal realization\n" \
    "\n" \
    "Returns:\n" \
    "  (ar, br, cr, dr, av, bv, cv, aw, bw, cw, ns, hsv, nr, iwarn, info):\n" \
    "    Reduced system matrices, transformed weight matrices, stable order,\n" \
    "    Hankel singular values, actual order, warning, and error code"

#define DOC_AB09IX "Accuracy enhanced balancing related model reduction.\n" \
    "\n" \
    "Computes a reduced order model (Ar,Br,Cr,Dr) for an original\n" \
    "state-space representation (A,B,C,D) using square-root or\n" \
    "balancing-free square-root Balance & Truncate (B&T) or\n" \
    "Singular Perturbation Approximation (SPA) methods.\n" \
    "\n" \
    "The computation uses Cholesky factors S and R of controllability\n" \
    "and observability Grammians. Hankel singular values are computed\n" \
    "as singular values of R*S.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  job (str): 'B' square-root B&T, 'F' balancing-free B&T,\n" \
    "             'S' square-root SPA, 'P' balancing-free SPA\n" \
    "  fact (str): 'S' if A is in Schur form, 'N' if general matrix\n" \
    "  ordsel (str): 'F' for fixed order, 'A' for automatic selection\n" \
    "  n (int): Order of original system (n >= 0)\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  p (int): Number of outputs (p >= 0)\n" \
    "  nr (int): Desired order if ordsel='F' (0 <= nr <= n)\n" \
    "  scalec (float): Scaling for controllability Cholesky factor (> 0)\n" \
    "  scaleo (float): Scaling for observability Cholesky factor (> 0)\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (p x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (p x m, F-order)\n" \
    "  ti (ndarray): Upper triangular Cholesky factor S (n x n, F-order)\n" \
    "  t (ndarray): Upper triangular Cholesky factor R (n x n, F-order)\n" \
    "  tol1 (float): Tolerance for order selection (used if ordsel='A')\n" \
    "  tol2 (float): Tolerance for minimal realization order\n" \
    "\n" \
    "Returns:\n" \
    "  (ar, br, cr, dr, ti, t, nr, nminr, hsv, iwarn, info):\n" \
    "    Reduced system matrices, truncation matrices, actual order,\n" \
    "    minimal realization order, Hankel singular values,\n" \
    "    warning indicator, and exit code"

#define DOC_AB09IY "Cholesky factors of frequency-weighted controllability/observability Grammians.\n" \
    "\n" \
    "Computes the Cholesky factors S and R of the frequency-weighted\n" \
    "controllability Grammian P = S*S' and observability Grammian Q = R'*R\n" \
    "for model reduction with frequency weighting using V*G*W.\n" \
    "\n" \
    "G, V and W must be stable with state matrices A, AV, AW in real Schur form.\n" \
    "The combination method of Enns and Lin-Chiu is used.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  jobc (str): 'S' standard combination, 'N' stability enhanced\n" \
    "  jobo (str): 'S' standard combination, 'N' stability enhanced\n" \
    "  weight (str): 'N' no weights, 'L' left V, 'R' right W, 'B' both\n" \
    "  n (int): Order of G (n >= 0)\n" \
    "  m (int): Number of G inputs (m >= 0)\n" \
    "  p (int): Number of G outputs (p >= 0)\n" \
    "  nv (int): Order of V (nv >= 0)\n" \
    "  pv (int): Number of V outputs (pv >= 0)\n" \
    "  nw (int): Order of W (nw >= 0)\n" \
    "  mw (int): Number of W inputs (mw >= 0)\n" \
    "  alphac (float): Combination parameter for controllability (0 < alphac <= 1)\n" \
    "  alphao (float): Combination parameter for observability (0 < alphao <= 1)\n" \
    "  a (ndarray): N-by-N state matrix of G in Schur form (F-order)\n" \
    "  b (ndarray): N-by-M input matrix of G (F-order)\n" \
    "  c (ndarray): P-by-N output matrix of G (F-order)\n" \
    "  av (ndarray): NV-by-NV state matrix of V (F-order)\n" \
    "  bv (ndarray): NV-by-P input matrix of V (F-order)\n" \
    "  cv (ndarray): PV-by-NV output matrix of V (F-order)\n" \
    "  dv (ndarray): PV-by-P feedthrough matrix of V (F-order)\n" \
    "  aw (ndarray): NW-by-NW state matrix of W (F-order)\n" \
    "  bw (ndarray): NW-by-MW input matrix of W (F-order)\n" \
    "  cw (ndarray): M-by-NW output matrix of W (F-order)\n" \
    "  dw (ndarray): M-by-MW feedthrough matrix of W (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (s, r, scalec, scaleo, info):\n" \
    "    - s: N-by-N upper triangular Cholesky factor (P = S*S')\n" \
    "    - r: N-by-N upper triangular Cholesky factor (Q = R'*R)\n" \
    "    - scalec: Scaling factor for controllability Grammian\n" \
    "    - scaleo: Scaling factor for observability Grammian\n" \
    "    - info: Exit code (0=success, 1=A unstable, 2=Lyapunov failed)"

#define DOC_AB09JD "Frequency-weighted Hankel-norm approximation with invertible weights.\n" \
    "\n" \
    "Computes a reduced order model (Ar,Br,Cr,Dr) using frequency weighted\n" \
    "optimal Hankel-norm approximation with invertible transfer-function\n" \
    "weights V and W.\n" \
    "\n" \
    "Minimizes ||op(V)*(G-Gr)*op(W)||_H where op(X) can be X, inv(X),\n" \
    "conj(X), or conj(inv(X)).\n" \
    "\n" \
    "Weight stability requirements:\n" \
    "- op(V)=V or op(W)=W: antistable weights\n" \
    "- op(V)=inv(V) or op(W)=inv(W): antistable zeros\n" \
    "- op(V)=conj(V) or op(W)=conj(W): stable weights\n" \
    "- op(V)=conj(inv(V)) or op(W)=conj(inv(W)): minimum-phase\n" \
    "\n" \
    "Parameters:\n" \
    "  jobv (str): 'N' V=I, 'V' op(V)=V, 'I' inv(V), 'C' conj(V), 'R' conj(inv(V))\n" \
    "  jobw (str): 'N' W=I, 'W' op(W)=W, 'I' inv(W), 'C' conj(W), 'R' conj(inv(W))\n" \
    "  jobinv (str): 'N' inverse-free, 'I' inversion-based, 'A' auto-switch\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  equil (str): 'S' to equilibrate (A,B,C), 'N' for no scaling\n" \
    "  ordsel (str): 'F' for fixed order, 'A' for automatic selection\n" \
    "  n (int): Order of G (n >= 0)\n" \
    "  nv (int): Order of left weight V (nv >= 0)\n" \
    "  nw (int): Order of right weight W (nw >= 0)\n" \
    "  m (int): Number of G inputs (m >= 0)\n" \
    "  p (int): Number of G outputs (p >= 0)\n" \
    "  nr (int): Desired reduced order if ordsel='F' (0 <= nr <= n)\n" \
    "  alpha (float): Stability boundary (continuous: alpha <= 0, discrete: 0 <= alpha <= 1)\n" \
    "  a (ndarray): N-by-N state matrix of G (F-order)\n" \
    "  b (ndarray): N-by-M input matrix of G (F-order)\n" \
    "  c (ndarray): P-by-N output matrix of G (F-order)\n" \
    "  d (ndarray): P-by-M feedthrough matrix of G (F-order)\n" \
    "  av (ndarray): NV-by-NV state matrix of V (F-order)\n" \
    "  bv (ndarray): NV-by-P input matrix of V (F-order)\n" \
    "  cv (ndarray): P-by-NV output matrix of V (F-order)\n" \
    "  dv (ndarray): P-by-P feedthrough matrix of V (F-order)\n" \
    "  aw (ndarray): NW-by-NW state matrix of W (F-order)\n" \
    "  bw (ndarray): NW-by-M input matrix of W (F-order)\n" \
    "  cw (ndarray): M-by-NW output matrix of W (F-order)\n" \
    "  dw (ndarray): M-by-M feedthrough matrix of W (F-order)\n" \
    "  tol1 (float): Tolerance for order selection (ordsel='A')\n" \
    "  tol2 (float): Tolerance for minimal realization\n" \
    "\n" \
    "Returns:\n" \
    "  (ar, br, cr, dr, av, bv, cv, aw, bw, cw, nr, ns, hsv, iwarn, info):\n" \
    "    - ar, br, cr, dr: Reduced system matrices\n" \
    "    - av, bv, cv: Transformed V matrices\n" \
    "    - aw, bw, cw: Transformed W matrices\n" \
    "    - nr: Actual reduced order\n" \
    "    - ns: Dimension of stable subsystem\n" \
    "    - hsv: Hankel singular values of stable part\n" \
    "    - iwarn: Warning (0=ok, 1=nr>nsmin, 2=nr<nu)\n" \
    "    - info: Error code (0=success, 1-21 various failures)"

#define DOC_AB09JV "Construct projection containing poles of G.\n" \
    "\n" \
    "Constructs the state-space representation for the projection of\n" \
    "V*G or conj(V)*G containing the poles of the transfer-function\n" \
    "matrix G, where V is a stable transfer-function matrix.\n" \
    "\n" \
    "Used in coprime factorization-based model reduction algorithms.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'V' for V*G, 'C' for conj(V)*G\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  jobev (str): 'G' for general EV, 'I' for identity EV\n" \
    "  stbchk (str): 'C' to check stability of AV, 'N' to skip check\n" \
    "  n (int): Order of system G (n >= 0)\n" \
    "  m (int): Number of inputs of G (m >= 0)\n" \
    "  p (int): Number of outputs of G (p >= 0)\n" \
    "  nv (int): Order of weight V (nv >= 0)\n" \
    "  pv (int): Number of rows of CV = number of inputs of V\n" \
    "  a (ndarray): State matrix A of G (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B of G (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C of G (p x n, F-order), modified on exit\n" \
    "  d (ndarray): Feedthrough matrix D of G (p x m, F-order), modified on exit\n" \
    "  av (ndarray): State matrix AV of V (nv x nv, F-order), modified on exit\n" \
    "  ev (ndarray): Descriptor matrix EV of V (nv x nv, F-order), modified on exit\n" \
    "  bv (ndarray): Input matrix BV of V (nv x pv, F-order), modified on exit\n" \
    "  cv (ndarray): Output matrix CV of V (pv x nv, F-order), modified on exit\n" \
    "  dv (ndarray): Feedthrough matrix DV of V (pv x p, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (c, d, av, ev, bv, cv, info):\n" \
    "    Modified system matrices and exit code.\n" \
    "    info=0: success\n" \
    "    info=1: AV not stable (when STBCHK='C')\n" \
    "    info=2: pencil (AV,EV) not in generalized Schur form\n" \
    "    info=3: Sylvester equation singular"

#define DOC_AB09JW "Construct projection of G*W or G*conj(W) containing poles of G.\n" \
    "\n" \
    "Constructs the state-space representation for the projection of\n" \
    "G*W or G*conj(W) containing the poles of the transfer-function\n" \
    "matrix G, where W is a right-side weight transfer-function matrix.\n" \
    "\n" \
    "Used in coprime factorization-based model reduction algorithms.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'W' for G*W, 'C' for G*conj(W)\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  jobew (str): 'G' for general EW, 'I' for identity EW\n" \
    "  stbchk (str): 'C' to check stability of AW, 'N' to skip check\n" \
    "  n (int): Order of system G (n >= 0)\n" \
    "  m (int): Number of inputs of G (m >= 0)\n" \
    "  p (int): Number of outputs of G (p >= 0)\n" \
    "  nw (int): Order of weight W (nw >= 0)\n" \
    "  mw (int): Number of inputs (job='W') or outputs (job='C') of W\n" \
    "  a (ndarray): State matrix A of G (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B of G (n x m, F-order), modified on exit\n" \
    "  c (ndarray): Output matrix C of G (p x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D of G (p x m, F-order), modified on exit\n" \
    "  aw (ndarray): State matrix AW of W (nw x nw, F-order), modified on exit\n" \
    "  ew (ndarray): Descriptor matrix EW of W (nw x nw, F-order), modified on exit\n" \
    "  bw (ndarray): Input matrix BW of W (nw x mbw, F-order), modified on exit\n" \
    "  cw (ndarray): Output matrix CW of W (pcw x nw, F-order), modified on exit\n" \
    "  dw (ndarray): Feedthrough matrix DW of W (pcw x mbw, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (b, d, aw, ew, bw, cw, info):\n" \
    "    Modified system matrices and exit code.\n" \
    "    info=0: success\n" \
    "    info=1: Schur form reduction failed\n" \
    "    info=2: Sylvester equation (generalized) singular\n" \
    "    info=3: Sylvester equation (standard) singular\n" \
    "    info=4: stability/antistability check failed"

#define DOC_AB09JX "Check stability/antistability of finite eigenvalues.\n" \
    "\n" \
    "Checks whether all finite eigenvalues lie within a specified\n" \
    "stability domain defined by DICO, STDOM, and ALPHA.\n" \
    "\n" \
    "Domain definitions:\n" \
    "- Continuous stable: Re(lambda) < ALPHA\n" \
    "- Continuous unstable: Re(lambda) > ALPHA\n" \
    "- Discrete stable: |lambda| < ALPHA\n" \
    "- Discrete unstable: |lambda| > ALPHA\n" \
    "\n" \
    "For EVTYPE='R', conditions apply to 1/lambda.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  stdom (str): 'S' for stability domain, 'U' for instability domain\n" \
    "  evtype (str): 'S' standard (ED=1), 'G' generalized, 'R' reciprocal\n" \
    "  n (int): Dimension of eigenvalue vectors (n >= 0)\n" \
    "  alpha (float): Boundary value for real parts or moduli\n" \
    "  er (ndarray): Real parts of eigenvalues, dimension (n)\n" \
    "  ei (ndarray): Imaginary parts of eigenvalues, dimension (n)\n" \
    "  ed (ndarray): Denominators for generalized eigenvalues, dimension (n)\n" \
    "  tolinf (float): Tolerance for infinite eigenvalue detection (0 <= tolinf < 1)\n" \
    "\n" \
    "Returns:\n" \
    "  info: 0 if all in domain, 1 if some outside, -i if param i invalid"

#define DOC_AB09KD "Frequency-weighted Hankel-norm approximation model reduction.\n" \
    "\n" \
    "Computes a reduced order model (Ar,Br,Cr,Dr) for an original\n" \
    "state-space representation (A,B,C,D) using frequency weighted\n" \
    "optimal Hankel-norm approximation.\n" \
    "\n" \
    "Minimizes ||V*(G-Gr)*W||_H or ||conj(V)*(G-Gr)*conj(W)||_H.\n" \
    "\n" \
    "For JOB='N': V and W must be antistable.\n" \
    "For JOB='C': V and W must be stable.\n" \
    "DV and DW must be invertible.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'N' for ||V*(G-Gr)*W||_H, 'C' for ||conj(V)*(G-Gr)*conj(W)||_H\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  weight (str): 'N' no weights, 'L' left only, 'R' right only, 'B' both\n" \
    "  equil (str): 'S' to equilibrate (A,B,C), 'N' for no scaling\n" \
    "  ordsel (str): 'F' for fixed order, 'A' for automatic selection\n" \
    "  n (int): Order of G (n >= 0)\n" \
    "  nv (int): Order of left weight V (nv >= 0)\n" \
    "  nw (int): Order of right weight W (nw >= 0)\n" \
    "  m (int): Number of G inputs (m >= 0)\n" \
    "  p (int): Number of G outputs (p >= 0)\n" \
    "  nr (int): Desired reduced order if ordsel='F' (0 <= nr <= n)\n" \
    "  alpha (float): Stability boundary (continuous: alpha <= 0, discrete: 0 <= alpha <= 1)\n" \
    "  a (ndarray): N-by-N state matrix of G (F-order)\n" \
    "  b (ndarray): N-by-M input matrix of G (F-order)\n" \
    "  c (ndarray): P-by-N output matrix of G (F-order)\n" \
    "  d (ndarray): P-by-M feedthrough matrix of G (F-order)\n" \
    "  av (ndarray): NV-by-NV state matrix of V (F-order)\n" \
    "  bv (ndarray): NV-by-P input matrix of V (F-order)\n" \
    "  cv (ndarray): P-by-NV output matrix of V (F-order)\n" \
    "  dv (ndarray): P-by-P feedthrough matrix of V (F-order)\n" \
    "  aw (ndarray): NW-by-NW state matrix of W (F-order)\n" \
    "  bw (ndarray): NW-by-M input matrix of W (F-order)\n" \
    "  cw (ndarray): M-by-NW output matrix of W (F-order)\n" \
    "  dw (ndarray): M-by-M feedthrough matrix of W (F-order)\n" \
    "  tol1 (float): Tolerance for order selection (ordsel='A')\n" \
    "  tol2 (float): Tolerance for minimal realization\n" \
    "\n" \
    "Returns:\n" \
    "  (ar, br, cr, dr, av, bv, cv, aw, bw, cw, nr, ns, hsv, iwarn, info):\n" \
    "    - ar, br, cr, dr: Reduced system matrices\n" \
    "    - av, bv, cv: Transformed V matrices (inverse of V)\n" \
    "    - aw, bw, cw: Transformed W matrices (inverse of W)\n" \
    "    - nr: Actual reduced order\n" \
    "    - ns: Dimension of stable subsystem\n" \
    "    - hsv: Hankel singular values of stable part\n" \
    "    - iwarn: Warning (0=ok, 1=nr>nsmin, 2=nr<nu)\n" \
    "    - info: Error code (0=success, 1-13 various failures)"

#define DOC_AB09KX "Stable projection of V*G*W or conj(V)*G*conj(W).\n" \
    "\n" \
    "Constructs a state-space representation (A,BS,CS,DS) of the stable\n" \
    "projection of V*G*W or conj(V)*G*conj(W) from the state-space\n" \
    "representations (A,B,C,D), (AV,BV,CV,DV), and (AW,BW,CW,DW) of\n" \
    "transfer-function matrices G, V and W respectively.\n" \
    "\n" \
    "G must be stable and A must be in real Schur form.\n" \
    "For JOB='N': V and W must be completely unstable.\n" \
    "For JOB='C': V and W must be stable.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'N' for V*G*W, 'C' for conj(V)*G*conj(W)\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  weight (str): 'N' no weights, 'L' left only, 'R' right only, 'B' both\n" \
    "  n (int): Order of G state matrix (n >= 0)\n" \
    "  nv (int): Order of V state matrix (nv >= 0)\n" \
    "  nw (int): Order of W state matrix (nw >= 0)\n" \
    "  m (int): Number of G inputs and W I/O dimension (m >= 0)\n" \
    "  p (int): Number of G outputs and V I/O dimension (p >= 0)\n" \
    "  a (ndarray): N-by-N state matrix of G in real Schur form (F-order)\n" \
    "  b (ndarray): N-by-M input matrix of G (F-order)\n" \
    "  c (ndarray): P-by-N output matrix of G (F-order)\n" \
    "  d (ndarray): P-by-M feedthrough matrix of G (F-order)\n" \
    "  av (ndarray): NV-by-NV state matrix of V (F-order)\n" \
    "  bv (ndarray): NV-by-P input matrix of V (F-order)\n" \
    "  cv (ndarray): P-by-NV output matrix of V (F-order)\n" \
    "  dv (ndarray): P-by-P feedthrough matrix of V (F-order)\n" \
    "  aw (ndarray): NW-by-NW state matrix of W (F-order)\n" \
    "  bw (ndarray): NW-by-M input matrix of W (F-order)\n" \
    "  cw (ndarray): M-by-NW output matrix of W (F-order)\n" \
    "  dw (ndarray): M-by-M feedthrough matrix of W (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (bs, cs, ds, av, bv, cv, aw, bw, cw, iwarn, info):\n" \
    "    - bs, cs, ds: Modified B, C, D matrices (stable projection)\n" \
    "    - av, bv, cv: Transformed V matrices (real Schur form)\n" \
    "    - aw, bw, cw: Transformed W matrices (real Schur form)\n" \
    "    - iwarn: Warning (0=ok, 1=AV issue, 2=AW issue, 3=both)\n" \
    "    - info: Error (0=ok, 1=AV Schur failed, 2=AW Schur failed,\n" \
    "            3=Sylvester (V) failed, 4=Sylvester (W) failed)"

#define DOC_AB09MD "Balance & Truncate model reduction for ALPHA-stable part of a system.\n" \
    "\n" \
    "Computes a reduced order model (Ar,Br,Cr) for an original\n" \
    "state-space representation (A,B,C) by using either square-root\n" \
    "or balancing-free square-root Balance & Truncate (B&T)\n" \
    "for the ALPHA-stable part of the system.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  job (str): 'B' for square-root B&T, 'N' for balancing-free B&T\n" \
    "  equil (str): 'S' to equilibrate (A,B,C), 'N' for no scaling\n" \
    "  ordsel (str): 'F' for fixed order, 'A' for automatic selection\n" \
    "  n (int): Order of original system (n >= 0)\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  p (int): Number of outputs (p >= 0)\n" \
    "  nr (int): Desired order if ordsel='F' (0 <= nr <= n)\n" \
    "  alpha (float): Stability boundary (continuous: alpha <= 0, discrete: 0 <= alpha <= 1)\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (p x n, F-order)\n" \
    "  tol (float): Tolerance for order selection (used if ordsel='A')\n" \
    "\n" \
    "Returns:\n" \
    "  (ar, br, cr, ns, hsv, nr, iwarn, info):\n" \
    "    Reduced system matrices, dimension of stable part,\n" \
    "    Hankel singular values, actual order, warning indicator, and exit code"

#define DOC_AB09ND "SPA model reduction for ALPHA-stable part of a system.\n" \
    "\n" \
    "Computes a reduced order model (Ar,Br,Cr,Dr) for an original\n" \
    "state-space representation (A,B,C,D) by using either square-root\n" \
    "or balancing-free square-root Singular Perturbation Approximation (SPA)\n" \
    "for the ALPHA-stable part of the system.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  job (str): 'B' for square-root SPA, 'N' for balancing-free SPA\n" \
    "  equil (str): 'S' to equilibrate (A,B,C), 'N' for no scaling\n" \
    "  ordsel (str): 'F' for fixed order, 'A' for automatic selection\n" \
    "  n (int): Order of original system (n >= 0)\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  p (int): Number of outputs (p >= 0)\n" \
    "  nr (int): Desired order if ordsel='F' (0 <= nr <= n)\n" \
    "  alpha (float): Stability boundary (continuous: alpha <= 0, discrete: 0 <= alpha <= 1)\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (p x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (p x m, F-order)\n" \
    "  tol1 (float): Tolerance for order selection (used if ordsel='A')\n" \
    "  tol2 (float): Tolerance for minimal realization order\n" \
    "\n" \
    "Returns:\n" \
    "  (ar, br, cr, dr, nr, ns, hsv, iwarn, info):\n" \
    "    Reduced system matrices, actual order, dimension of stable part,\n" \
    "    Hankel singular values, warning indicator, and exit code"

#define DOC_AB13AD "Compute Hankel-norm of ALPHA-stable projection.\n" \
    "\n" \
    "Computes the Hankel-norm of the ALPHA-stable projection of the\n" \
    "transfer-function matrix G of the state-space system (A,B,C).\n" \
    "\n" \
    "The routine decomposes G = G1 + G2 where G1 has only ALPHA-stable\n" \
    "poles, then returns the Hankel-norm of G1 (max Hankel singular value).\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  equil (str): 'S' to scale (A,B,C), 'N' for no scaling\n" \
    "  n (int): Order of matrix A (n >= 0)\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  p (int): Number of outputs (p >= 0)\n" \
    "  alpha (float): Stability boundary:\n" \
    "                 Continuous: alpha <= 0, eigenvalues with Re < alpha are stable\n" \
    "                 Discrete: 0 <= alpha <= 1, eigenvalues with |.| < alpha are stable\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (p x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (hankel_norm, ns, hsv, info):\n" \
    "    - hankel_norm: Hankel-norm of ALPHA-stable projection (0 if error)\n" \
    "    - ns: Dimension of ALPHA-stable subsystem\n" \
    "    - hsv: Hankel singular values of stable part (length N, leading NS used)\n" \
    "    - info: Error code (0=success, 1=Schur failed, 2=separation failed,\n" \
    "            3=marginally stable, 4=HSV computation failed)"

#define DOC_AB13BD "Compute H2 or L2 norm of a transfer-function matrix.\n" \
    "\n" \
    "Computes the H2-norm (continuous) or L2-norm (discrete) of\n" \
    "transfer-function matrix G(lambda) = C*inv(lambda*I - A)*B + D.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  jobn (str): 'H' for H-infinity Gramian approach, 'L' for L2 gain\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (p x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (p x m, F-order)\n" \
    "  tol (float): Tolerance for rank determination\n" \
    "\n" \
    "Returns:\n" \
    "  (h2norm, nq, iwarn, info): H2/L2 norm, order of minimal realization,\n" \
    "    warning indicator, exit code"

#define DOC_AB13CD "Compute H-infinity norm of continuous-time stable system.\n" \
    "\n" \
    "Computes the H-infinity norm of the continuous-time stable system\n" \
    "G(s) = C*inv(sI - A)*B + D, which is the peak gain of the frequency\n" \
    "response (largest singular value in MIMO case).\n" \
    "\n" \
    "Uses the algorithm from Bruinsma & Steinbuch (1990).\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): System order (n >= 0)\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  np (int): Number of outputs (np >= 0)\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (np x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (np x m, F-order)\n" \
    "  tol (float): Tolerance for accuracy in determining the norm\n" \
    "\n" \
    "Returns:\n" \
    "  (hnorm, fpeak, info): H-infinity norm, peak frequency, exit code\n" \
    "    info: 0=success, 1=unstable, 2=tolerance too small, 3=eigenvalue error, 4=SVD error"

#define DOC_AB13DD "Compute L-infinity norm of state-space system.\n" \
    "\n" \
    "Computes the L-infinity norm of a continuous-time or discrete-time\n" \
    "system G(lambda) = C*inv(lambda*E - A)*B + D.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  jobe (str): 'I' for E=identity, 'G' for general E\n" \
    "  equil (str): 'S' to scale system, 'N' for no scaling\n" \
    "  jobd (str): 'Z' for D=zero, 'D' for general D\n" \
    "  n (int): System order\n" \
    "  m (int): Number of inputs\n" \
    "  p (int): Number of outputs\n" \
    "  fpeak (ndarray): Frequency estimate [omega, scale]\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  e (ndarray): Descriptor matrix E (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (p x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (p x m, F-order)\n" \
    "  tol (float): Tolerance for convergence (0 <= tol < 1)\n" \
    "\n" \
    "Returns:\n" \
    "  (gpeak, fpeak, info): Peak gain [norm, scale], peak frequency, exit code"

#define DOC_AB13DX "Compute transfer function gain at specific frequency.\n" \
    "\n" \
    "Computes maximum singular value of G(lambda) = C*inv(lambda*E - A)*B + D\n" \
    "at a given frequency omega.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  jobe (str): 'I' for E=identity, 'G' for general E\n" \
    "  jobd (str): 'Z' for D=zero, 'D' for general D\n" \
    "  n (int): System order\n" \
    "  m (int): Number of inputs\n" \
    "  p (int): Number of outputs\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  e (ndarray): Descriptor matrix E (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (p x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (p x m, F-order)\n" \
    "  omega (float): Frequency value\n" \
    "\n" \
    "Returns:\n" \
    "  (result, info): Maximum singular value, exit code"

#define DOC_AB13ED "Estimate distance to instability (stability radius).\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): State matrix A (n x n)\n" \
    "  tol (float): Tolerance\n" \
    "\n" \
    "Returns:\n" \
    "  (low, high, info): Lower/Upper bounds, exit code"

#define DOC_AB13FD "Compute complex stability radius using SVD.\n" \
    "\n" \
    "Computes beta(A), the 2-norm distance from a real matrix A to the\n" \
    "nearest complex matrix with an eigenvalue on the imaginary axis.\n" \
    "beta(A) = min_w sigma_min(A - jwI)\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  tol (float): Accuracy tolerance (default 0.0)\n" \
    "\n" \
    "Returns:\n" \
    "  (beta, omega, info): Stability radius, minimizing frequency, exit code"

#define DOC_AB13HD "Compute L-infinity norm of proper continuous-time or causal discrete-time descriptor system.\n" \
    "\n" \
    "Computes the L-infinity norm of a proper and target continuous-time\n" \
    "or a causal and target discrete-time system:\n" \
    "  G(lambda) = C * inv(lambda*E - A) * B + D\n" \
    "\n" \
    "where 'proper' means system doesn't have infinite eigenvalues in E,\n" \
    "and 'causal' means system is causal in discrete-time.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  jobe (str): 'I' for E=identity, 'G' for general E\n" \
    "  equil (str): 'S' to scale system, 'N' for no scaling\n" \
    "  jobd (str): 'Z' for D=zero, 'D' for general D, 'F' for feedthrough factor form\n" \
    "  ckprop (str): 'C' to check properness, 'N' to not check\n" \
    "  reduce (str): 'R' to reduce system order, 'N' for no reduction\n" \
    "  poles (str): 'P' to output poles, 'N' to not output\n" \
    "  n (int): System order (n >= 0)\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  p (int): Number of outputs (p >= 0)\n" \
    "  ranke (int): Rank of E matrix (0 <= ranke <= n for jobe='G')\n" \
    "  fpeak (ndarray): Frequency estimate [omega, scale] (2 elements)\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  e (ndarray): Descriptor matrix E (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (p x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (p x m, F-order)\n" \
    "  tol (ndarray): Tolerances (3 elements)\n" \
    "\n" \
    "Returns:\n" \
    "  (gpeak, fpeak, nr, iwarn, info): Peak gain [norm, scale], peak frequency,\n" \
    "    reduced order, warning indicator, exit code"

#define DOC_AB13ID "Check if descriptor system transfer function is proper.\n" \
    "\n" \
    "Determines whether the transfer function G(lambda) = C*inv(lambda*E - A)*B\n" \
    "of a descriptor system (A-lambda*E, B, C) is proper. A transfer function\n" \
    "is proper if it has no poles at infinity.\n" \
    "\n" \
    "The routine optionally computes a deflated system (Ar-lambda*Er, Br, Cr)\n" \
    "with the same transfer function but with E upper triangular and nonsingular.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobsys (str): 'R' to reduce system order, 'N' for no reduction\n" \
    "  jobeig (str): 'A' remove all infinite eigenvalues, 'I' only those uncontrollable from infinity\n" \
    "  equil (str): 'S' to scale system, 'N' for no scaling\n" \
    "  cksing (str): 'C' to check singularity of E, 'N' to skip check\n" \
    "  restor (str): 'R' to restore (A,E,B,C) on exit if improper, 'N' for no restoration\n" \
    "  update (str): 'U' to update (A,E,B,C), 'N' for no update\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  e (ndarray): Descriptor matrix E (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (p x n, F-order)\n" \
    "  n (int, optional): System order (default from A shape)\n" \
    "  m (int, optional): Number of inputs (default from B shape)\n" \
    "  p (int, optional): Number of outputs (default from C shape)\n" \
    "  tol (ndarray, optional): Tolerances [tol1, tol2] (default [0.0, 0.0])\n" \
    "\n" \
    "Returns:\n" \
    "  (is_proper, nr, ranke, a, e, b, c, iwarn, info):\n" \
    "    - is_proper: True if transfer function is proper, False otherwise\n" \
    "    - nr: Order of reduced descriptor system\n" \
    "    - ranke: Rank of E (number of finite generalized eigenvalues)\n" \
    "    - a, e, b, c: Modified system matrices\n" \
    "    - iwarn: Warning (1=E nearly singular, 2=system nearly improper)\n" \
    "    - info: Exit code (0=success, 1=Schur failed, 2=QZ failed)"

#define DOC_AB13MD "Compute upper bound on structured singular value (mu).\n" \
    "\n" \
    "Computes an upper bound on the structured singular value for a given\n" \
    "square complex matrix and a given block structure of the uncertainty.\n" \
    "\n" \
    "Parameters:\n" \
    "  z (ndarray): Complex matrix Z (n x n, F-order)\n" \
    "  nblock (ndarray): Block sizes, dimension (m), int32\n" \
    "  itype (ndarray): Block types (1=real, 2=complex), dimension (m), int32\n" \
    "  fact (str, optional): 'F' to use x from previous call, 'N' fresh (default)\n" \
    "  x (ndarray, optional): Scaling vector from previous call\n" \
    "\n" \
    "Returns:\n" \
    "  (bound, d, g, x, info): Upper bound on mu, scaling matrices D and G,\n" \
    "    scaling vector X for next call, exit code"

#define DOC_AB8NXZ "Extract reduced system with same transmission zeros (complex).\n" \
    "\n" \
    "Extracts from compound system [B A; D C] a reduced system\n" \
    "[B' A'; D' C'] with same transmission zeros but D' full row rank.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Number of state variables (n >= 0)\n" \
    "  m (int): Number of system inputs (m >= 0)\n" \
    "  p (int): Number of system outputs (p >= 0)\n" \
    "  ro (int): p for original, max(p-m,0) for pertransposed system\n" \
    "  sigma (int): 0 for original, m for pertransposed system\n" \
    "  svlmax (float): Estimate of largest singular value (>= 0)\n" \
    "  abcd (ndarray): Compound matrix (n+p x m+n, F-order, complex)\n" \
    "  tol (float): Tolerance for rank decisions\n" \
    "\n" \
    "Returns:\n" \
    "  (abcd, ro, sigma, mu, nu, ninfz, infz, kronl, info): Reduced matrix,\n" \
    "    final ro/sigma, dimensions (mu,nu), infinite zero count/degrees,\n" \
    "    left Kronecker indices, exit code"

#define DOC_AG07BD "Descriptor inverse of a state-space or descriptor system.\n" \
    "\n" \
    "Computes the inverse (Ai-lambda*Ei, Bi, Ci, Di) of a given\n" \
    "descriptor system (A-lambda*E, B, C, D) using the formulas:\n" \
    "  Ei = [E 0; 0 0], Ai = [A B; C D], Bi = [0; -I], Ci = [0 I], Di = 0\n" \
    "\n" \
    "The inverse system has order N+M.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobe (str): 'G' for general E, 'I' for identity E\n" \
    "  n (int): Order of A and E matrices (n >= 0)\n" \
    "  m (int): Number of inputs/outputs (m >= 0)\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (m x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (m x m, F-order)\n" \
    "  e (ndarray, optional): Descriptor matrix E (n x n, F-order), required if jobe='G'\n" \
    "\n" \
    "Returns:\n" \
    "  (ai, ei, bi, ci, di, info):\n" \
    "    - ai: State matrix of inverse ((n+m) x (n+m))\n" \
    "    - ei: Descriptor matrix of inverse ((n+m) x (n+m))\n" \
    "    - bi: Input matrix of inverse ((n+m) x m)\n" \
    "    - ci: Output matrix of inverse (m x (n+m))\n" \
    "    - di: Feedthrough matrix of inverse (m x m, always zero)\n" \
    "    - info: Exit code (0 = success)"

#define DOC_AG08BD "Zeros and Kronecker structure of a descriptor system pencil.\n" \
    "\n" \
    "Extracts from the system pencil S(lambda) = [A-lambda*E, B; C, D]\n" \
    "a regular pencil Af-lambda*Ef with finite Smith zeros as eigenvalues.\n" \
    "Also computes infinite zero orders and Kronecker structure.\n" \
    "\n" \
    "Parameters:\n" \
    "  equil (str): 'S' to balance, 'N' no balancing\n" \
    "  l (int): Number of rows of A, B, E (l >= 0)\n" \
    "  n (int): Number of columns of A, E, C (n >= 0)\n" \
    "  m (int): Number of columns of B (m >= 0)\n" \
    "  p (int): Number of rows of C (p >= 0)\n" \
    "  a (ndarray): State matrix (l x n, F-order)\n" \
    "  e (ndarray): Descriptor matrix (l x n, F-order)\n" \
    "  b (ndarray): Input matrix (l x m, F-order)\n" \
    "  c (ndarray): Output matrix (p x n, F-order)\n" \
    "  d (ndarray): Direct transmission matrix (p x m, F-order)\n" \
    "  tol (float): Tolerance for rank decisions\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, nfz, nrank, niz, dinfz, nkror, ninfe, nkrol,\n" \
    "   infz, kronr, infe, kronl, info):\n" \
    "    - a: Contains Af in leading nfz x nfz block\n" \
    "    - e: Contains Ef in leading nfz x nfz block\n" \
    "    - nfz: Number of finite zeros\n" \
    "    - nrank: Normal rank of system pencil\n" \
    "    - niz: Number of infinite zeros\n" \
    "    - dinfz: Max multiplicity of infinite zeros\n" \
    "    - nkror: Number of right Kronecker indices\n" \
    "    - ninfe: Number of elementary infinite blocks\n" \
    "    - nkrol: Number of left Kronecker indices\n" \
    "    - infz: Infinite zero degrees\n" \
    "    - kronr: Right Kronecker indices\n" \
    "    - infe: Multiplicities of infinite eigenvalues\n" \
    "    - kronl: Left Kronecker indices\n" \
    "    - info: Exit code (0 = success)"

#define DOC_AG08BY "Extract reduced descriptor system pencil preserving finite Smith zeros.\n" \
    "\n" \
    "Extracts from the (N+P)-by-(M+N) descriptor system pencil\n" \
    "S(lambda) = [B, A-lambda*E; D, C] with E upper triangular\n" \
    "a reduced pencil Sr(lambda) = [Br, Ar-lambda*Er; Dr, Cr]\n" \
    "with the same finite Smith zeros but Dr full row rank.\n" \
    "\n" \
    "Parameters:\n" \
    "  first (bool): True if first call, False for already reduced system\n" \
    "  n (int): Order of A and E (n >= 0)\n" \
    "  m (int): Number of columns of B and D (m >= 0)\n" \
    "  p (int): Number of rows of C and D (p >= 0)\n" \
    "  svlmax (float): Max singular value estimate (>= 0)\n" \
    "  abcd (ndarray): Compound matrix [B A; D C] ((n+p) x (m+n), F-order)\n" \
    "  e (ndarray): Upper triangular E (n x n, F-order)\n" \
    "  tol (float): Tolerance for rank decisions\n" \
    "\n" \
    "Returns:\n" \
    "  (abcd, e, nr, pr, ninfz, dinfz, nkronl, infz, kronl, info):\n" \
    "    - abcd: Reduced compound matrix\n" \
    "    - e: Reduced upper triangular matrix\n" \
    "    - nr: Order of reduced system\n" \
    "    - pr: Rank of Dr\n" \
    "    - ninfz: Number of infinite zeros\n" \
    "    - dinfz: Max multiplicity of infinite zeros\n" \
    "    - nkronl: Max dimension of left Kronecker blocks\n" \
    "    - infz: Infinite zero degrees\n" \
    "    - kronl: Left Kronecker block counts\n" \
    "    - info: Exit code (0 = success)"

#define DOC_AG08BZ "Zeros and Kronecker structure of a complex descriptor system pencil.\n" \
    "\n" \
    "Complex version of ag08bd. Extracts from the system pencil\n" \
    "S(lambda) = [A-lambda*E, B; C, D] a regular pencil Af-lambda*Ef\n" \
    "with finite Smith zeros as eigenvalues. Also computes infinite\n" \
    "zero orders and Kronecker structure.\n" \
    "\n" \
    "Parameters:\n" \
    "  equil (str): 'S' to balance, 'N' no balancing\n" \
    "  l (int): Number of rows of A, B, E (l >= 0)\n" \
    "  n (int): Number of columns of A, E, C (n >= 0)\n" \
    "  m (int): Number of columns of B (m >= 0)\n" \
    "  p (int): Number of rows of C (p >= 0)\n" \
    "  a (ndarray): Complex state matrix (l x n, F-order)\n" \
    "  e (ndarray): Complex descriptor matrix (l x n, F-order)\n" \
    "  b (ndarray): Complex input matrix (l x m, F-order)\n" \
    "  c (ndarray): Complex output matrix (p x n, F-order)\n" \
    "  d (ndarray): Complex direct transmission matrix (p x m, F-order)\n" \
    "  tol (float): Tolerance for rank decisions\n" \
    "\n" \
    "Returns:\n" \
    "  (nfz, nrank, niz, dinfz, nkror, ninfe, nkrol,\n" \
    "   infz, kronr, infe, kronl, info):\n" \
    "    - nfz: Number of finite zeros\n" \
    "    - nrank: Normal rank of system pencil\n" \
    "    - niz: Number of infinite zeros\n" \
    "    - dinfz: Max multiplicity of infinite zeros\n" \
    "    - nkror: Number of right Kronecker indices\n" \
    "    - ninfe: Number of elementary infinite blocks\n" \
    "    - nkrol: Number of left Kronecker indices\n" \
    "    - infz: Infinite zero degrees\n" \
    "    - kronr: Right Kronecker indices\n" \
    "    - infe: Multiplicities of infinite eigenvalues\n" \
    "    - kronl: Left Kronecker indices\n" \
    "    - info: Exit code (0 = success)"

#define DOC_AG8BYZ "Extract reduced descriptor system pencil preserving finite Smith zeros (complex).\n" \
    "\n" \
    "Complex version of ag08by. Extracts from the (N+P)-by-(M+N) complex\n" \
    "descriptor system pencil S(lambda) = [B, A-lambda*E; D, C] with E\n" \
    "upper triangular a reduced pencil Sr(lambda) = [Br, Ar-lambda*Er; Dr, Cr]\n" \
    "with the same finite Smith zeros but Dr full row rank.\n" \
    "\n" \
    "Parameters:\n" \
    "  first (bool): True if first call, False for already reduced system\n" \
    "  n (int): Order of A and E (n >= 0)\n" \
    "  m (int): Number of columns of B and D (m >= 0)\n" \
    "  p (int): Number of rows of C and D (p >= 0)\n" \
    "  svlmax (float): Max singular value estimate (>= 0)\n" \
    "  abcd (ndarray): Complex compound matrix [B A; D C] ((n+p) x (m+n), F-order)\n" \
    "  e (ndarray): Complex upper triangular E (n x n, F-order)\n" \
    "  tol (float): Tolerance for rank decisions\n" \
    "\n" \
    "Returns:\n" \
    "  (abcd, e, nr, pr, ninfz, dinfz, nkronl, infz, kronl, info):\n" \
    "    - abcd: Reduced compound matrix (complex)\n" \
    "    - e: Reduced upper triangular matrix (complex)\n" \
    "    - nr: Order of reduced system\n" \
    "    - pr: Rank of Dr\n" \
    "    - ninfz: Number of infinite zeros\n" \
    "    - dinfz: Max multiplicity of infinite zeros\n" \
    "    - nkronl: Max dimension of left Kronecker blocks\n" \
    "    - infz: Infinite zero degrees\n" \
    "    - kronl: Left Kronecker block counts\n" \
    "    - info: Exit code (0 = success)"

#define DOC_BB01AD "Generate benchmark examples for continuous-time algebraic Riccati equations.\n" \
    "\n" \
    "Generates benchmark examples for the numerical solution of CAREs:\n" \
    "  0 = Q + A'X + XA - XGX\n" \
    "corresponding to the Hamiltonian matrix H = [A G; Q -A'].\n" \
    "\n" \
    "The examples come from the CAREX collection (Kenney/Laub/Wette 1989).\n" \
    "\n" \
    "Parameters:\n" \
    "  def (str): 'D' use default parameters, 'N' use provided parameters\n" \
    "  nr (list/ndarray): Example identifier [group, number] (int32)\n" \
    "  dpar (ndarray): Real parameters, dimension 7 (F-order, float64)\n" \
    "  ipar (ndarray): Integer parameters [n, m, p, ...], dimension 4 (int32)\n" \
    "  bpar (ndarray): Boolean parameters, dimension 6 (bool):\n" \
    "    bpar[0]: True=compute G, False=return B,R factors\n" \
    "    bpar[1]: True=G full storage, False=packed\n" \
    "    bpar[2]: True=G upper packed, False=lower\n" \
    "    bpar[3]: True=compute Q, False=return C,W factors\n" \
    "    bpar[4]: True=Q full storage, False=packed\n" \
    "    bpar[5]: True=Q upper packed, False=lower\n" \
    "\n" \
    "Returns:\n" \
    "  (vec, n, m, p, a, b, c, g, q, x, info):\n" \
    "    - vec: Boolean flags (9 elements)\n" \
    "    - n, m, p: Problem dimensions\n" \
    "    - a: State matrix (n x n)\n" \
    "    - b: Input matrix (n x m)\n" \
    "    - c: Output matrix (p x n)\n" \
    "    - g: G matrix or packed form\n" \
    "    - q: Q matrix or packed form\n" \
    "    - x: Solution matrix (if vec[8]=True)\n" \
    "    - info: 0=success, 1=not implemented, 2=invalid param, 3=singular R"

#define DOC_BB02AD "Generate benchmark examples for discrete-time algebraic Riccati equations.\n" \
    "\n" \
    "Generates benchmark examples for the numerical solution of DAREs:\n" \
    "  0 = A'XA - X - (A'XB + S)(R + B'XB)^{-1}(B'XA + S') + Q\n" \
    "\n" \
    "The examples come from the DAREX collection (Benner/Laub/Mehrmann 1995).\n" \
    "\n" \
    "Parameters:\n" \
    "  def (str): 'D' use default parameters, 'N' use provided parameters\n" \
    "  nr (list/ndarray): Example identifier [group, number] (int32)\n" \
    "  dpar (ndarray): Real parameters, dimension 4 (F-order, float64)\n" \
    "  ipar (ndarray): Integer parameters [n, m, p], dimension 3 (int32)\n" \
    "  bpar (ndarray): Boolean parameters, dimension 7 (bool):\n" \
    "    bpar[0]: True=compute Q, False=return C,Q0 factors\n" \
    "    bpar[1]: True=Q full storage, False=packed\n" \
    "    bpar[2]: True=Q upper packed, False=lower\n" \
    "    bpar[3]: True=compute G, False=return B,R factors\n" \
    "    bpar[4]: True=R/G full storage, False=packed\n" \
    "    bpar[5]: True=R/G upper packed, False=lower\n" \
    "    bpar[6]: True=return S matrix, False=don't return S\n" \
    "\n" \
    "Returns:\n" \
    "  (vec, n, m, p, a, b, c, q, r, s, x, info):\n" \
    "    - vec: Boolean flags (10 elements)\n" \
    "    - n, m, p: Problem dimensions\n" \
    "    - a: State matrix (n x n)\n" \
    "    - b: Input matrix (n x m)\n" \
    "    - c: Output matrix (p x n)\n" \
    "    - q: Q matrix or packed form\n" \
    "    - r: R matrix or G if bpar[3]=True\n" \
    "    - s: Coefficient matrix S (if bpar[6]=True)\n" \
    "    - x: Solution matrix (if vec[9]=True)\n" \
    "    - info: 0=success, 1=not implemented, 2=div by zero, 3=singular R"

#define DOC_BB03AD "Generate benchmark examples for continuous-time Lyapunov equations.\n" \
    "\n" \
    "Generates benchmark examples of (generalized) continuous-time Lyapunov equations:\n" \
    "  A'XE + E'XA = Y\n" \
    "\n" \
    "In some examples, Y = -B'B and X = U'U (Cholesky factor).\n" \
    "Implements the CTLEX collection (Kressner/Mehrmann/Penzl 1999).\n" \
    "\n" \
    "Parameters:\n" \
    "  def (str): 'D' use default parameters, 'N' use provided parameters\n" \
    "  nr (list/ndarray): Example identifier [group, number] (int32)\n" \
    "    Only group 4 (scalable parameter-dependent) is supported:\n" \
    "    nr[1]=1: Example 4.1 (r, s parameters)\n" \
    "    nr[1]=2: Example 4.2 (lambda, s parameters)\n" \
    "    nr[1]=3: Example 4.3 (generalized, t parameter)\n" \
    "    nr[1]=4: Example 4.4 (generalized, t parameter, n=3*q)\n" \
    "  dpar (ndarray): Real parameters, dimension 2 (float64)\n" \
    "  ipar (ndarray): Integer parameters [n or q], dimension 1 (int32)\n" \
    "\n" \
    "Returns:\n" \
    "  (vec, n, m, e, a, y, b, x, u, note, info):\n" \
    "    - vec: Boolean flags (8 elements):\n" \
    "      vec[0-1]: N, M always available\n" \
    "      vec[2]: True if E is NOT identity\n" \
    "      vec[3-4]: A, Y always available\n" \
    "      vec[5]: True if B provided\n" \
    "      vec[6]: True if X (solution) provided\n" \
    "      vec[7]: True if U (Cholesky) provided\n" \
    "    - n: Problem dimension\n" \
    "    - m: Rows in B (0 if not provided)\n" \
    "    - e, a, y: Equation matrices (n x n)\n" \
    "    - b: RHS factor (m x n if provided)\n" \
    "    - x: Solution (n x n if provided)\n" \
    "    - u: Cholesky factor (n x n if provided)\n" \
    "    - note: Example description string\n" \
    "    - info: 0=success, <0=invalid argument"

#define DOC_BB04AD "Generate benchmark examples for discrete-time Lyapunov equations.\n" \
    "\n" \
    "Generates benchmark examples of (generalized) discrete-time Lyapunov equations:\n" \
    "  A'XA - E'XE = Y\n" \
    "\n" \
    "In some examples, Y = -B'B and X = U'U (Cholesky factor).\n" \
    "Implements the DTLEX collection (Kressner/Mehrmann/Penzl 1999).\n" \
    "\n" \
    "Parameters:\n" \
    "  def (str): 'D' use default parameters, 'N' use provided parameters\n" \
    "  nr (list/ndarray): Example identifier [group, number] (int32)\n" \
    "    Only group 4 (scalable parameter-dependent) is supported:\n" \
    "    nr[1]=1: Example 4.1 (r, s parameters, both > 1)\n" \
    "    nr[1]=2: Example 4.2 (lambda in (-1,1), s > 1)\n" \
    "    nr[1]=3: Example 4.3 (generalized, t >= 0)\n" \
    "    nr[1]=4: Example 4.4 (generalized, t >= 1, n=3*q)\n" \
    "  dpar (ndarray): Real parameters, dimension 2 (float64)\n" \
    "  ipar (ndarray): Integer parameters [n or q], dimension 1 (int32)\n" \
    "\n" \
    "Returns:\n" \
    "  (vec, n, m, e, a, y, b, x, u, note, info):\n" \
    "    - vec: Boolean flags (8 elements):\n" \
    "      vec[0-1]: N, M always available\n" \
    "      vec[2]: True if E is NOT identity\n" \
    "      vec[3-4]: A, Y always available\n" \
    "      vec[5]: True if B provided\n" \
    "      vec[6]: True if X (solution) provided\n" \
    "      vec[7]: True if U (Cholesky) provided\n" \
    "    - n: Problem dimension\n" \
    "    - m: Rows in B (0 if not provided)\n" \
    "    - e, a, y: Equation matrices (n x n)\n" \
    "    - b: RHS factor (m x n if provided)\n" \
    "    - x: Solution (n x n if provided)\n" \
    "    - u: Cholesky factor (n x n if provided)\n" \
    "    - note: Example description string\n" \
    "    - info: 0=success, <0=invalid argument"

#define DOC_BD01AD "Generate benchmark examples for continuous-time dynamical systems.\n" \
    "\n" \
    "Generates benchmark examples for time-invariant, continuous-time\n" \
    "dynamical systems (CTDSX collection from SLICOT Working Note 1998-9):\n" \
    "  E x'(t) = A x(t) + B u(t)\n" \
    "    y(t)  = C x(t) + D u(t)\n" \
    "\n" \
    "Parameters:\n" \
    "  def (str): 'D' use default parameters, 'N' use provided parameters\n" \
    "  nr (list/ndarray): Example identifier [group, number] (int32)\n" \
    "    Group 1: Parameter-free problems of fixed size\n" \
    "    Group 2: Parameter-dependent problems of fixed size\n" \
    "    Group 3: Parameter-free problems of scalable size\n" \
    "    Group 4: Parameter-dependent problems of scalable size\n" \
    "  dpar (ndarray): Real parameters, dimension 7 (float64)\n" \
    "  ipar (ndarray): Integer parameters, dimension 1 (int32)\n" \
    "\n" \
    "Returns:\n" \
    "  (vec, n, m, p, e, a, b, c, d, note, info):\n" \
    "    - vec: Boolean flags (8 elements):\n" \
    "      vec[0-2]: N, M, P always available\n" \
    "      vec[3]: True if E is NOT identity\n" \
    "      vec[4-6]: A, B, C always available\n" \
    "      vec[7]: True if D is NOT zero\n" \
    "    - n, m, p: Problem dimensions (states, inputs, outputs)\n" \
    "    - e: Descriptor matrix (n x n), identity if vec[3]=False\n" \
    "    - a: State matrix (n x n)\n" \
    "    - b: Input matrix (n x m)\n" \
    "    - c: Output matrix (p x n)\n" \
    "    - d: Feedthrough matrix (p x m), zero if vec[7]=False\n" \
    "    - note: Example description string\n" \
    "    - info: 0=success, 1=data file required, <0=invalid argument"

#define DOC_BD02AD "Generate benchmark examples for discrete-time dynamical systems.\n" \
    "\n" \
    "Generates benchmark examples for time-invariant, discrete-time\n" \
    "dynamical systems (DTDSX collection from SLICOT Working Note 1998-10):\n" \
    "  E x_{k+1} = A x_k + B u_k\n" \
    "        y_k = C x_k + D u_k\n" \
    "\n" \
    "Parameters:\n" \
    "  def (str): 'D' use default parameters, 'N' use provided parameters\n" \
    "  nr (list/ndarray): Example identifier [group, number] (int32)\n" \
    "    Group 1: Parameter-free problems of fixed size\n" \
    "    Group 2: Parameter-dependent problems of fixed size\n" \
    "    Group 3: Parameter-free problems of scalable size\n" \
    "    Group 4: Parameter-dependent problems of scalable size\n" \
    "  dpar (ndarray): Real parameters, dimension 7 (float64)\n" \
    "  ipar (ndarray): Integer parameters, dimension 1 (int32)\n" \
    "\n" \
    "Returns:\n" \
    "  (vec, n, m, p, e, a, b, c, d, note, info):\n" \
    "    - vec: Boolean flags (8 elements):\n" \
    "      vec[0-2]: N, M, P always available\n" \
    "      vec[3]: True if E is NOT identity\n" \
    "      vec[4-6]: A, B, C always available\n" \
    "      vec[7]: True if D is NOT zero\n" \
    "    - n, m, p: Problem dimensions (states, inputs, outputs)\n" \
    "    - e: Descriptor matrix (n x n), identity if vec[3]=False\n" \
    "    - a: State matrix (n x n)\n" \
    "    - b: Input matrix (n x m)\n" \
    "    - c: Output matrix (p x n)\n" \
    "    - d: Feedthrough matrix (p x m), zero if vec[7]=False\n" \
    "    - note: Example description string\n" \
    "    - info: 0=success, 1=data file required, <0=invalid argument"

#define DOC_DE01OD "Compute convolution or deconvolution of two real signals.\n" \
    "\n" \
    "Parameters:\n" \
    "  conv (str): 'C'=convolution, 'D'=deconvolution\n" \
    "  a (ndarray): First signal (N samples, must be power of 2)\n" \
    "  b (ndarray): Second signal (N samples, overwritten)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, info): Result signal and exit code"

#define DOC_DE01PD "Compute convolution or deconvolution using Hartley transform.\n" \
    "\n" \
    "Computes convolution or deconvolution of two real signals A and B using\n" \
    "three scrambled Hartley transforms (DG01OD). O(N*log(N)) complexity.\n" \
    "\n" \
    "Parameters:\n" \
    "  conv (str): 'C'=convolution, 'D'=deconvolution\n" \
    "  wght (str): 'A'=weights available, 'N'=not available\n" \
    "  a (ndarray): First signal (N samples, power of 2, modified in-place)\n" \
    "  b (ndarray): Second signal (N samples, overwritten)\n" \
    "  w (ndarray): Weight vector (N - log2(N) elements for N > 1)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, w, info): Result signal, weight vector, and exit code"

#define DOC_DF01MD "Compute sine or cosine transform of a real signal.\n" \
    "\n" \
    "Transforms a real signal using FFT-based sine or cosine transform.\n" \
    "N must be a power of 2 plus 1 (e.g., 5, 9, 17, 33, 65, 129).\n" \
    "\n" \
    "Parameters:\n" \
    "  sico (str): 'S'=sine transform, 'C'=cosine transform\n" \
    "  dt (float): Sampling time of the signal\n" \
    "  a (ndarray): Signal array (N samples, N = 2^k + 1)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, info): Transform coefficients and exit code\n" \
    "\n" \
    "Notes:\n" \
    "  For sine transform: first and last coefficients are always 0\n" \
    "  For cosine transform: coefficients scaled by sampling time dt"

#define DOC_DG01MD "Compute discrete Fourier transform or inverse of complex signal.\n" \
    "\n" \
    "Parameters:\n" \
    "  indi (str): 'D'=forward FFT, 'I'=inverse FFT\n" \
    "  xr (ndarray): Real part of signal (N samples, must be power of 2)\n" \
    "  xi (ndarray): Imaginary part of signal (N samples)\n" \
    "\n" \
    "Returns:\n" \
    "  (xr, xi, info): Transformed signal (real, imag) and exit code"

#define DOC_DG01ND "Compute discrete Fourier transform or inverse of real signal.\n" \
    "\n" \
    "Transforms a real signal of 2*N samples, represented as odd/even parts\n" \
    "(for forward transform) or N+1 complex frequency components (for inverse).\n" \
    "\n" \
    "Parameters:\n" \
    "  indi (str): 'D'=forward FFT, 'I'=inverse FFT\n" \
    "  xr (ndarray): Odd samples (forward) or real part (inverse), N samples\n" \
    "  xi (ndarray): Even samples (forward) or imag part (inverse), N samples\n" \
    "\n" \
    "Returns:\n" \
    "  (xr, xi, info): N+1 (forward) or N (inverse) components, and exit code"

#define DOC_DG01OD "Compute (scrambled) discrete Hartley transform of real signal.\n" \
    "\n" \
    "The Hartley transform is self-inverse: applying twice returns N * original.\n" \
    "N must be a power of 2.\n" \
    "\n" \
    "Parameters:\n" \
    "  scr (str): Scrambling mode: 'N'=none, 'I'=input bit-reversed, 'O'=output bit-reversed\n" \
    "  wght (str): Weight availability: 'A'=available, 'N'=not available\n" \
    "  a (ndarray): Signal array (N samples, power of 2)\n" \
    "  w (ndarray): Weight vector (N - log2(N) elements)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, w, info): Transformed signal, weights (for reuse), and exit code"

#define DOC_DGEGV "Compute generalized eigenvalues and eigenvectors.\n" \
    "\n" \
    "This routine is deprecated and replaced by DGGEV.\n" \
    "\n" \
    "Computes eigenvalues and optionally left/right eigenvectors of\n" \
    "a real matrix pair (A,B). Given A*x = lambda*B*x, returns\n" \
    "alpha and beta such that lambda = alpha/beta.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobvl (str): 'N' no left eigenvectors, 'V' compute left eigenvectors\n" \
    "  jobvr (str): 'N' no right eigenvectors, 'V' compute right eigenvectors\n" \
    "  a (ndarray): Matrix A (n x n, F-order), overwritten on exit\n" \
    "  b (ndarray): Matrix B (n x n, F-order), overwritten on exit\n" \
    "  lwork (int, optional): Workspace size (-1 for query, default auto)\n" \
    "\n" \
    "Returns:\n" \
    "  (alphar, alphai, beta, vl, vr, info):\n" \
    "    alphar: Real parts of eigenvalue numerators (n,)\n" \
    "    alphai: Imaginary parts of eigenvalue numerators (n,)\n" \
    "    beta: Denominators of eigenvalues (n,)\n" \
    "    vl: Left eigenvectors (n x n) if jobvl='V', else empty\n" \
    "    vr: Right eigenvectors (n x n) if jobvr='V', else empty\n" \
    "    info: Exit code (0=success)"

#define DOC_DK01MD "Apply anti-aliasing window to a real signal.\n" \
    "\n" \
    "Parameters:\n" \
    "  type (str): 'M'=Hamming, 'N'=Hann, 'Q'=Quadratic\n" \
    "  a (ndarray): Signal array (N samples)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, info): Windowed signal and exit code"

#define DOC_DLATZM "Apply Householder matrix generated by DTZRQF.\n" \
    "\n" \
    "This routine is deprecated and replaced by DORMRZ. It applies\n" \
    "a Householder matrix P = I - tau*u*u^T to a matrix C.\n" \
    "\n" \
    "If SIDE = 'L', C = [C1; C2] is overwritten by P*C.\n" \
    "If SIDE = 'R', C = [C1, C2] is overwritten by C*P.\n" \
    "\n" \
    "Parameters:\n" \
    "  side (str): 'L' for left multiplication, 'R' for right\n" \
    "  m (int): Number of rows of C\n" \
    "  n (int): Number of columns of C\n" \
    "  v (ndarray): Vector in representation of P\n" \
    "  incv (int): Increment between elements of v (incv != 0)\n" \
    "  tau (float): Scalar in representation of P\n" \
    "  c1 (ndarray): First row/column of C (modified in-place)\n" \
    "  c2 (ndarray): Remaining rows/columns of C (modified in-place)\n" \
    "\n" \
    "Returns:\n" \
    "  (c1, c2): Modified matrices"

#define DOC_FB01QD "Time-varying square root covariance Kalman filter (dense matrices).\n" \
    "\n" \
    "Calculates a combined measurement and time update of one iteration\n" \
    "of the time-varying Kalman filter using square root covariance filter\n" \
    "with dense matrices. Performs one recursion of the square root covariance\n" \
    "filter algorithm using Householder transformations.\n" \
    "\n" \
    "The algorithm triangularizes:\n" \
    "  | R^{1/2}_i   C*S_{i-1}    0       |     | (RINOV)^{1/2}_i  0    0 |\n" \
    "  |                                  | T = |                        |\n" \
    "  | 0          A*S_{i-1}  B*Q^{1/2}  |     |     AK_i        S_i  0 |\n" \
    "\n" \
    "The state covariance is P_{i|i-1} = S_i * S_i' (left Cholesky factor).\n" \
    "The Kalman gain is K_i = AK_i * (RINOV_i)^{-1/2}.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobk (str): 'K' = compute Kalman gain K_i, 'N' = K not required (returns AK_i)\n" \
    "  multbq (str): 'P' = B contains B*Q^{1/2}, Q not used; 'N' = B and Q separate\n" \
    "  s (ndarray): N x N lower triangular S_{i-1} on entry, S_i on exit (F-order)\n" \
    "  a (ndarray): N x N state transition matrix A_i (F-order)\n" \
    "  b (ndarray): N x M input matrix B_i or B_i*Q^{1/2} if MULTBQ='P' (F-order)\n" \
    "  q (ndarray): M x M lower triangular Q^{1/2}_i if MULTBQ='N' (F-order)\n" \
    "  c (ndarray): P x N output matrix C_i (F-order)\n" \
    "  r (ndarray): P x P lower triangular R^{1/2}_i, returns (RINOV)^{1/2}_i (F-order)\n" \
    "  tol (float, optional): Tolerance for singularity test (default 0.0 = P*P*EPS)\n" \
    "\n" \
    "Returns:\n" \
    "  (s, k, r, rcond, info):\n" \
    "    s: Updated state covariance square root S_i (lower triangular)\n" \
    "    k: Kalman gain K_i (N x P) if JOBK='K', else AK_i\n" \
    "    r: Lower triangular (RINOV)^{1/2}_i\n" \
    "    rcond: Reciprocal condition number of (RINOV)^{1/2}_i (if JOBK='K')\n" \
    "    info: Exit code (0=success, 1=singular matrix, <0=parameter error)"

#define DOC_FB01RD "Time-invariant square root covariance Kalman filter (observer Hessenberg form).\n" \
    "\n" \
    "Calculates one iteration of the Kalman filter in square root covariance form\n" \
    "using the condensed observer Hessenberg form. Exploits sparsity of the\n" \
    "observer Hessenberg form for efficiency.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobk (str): 'K' = compute Kalman gain K_i, 'N' = not required\n" \
    "  multbq (str): 'P' = B contains B*Q^{1/2}, Q not used; 'N' = separate\n" \
    "  s (ndarray): N x N lower triangular S_{i-1} on entry, S_i on exit (F-order)\n" \
    "  a (ndarray): N x N state matrix in observer Hessenberg form (F-order)\n" \
    "  b (ndarray): N x M input matrix or B*Q^{1/2} (F-order)\n" \
    "  q (ndarray): M x M lower triangular Q^{1/2} if MULTBQ='N' (F-order)\n" \
    "  c (ndarray): P x N output matrix in observer Hessenberg form (F-order)\n" \
    "  r (ndarray): P x P lower triangular R^{1/2}, returns (RINOV)^{1/2} (F-order)\n" \
    "  tol (float): Singularity tolerance for gain computation\n" \
    "\n" \
    "Returns:\n" \
    "  (s, r, k, info): Updated S, updated R, gain K, exit code"

#define DOC_FB01SD "Time-varying square root information Kalman filter (dense matrices).\n" \
    "\n" \
    "Calculates a combined measurement and time update of one iteration\n" \
    "of the time-varying Kalman filter using square root information filter\n" \
    "with dense matrices. Performs one recursion of the square root information\n" \
    "filter algorithm using Householder transformations.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobx (str): 'X' = compute filtered state X_{i+1}, 'N' = not required\n" \
    "  multab (str): 'P' = AINV, B contain A^{-1}, A^{-1}B; 'N' = AINV, B contain A^{-1}, B\n" \
    "  multrc (str): 'P' = C contains R^{-1/2}C, RINV not used; 'N' = RINV, C separate\n" \
    "  sinv (ndarray): N x N upper triangular S_i^{-1} on entry, S_{i+1}^{-1} on exit (F-order)\n" \
    "  ainv (ndarray): N x N matrix A_i^{-1} (F-order)\n" \
    "  b (ndarray): N x M input matrix B_i or A_i^{-1}B_i if MULTAB='P' (F-order)\n" \
    "  rinv (ndarray): P x P upper triangular R_{i+1}^{-1/2} if MULTRC='N' (F-order)\n" \
    "  c (ndarray): P x N output matrix C_{i+1} or product if MULTRC='P' (F-order)\n" \
    "  qinv (ndarray): M x M upper triangular Q_i^{-1/2}, updated on exit (F-order)\n" \
    "  x (ndarray): Filtered state X_i on entry, X_{i+1} on exit if JOBX='X' (length N)\n" \
    "  rinvy (ndarray): Product R_{i+1}^{-1/2}Y_{i+1} (length P)\n" \
    "  z (ndarray): Process noise mean Z_i (length M)\n" \
    "  e (ndarray): Output error estimate E_{i+1} (length P)\n" \
    "  tol (float): Singularity tolerance for X computation (only if JOBX='X')\n" \
    "\n" \
    "Returns:\n" \
    "  (sinv, qinv, x, e, info):\n" \
    "    sinv: Updated information square root\n" \
    "    qinv: Updated process noise innovation square root\n" \
    "    x: Updated/transformed state estimate\n" \
    "    e: Estimated error\n" \
    "    info: Exit code (0=success, 1=singular matrix, <0=parameter error)"

#define DOC_FB01TD "Time-invariant square root information Kalman filter (controller Hessenberg form).\n" \
    "\n" \
    "Calculates a combined measurement and time update of one iteration\n" \
    "of the time-invariant Kalman filter using square root information filter\n" \
    "with condensed controller Hessenberg form. Performs one recursion using\n" \
    "Householder transformations, exploiting the sparsity of the Hessenberg form.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobx (str): 'X' = compute filtered state X_{i+1}, 'N' = not required\n" \
    "  multrc (str): 'P' = C contains R^{-1/2}C, RINV not used; 'N' = RINV, C separate\n" \
    "  sinv (ndarray): N x N upper triangular S_i^{-1} on entry, S_{i+1}^{-1} on exit (F-order)\n" \
    "  ainv (ndarray): N x N matrix A_i^{-1} in controller Hessenberg form (F-order)\n" \
    "  ainvb (ndarray): N x M product A_i^{-1}B in controller Hessenberg form (F-order)\n" \
    "  rinv (ndarray): P x P upper triangular R_{i+1}^{-1/2} if MULTRC='N' (F-order)\n" \
    "  c (ndarray): P x N output matrix C_{i+1} or product if MULTRC='P' (F-order)\n" \
    "  qinv (ndarray): M x M upper triangular Q_i^{-1/2}, updated on exit (F-order)\n" \
    "  x (ndarray): Filtered state X_i on entry, X_{i+1} on exit if JOBX='X' (length N)\n" \
    "  rinvy (ndarray): Product R_{i+1}^{-1/2}Y_{i+1} (length P)\n" \
    "  z (ndarray): Process noise mean Z_i (length M)\n" \
    "  e (ndarray): Output error estimate E_{i+1} (length P)\n" \
    "  tol (float): Singularity tolerance for X computation (only if JOBX='X')\n" \
    "\n" \
    "Returns:\n" \
    "  (sinv, qinv, x, e, info):\n" \
    "    sinv: Updated information square root\n" \
    "    qinv: Updated process noise innovation square root (QINOV)^{-1/2}\n" \
    "    x: Updated/transformed state estimate\n" \
    "    e: Estimated error\n" \
    "    info: Exit code (0=success, 1=singular matrix, <0=parameter error)"

#define DOC_FB01VD "One recursion of the conventional Kalman filter.\n" \
    "\n" \
    "Computes one update of the Riccati difference equation and the Kalman\n" \
    "filter gain. The conventional Kalman filter gain is:\n" \
    "  K_i = P_{i|i-1} * C_i' * RINOV_i^{-1}\n" \
    "where RINOV_i = C_i * P_{i|i-1} * C_i' + R_i.\n" \
    "\n" \
    "The state covariance matrix is updated by:\n" \
    "  P_{i+1|i} = A_i * (P_{i|i-1} - K_i * C_i * P_{i|i-1}) * A_i' + B_i * Q_i * B_i'\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): State dimension (n >= 0)\n" \
    "  m (int): Input dimension (m >= 0)\n" \
    "  l (int): Output dimension (l >= 0)\n" \
    "  p (ndarray): N x N state covariance P_{i|i-1} (upper tri, F-order), updated to P_{i+1|i}\n" \
    "  a (ndarray): N x N state transition matrix A_i (F-order)\n" \
    "  b (ndarray): N x M input weight matrix B_i (F-order)\n" \
    "  c (ndarray): L x N output weight matrix C_i (F-order)\n" \
    "  q (ndarray): M x M process noise covariance Q_i (F-order)\n" \
    "  r (ndarray): L x L measurement noise covariance R_i (F-order), updated to Cholesky(RINOV)\n" \
    "  tol (float, optional): Tolerance for singularity test (default 0.0 = L*L*EPS)\n" \
    "\n" \
    "Returns:\n" \
    "  (p, k, r, rcond, info):\n" \
    "    p: Updated state covariance P_{i+1|i} (upper triangular)\n" \
    "    k: Kalman filter gain K_i (N x L)\n" \
    "    r: Upper triangular Cholesky factor of RINOV_i\n" \
    "    rcond: Reciprocal condition number of RINOV_i\n" \
    "    info: Exit code (0=success, 1-L=Cholesky failed, L+1=RINOV singular)"

#define DOC_FD01AD "Fast recursive least-squares filtering.\n" \
    "\n" \
    "Solves the least-squares filtering problem recursively in time.\n" \
    "Each call implements one time update of the solution using a\n" \
    "fast QR-decomposition based approach.\n" \
    "\n" \
    "The algorithm minimizes an exponentially weighted sum of output errors:\n" \
    "  sum_{k=1}^{n} [LAMBDA^{2(n-k)} * EOUT(k)^2]\n" \
    "\n" \
    "where EOUT(n) = YIN(n) - sum_{i=0}^{L-1} h_i * XIN(n-i)\n" \
    "\n" \
    "Parameters:\n" \
    "  jp (str): 'B' = both prediction and filtering, 'P' = prediction only\n" \
    "  l (int): Length of impulse response (l >= 1)\n" \
    "  lambda (float): Square root of forgetting factor (0 < lambda <= 1)\n" \
    "  xin (float): Input sample at instant n\n" \
    "  yin (float): Reference sample at instant n (only if jp='B')\n" \
    "  efor (float): Square root of prediction error energy at instant (n-1)\n" \
    "  xf (ndarray): Transformed forward prediction variables (length l)\n" \
    "  epsbck (ndarray): Backward prediction residuals + conversion factor (length l+1)\n" \
    "  cteta (ndarray): Cosines of rotation angles (length l)\n" \
    "  steta (ndarray): Sines of rotation angles (length l)\n" \
    "  yq (ndarray): Transformed reference vector / tap multipliers (length l, jp='B' only)\n" \
    "\n" \
    "Returns:\n" \
    "  (xf, epsbck, cteta, steta, yq, efor, epos, salph, eout, iwarn, info):\n" \
    "    xf: Updated forward prediction variables\n" \
    "    epsbck: Updated backward prediction residuals\n" \
    "    cteta: Updated cosines of rotation angles\n" \
    "    steta: Updated sines of rotation angles\n" \
    "    yq: Updated reference vector (jp='B' only)\n" \
    "    efor: Updated prediction error energy square root\n" \
    "    epos: A posteriori forward prediction error residual\n" \
    "    salph: Opposite of reflection coefficients (length l)\n" \
    "    eout: A posteriori output error residual (jp='B' only)\n" \
    "    iwarn: Warning (0=ok, 1=numerical precision issue)\n" \
    "    info: Exit code (0=success, <0=parameter error)"

#define DOC_IB01AD "System identification driver - MOESP/N4SID preprocessing and order estimation.\n" \
    "\n" \
    "Preprocesses input-output data for estimating state-space matrices and finds\n" \
    "an estimate of the system order using MOESP or N4SID subspace identification.\n" \
    "\n" \
    "Parameters:\n" \
    "  meth (str): 'M' for MOESP, 'N' for N4SID\n" \
    "  alg (str): 'C' for Cholesky, 'F' for Fast QR, 'Q' for QR\n" \
    "  jobd (str): 'M' or 'N' for MOESP BD computation mode (not used for N4SID)\n" \
    "  batch (str): 'F' first, 'I' intermediate, 'L' last, 'O' one block\n" \
    "  conct (str): 'C' connected blocks, 'N' not connected\n" \
    "  ctrl (str): 'C' user confirmation, 'N' no confirmation\n" \
    "  nobr (int): Number of block rows (nobr > 0)\n" \
    "  m (int): Number of system inputs (m >= 0)\n" \
    "  l (int): Number of system outputs (l > 0)\n" \
    "  u (ndarray): NSMP-by-M input data (F-order)\n" \
    "  y (ndarray): NSMP-by-L output data (F-order)\n" \
    "  rcond (float): Rank tolerance for N4SID (0 for default)\n" \
    "  tol (float): Order estimation tolerance (>=0: threshold, <0: gap-based)\n" \
    "\n" \
    "Returns:\n" \
    "  (n, r, sv, iwarn, info): Order, R/S factor, singular values, warning, status"

#define DOC_IB01BD "State-space matrices estimation from N4SID/MOESP triangular factor.\n" \
    "\n" \
    "Estimates system matrices (A,C,B,D), optionally noise covariance matrices\n" \
    "(Q,Ry,S), and Kalman gain K from the triangular factor R computed by IB01AD.\n" \
    "\n" \
    "Parameters:\n" \
    "  meth (str): 'M' MOESP, 'N' N4SID, 'C' combined\n" \
    "  job (str): 'A' all matrices, 'C' A,C only, 'B' B,D only, 'D' D only\n" \
    "  jobck (str): 'K' Kalman gain, 'C' covariances, 'N' neither\n" \
    "  nobr (int): Number of block rows (nobr > 1)\n" \
    "  n (int): System order (0 < n < nobr)\n" \
    "  m (int): Number of system inputs (m >= 0)\n" \
    "  l (int): Number of system outputs (l > 0)\n" \
    "  nsmpl (int): Number of samples\n" \
    "  r (ndarray): Triangular factor from IB01AD\n" \
    "  tol (float): Tolerance for rank estimation\n" \
    "\n" \
    "Returns:\n" \
    "  (A, C, B, D, Q, Ry, S, K, iwarn, info): Matrices, covariances, gain, status"

#define DOC_IB01CD "Estimate initial state and system matrices B, D (driver routine).\n" \
    "\n" \
    "Driver that transforms A to Schur form and calls IB01QD/IB01RD:\n" \
    "  x(k+1) = A*x(k) + B*u(k)\n" \
    "  y(k)   = C*x(k) + D*u(k)\n" \
    "\n" \
    "Parameters:\n" \
    "  jobx0 (str): 'X' to compute x0, 'N' to set x0=0\n" \
    "  comuse (str): 'C' compute B/D, 'U' use given B/D, 'N' neither\n" \
    "  job (str): 'B' compute B only, 'D' compute B and D\n" \
    "  n (int): System order (n >= 0)\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  l (int): Number of outputs (l > 0)\n" \
    "  a (ndarray): N-by-N state matrix A (F-order)\n" \
    "  b (ndarray): N-by-M input matrix B (F-order)\n" \
    "  c (ndarray): L-by-N output matrix C (F-order)\n" \
    "  d (ndarray): L-by-M feedthrough D (F-order)\n" \
    "  u (ndarray): NSMP-by-M input data (F-order)\n" \
    "  y (ndarray): NSMP-by-L output data (F-order)\n" \
    "  tol (float): Tolerance for rank estimation\n" \
    "\n" \
    "Returns:\n" \
    "  (x0, B, D, V, rcond, iwarn, info):\n" \
    "  - x0: Estimated initial state (n,)\n" \
    "  - B: Input matrix (n,m)\n" \
    "  - D: Feedthrough matrix (l,m)\n" \
    "  - V: Orthogonal Schur transformation (n,n)\n" \
    "  - rcond: Reciprocal condition number\n" \
    "  - iwarn: Warning (6 = A not stable)\n" \
    "  - info: Exit code (0=success, 1=Schur failed)"

#define DOC_IB01MD "Upper triangular factor R of concatenated block Hankel matrices.\n" \
    "\n" \
    "Constructs R from QR factorization of block Hankel matrices built from\n" \
    "input-output data. Data can be processed sequentially.\n" \
    "\n" \
    "Parameters:\n" \
    "  meth (str): 'M' for MOESP, 'N' for N4SID\n" \
    "  alg (str): 'C' for Cholesky, 'F' for Fast QR, 'Q' for QR\n" \
    "  batch (str): 'F' first, 'I' intermediate, 'L' last, 'O' one block\n" \
    "  conct (str): 'C' connected blocks, 'N' not connected\n" \
    "  nobr (int): Number of block rows (nobr > 0)\n" \
    "  m (int): Number of system inputs (m >= 0)\n" \
    "  l (int): Number of system outputs (l > 0)\n" \
    "  u (ndarray): NSMP-by-M input data (F-order)\n" \
    "  y (ndarray): NSMP-by-L output data (F-order)\n" \
    "  r (ndarray, optional): Previous R for sequential (F-order)\n" \
    "  iwork (ndarray, optional): Previous iwork for sequential\n" \
    "\n" \
    "Returns for batch='O' or 'L':\n" \
    "  (r, iwarn, info): Upper triangular R, warning, exit code\n" \
    "Returns for batch='F' or 'I':\n" \
    "  (r, iwork, iwarn, info): R, iwork state, warning, exit code"

#define DOC_IB01ND "SVD system order via block Hankel.\n" \
    "\n" \
    "Computes SVD of triangular factor R from QR factorization of\n" \
    "concatenated block Hankel matrices to determine system order.\n" \
    "\n" \
    "Parameters:\n" \
    "  meth (str): 'M' for MOESP, 'N' for N4SID\n" \
    "  jobd (str): 'M' or 'N' for MOESP BD computation mode\n" \
    "  nobr (int): Number of block rows (nobr > 0)\n" \
    "  m (int): Number of system inputs (m >= 0)\n" \
    "  l (int): Number of system outputs (l > 0)\n" \
    "  r (ndarray): Upper triangular R matrix, dimension 2*(m+l)*nobr x 2*(m+l)*nobr\n" \
    "  tol (float): Tolerance for rank estimation (N4SID only)\n" \
    "\n" \
    "Returns:\n" \
    "  (r, sv, rcond1, rcond2, iwarn, info): Processed R, singular values, rconds, status"

#define DOC_IB01OD "Estimate system order from Hankel singular values.\n" \
    "\n" \
    "Estimates system order based on singular values of triangular factor\n" \
    "from QR factorization of concatenated block Hankel matrices.\n" \
    "\n" \
    "Parameters:\n" \
    "  ctrl (str): 'C' for user confirmation, 'N' for no confirmation\n" \
    "  nobr (int): Number of block rows (nobr > 0)\n" \
    "  l (int): Number of system outputs (l > 0)\n" \
    "  sv (ndarray): Singular values, dimension (l*nobr), descending order\n" \
    "  tol (float): Tolerance (>=0: threshold, 0: default, <0: gap-based)\n" \
    "\n" \
    "Returns:\n" \
    "  (n, iwarn, info): Estimated order, warning, exit code"

#define DOC_IB01OY "User's confirmation of the system order.\n" \
    "\n" \
    "Non-interactive version for library use. Validates parameters and ensures N <= NMAX.\n" \
    "\n" \
    "Parameters:\n" \
    "  ns (int): Number of singular values (ns > 0)\n" \
    "  nmax (int): Maximum value of system order (0 <= nmax <= ns)\n" \
    "  n (int): Estimated system order (0 <= n <= ns)\n" \
    "  sv (ndarray): Singular values, dimension (ns), descending order\n" \
    "\n" \
    "Returns:\n" \
    "  (n, info): Validated order, exit code"

#define DOC_IB01PD "Estimate system matrices from R factor (subspace identification).\n" \
    "\n" \
    "Estimates state-space matrices A, C, B, D from the R factor produced by\n" \
    "IB01MD. Optionally computes covariance matrices for Kalman gain.\n" \
    "\n" \
    "Parameters:\n" \
    "  meth (str): 'M' for MOESP, 'N' for N4SID\n" \
    "  job (str): 'A' all matrices, 'C' A/C only, 'B' B only, 'D' B/D only\n" \
    "  jobcv (str): 'C' compute covariances, 'N' no covariances\n" \
    "  nobr (int): Number of block rows (nobr > 1)\n" \
    "  n (int): System order (0 < n < nobr)\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  l (int): Number of outputs (l > 0)\n" \
    "  nsmpl (int): Number of samples (for covariances)\n" \
    "  r (ndarray): R factor from IB01MD (2*(m+l)*nobr x 2*(m+l)*nobr, F-order)\n" \
    "  tol (float): Tolerance for rank estimation\n" \
    "  a (ndarray, optional): Input A for JOB='B'/'D' with N4SID\n" \
    "  c (ndarray, optional): Input C for JOB='B'/'D' with N4SID\n" \
    "\n" \
    "Returns (varies by job/jobcv):\n" \
    "  JOB='A', JOBCV='N': (a, c, b, d, rcond, iwarn, info)\n" \
    "  JOB='A', JOBCV='C': (a, c, b, d, q, ry, s, o, rcond, iwarn, info)\n" \
    "  JOB='C', JOBCV='N': (a, c, rcond, iwarn, info)\n" \
    "  JOB='D', JOBCV='N': (b, d, rcond, iwarn, info)"

#define DOC_IB01QD "Estimate initial state and system matrices B, D.\n" \
    "\n" \
    "Given (A, C) and input/output trajectories, estimates B, D, and x(0)\n" \
    "for: x(k+1) = A*x(k) + B*u(k), y(k) = C*x(k) + D*u(k).\n" \
    "Matrix A must be in real Schur form.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobx0 (str): 'X' to compute x(0), 'N' if x(0) known to be zero\n" \
    "  job (str): 'B' for B only (D=0), 'D' for B and D\n" \
    "  n (int): System order (n >= 0)\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  l (int): Number of outputs (l > 0)\n" \
    "  a (ndarray): N-by-N state matrix A in Schur form (F-order)\n" \
    "  c (ndarray): L-by-N output matrix C (F-order)\n" \
    "  u (ndarray): NSMP-by-M input data (F-order)\n" \
    "  y (ndarray): NSMP-by-L output data (F-order)\n" \
    "  tol (float): Tolerance for rank estimation (<= 0 uses machine eps)\n" \
    "\n" \
    "Returns:\n" \
    "  (x0, b, d, rcond_w2, rcond_u, iwarn, info):\n" \
    "  - x0: Estimated initial state (n,)\n" \
    "  - b: Estimated input matrix B (n x m)\n" \
    "  - d: Estimated direct transmission D (l x m)\n" \
    "  - rcond_w2: Reciprocal condition of W2\n" \
    "  - rcond_u: Reciprocal condition of U (if JOB='D')\n" \
    "  - iwarn: Warning (4 = rank-deficient)\n" \
    "  - info: Exit code (0=success, 2=SVD failed)"

#define DOC_IB01RD "Estimate initial state for discrete-time LTI system.\n" \
    "\n" \
    "Given (A,B,C,D) and input/output trajectories, estimates initial state x(0)\n" \
    "for: x(k+1) = A*x(k) + B*u(k), y(k) = C*x(k) + D*u(k).\n" \
    "Matrix A must be in real Schur form.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'Z' if D is zero, 'N' if D is not zero\n" \
    "  n (int): System order (n >= 0)\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  l (int): Number of outputs (l > 0)\n" \
    "  nsmp (int): Number of samples (nsmp >= n)\n" \
    "  a (ndarray): N-by-N state matrix A in Schur form (F-order)\n" \
    "  b (ndarray): N-by-M input matrix B (F-order)\n" \
    "  c (ndarray): L-by-N output matrix C (F-order)\n" \
    "  d (ndarray): L-by-M direct transmission D (F-order)\n" \
    "  u (ndarray): NSMP-by-M input data (F-order)\n" \
    "  y (ndarray): NSMP-by-L output data (F-order)\n" \
    "  tol (float): Tolerance for rank estimation (<= 0 uses machine eps)\n" \
    "\n" \
    "Returns:\n" \
    "  (x0, rcond, iwarn, info):\n" \
    "  - x0: Estimated initial state (n,)\n" \
    "  - rcond: Reciprocal condition of triangular factor\n" \
    "  - iwarn: Warning (4 = rank-deficient)\n" \
    "  - info: Exit code (0=success, 2=SVD failed)"

#define DOC_IB03AD "Wiener system identification with algorithm choice.\n" \
    "\n" \
    "Computes parameters for approximating a Wiener system consisting of a\n" \
    "linear state-space part and a static nonlinearity (neural network):\n" \
    "  x(t+1) = A*x(t) + B*u(t)       (linear state-space)\n" \
    "  z(t)   = C*x(t) + D*u(t)\n" \
    "  y(t)   = f(z(t), wb(1:L))      (nonlinear function)\n" \
    "\n" \
    "Parameters:\n" \
    "  init (str): 'L' init linear, 'S' init nonlinear, 'B' init both, 'N' no init\n" \
    "  alg (str): 'D' Cholesky (direct), 'I' CG (iterative)\n" \
    "  stor (str): 'F' full storage, 'P' packed storage\n" \
    "  nobr (int): Block rows for MOESP/N4SID (used if init='L' or 'B')\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  l (int): Number of outputs (l >= 0, l > 0 if init='L' or 'B')\n" \
    "  nsmp (int): Number of input/output samples\n" \
    "  n (int): System order (n < nobr if init='L' or 'B')\n" \
    "  nn (int): Number of neurons (nn >= 0)\n" \
    "  itmax1 (int): Max iterations for nonlinear init\n" \
    "  itmax2 (int): Max iterations for whole optimization\n" \
    "  u (ndarray): Input samples (nsmp x m, F-order)\n" \
    "  y (ndarray): Output samples (nsmp x l, F-order)\n" \
    "  tol1 (float): Tolerance for nonlinear init (< 0 uses sqrt(eps))\n" \
    "  tol2 (float): Tolerance for whole optimization (< 0 uses sqrt(eps))\n" \
    "  dwork_seed (ndarray, optional): Seed for random init (4 values)\n" \
    "  x_init (ndarray, optional): Initial parameters\n" \
    "  nprint (int, optional): Print control (default 0)\n" \
    "\n" \
    "Returns:\n" \
    "  (x, iwarn, info, dwork): Parameters, warning, status, workspace info"

#define DOC_IB03BD "Wiener system identification using Levenberg-Marquardt algorithm.\n" \
    "\n" \
    "Computes parameters for approximating a Wiener system consisting of a\n" \
    "linear state-space part and a static nonlinearity (neural network):\n" \
    "  x(t+1) = A*x(t) + B*u(t)       (linear state-space)\n" \
    "  z(t)   = C*x(t) + D*u(t)\n" \
    "  y(t)   = f(z(t), wb(1:L))      (nonlinear function)\n" \
    "\n" \
    "Parameters:\n" \
    "  init (str): 'L' init linear, 'S' init nonlinear, 'B' init both, 'N' no init\n" \
    "  nobr (int): Block rows for MOESP/N4SID (used if init='L' or 'B')\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  l (int): Number of outputs (l >= 0, l > 0 if init='L' or 'B')\n" \
    "  nsmp (int): Number of input/output samples\n" \
    "  n (int): System order (n < nobr if init='L' or 'B')\n" \
    "  nn (int): Number of neurons (nn >= 0)\n" \
    "  itmax1 (int): Max iterations for nonlinear init\n" \
    "  itmax2 (int): Max iterations for whole optimization\n" \
    "  u (ndarray): Input samples (nsmp x m, F-order)\n" \
    "  y (ndarray): Output samples (nsmp x l, F-order)\n" \
    "  tol1 (float): Tolerance for nonlinear init (< 0 uses sqrt(eps))\n" \
    "  tol2 (float): Tolerance for whole optimization (< 0 uses sqrt(eps))\n" \
    "  dwork_seed (ndarray, optional): Seed for random init (4 values)\n" \
    "  x_init (ndarray, optional): Initial parameters\n" \
    "  nprint (int, optional): Print control (default 0)\n" \
    "\n" \
    "Returns:\n" \
    "  (x, iwarn, info, dwork): Parameters, warning, status, workspace info"

#define DOC_MA01AD "Compute complex square root in real arithmetic.\n" \
    "\n" \
    "Computes the complex square root YR + i*YI of XR + i*XI.\n" \
    "The result satisfies: YR >= 0 and SIGN(YI) = SIGN(XI).\n" \
    "\n" \
    "Parameters:\n" \
    "  xr (float): Real part of input complex number\n" \
    "  xi (float): Imaginary part of input complex number\n" \
    "\n" \
    "Returns:\n" \
    "  (yr, yi): Real and imaginary parts of square root"

#define DOC_MA01BD "Compute general product of K real scalars without overflow/underflow.\n" \
    "\n" \
    "Computes the product of K scalars stored in array A, controlled by\n" \
    "signature array S. Result is ALPHA / BETA * BASE^SCAL.\n" \
    "\n" \
    "Parameters:\n" \
    "  base (float): Machine base\n" \
    "  lgbas (float): Logarithm of BASE\n" \
    "  k (int): Number of scalars (k >= 1)\n" \
    "  s (ndarray): Signature array (int32). s[i]=1 multiply, s[i]=-1 divide\n" \
    "  a (ndarray): Array of real scalars\n" \
    "  inca (int): Increment for array A (inca != 0)\n" \
    "\n" \
    "Returns:\n" \
    "  (alpha, beta, scal): Scaled result: ALPHA / BETA * BASE^SCAL"

#define DOC_MA01BZ "Compute general product of K complex scalars without overflow/underflow.\n" \
    "\n" \
    "Computes the product of K complex scalars stored in array A, controlled by\n" \
    "signature array S. Result is ALPHA / BETA * BASE^SCAL.\n" \
    "\n" \
    "Parameters:\n" \
    "  base (float): Machine base\n" \
    "  k (int): Number of scalars (k >= 1)\n" \
    "  s (ndarray): Signature array (int32). s[i]=1 multiply, s[i]=-1 divide\n" \
    "  a (ndarray): Array of complex scalars (complex128)\n" \
    "  inca (int): Increment for array A (inca != 0)\n" \
    "\n" \
    "Returns:\n" \
    "  (alpha, beta, scal): Scaled result: ALPHA / BETA * BASE^SCAL\n" \
    "                       alpha: complex numerator with 1 <= |alpha| < BASE\n" \
    "                       beta: complex (0 or 1)"

#define DOC_MA01CD "Compute sign of sum of two scaled numbers without overflow.\n" \
    "\n" \
    "Computes the sign of A * BASE^IA + B * BASE^IB without overflow.\n" \
    "Any base can be used, but must be the same for both numbers.\n" \
    "\n" \
    "Parameters:\n" \
    "  a (float): First real scalar\n" \
    "  ia (int): Exponent for first scalar (A * BASE^IA)\n" \
    "  b (float): Second real scalar\n" \
    "  ib (int): Exponent for second scalar (B * BASE^IB)\n" \
    "\n" \
    "Returns:\n" \
    "  int: Sign of the sum: 1 (positive), 0 (zero), or -1 (negative)"

#define DOC_MA01DD "Compute approximate symmetric chordal metric for two complex numbers.\n" \
    "\n" \
    "Computes D = MIN(|A1 - A2|, |1/A1 - 1/A2|) for A1 = AR1 + i*AI1 and\n" \
    "A2 = AR2 + i*AI2. The chordal metric is finite even if both numbers\n" \
    "are infinite, or if one is infinite and the other is finite and nonzero.\n" \
    "\n" \
    "Parameters:\n" \
    "  ar1 (float): Real part of first complex number A1\n" \
    "  ai1 (float): Imaginary part of first complex number A1\n" \
    "  ar2 (float): Real part of second complex number A2\n" \
    "  ai2 (float): Imaginary part of second complex number A2\n" \
    "  eps (float): Relative machine precision (DLAMCH('E'))\n" \
    "  safemn (float): Safe minimum (DLAMCH('S'))\n" \
    "\n" \
    "Returns:\n" \
    "  float: The approximate symmetric chordal metric D (D >= 0)"

#define DOC_MA01DZ "Compute approximate symmetric chordal metric for two complex rationals.\n" \
    "\n" \
    "Computes D = MIN(|A1 - A2|, |1/A1 - 1/A2|) for complex numbers A1 and A2\n" \
    "represented as rationals: Aj = (ARj + i*AIj) / Bj.\n" \
    "\n" \
    "Special cases:\n" \
    "  - Bj = 0 with nonzero numerator means Aj is infinite\n" \
    "  - ARj = AIj = Bj = 0 means Aj is not a number (NaN)\n" \
    "\n" \
    "Parameters:\n" \
    "  ar1 (float): Real part of numerator of A1\n" \
    "  ai1 (float): Imaginary part of numerator of A1\n" \
    "  b1 (float): Denominator of A1 (b1 >= 0)\n" \
    "  ar2 (float): Real part of numerator of A2\n" \
    "  ai2 (float): Imaginary part of numerator of A2\n" \
    "  b2 (float): Denominator of A2 (b2 >= 0)\n" \
    "  eps (float): Relative machine precision (DLAMCH('E'))\n" \
    "  safemn (float): Safe minimum (DLAMCH('S'))\n" \
    "\n" \
    "Returns:\n" \
    "  (d1, d2, iwarn): Chordal metric D = d1/d2\n" \
    "    d1: Numerator (d1 >= 0)\n" \
    "    d2: Denominator (0 or 1)\n" \
    "    iwarn: 0 = success, 1 = NaN input (d1 = d2 = 0)"

#define DOC_MA02AD "Transpose all or part of a matrix.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'U' for upper triangular, 'L' for lower, else full matrix\n" \
    "  a (ndarray): Input matrix (m x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  b (ndarray): Transposed matrix (n x m, F-order)"

#define DOC_MA02AZ "Transpose or conjugate-transpose a complex matrix.\n" \
    "\n" \
    "(Conjugate) transposes all or part of a two-dimensional complex matrix A into another matrix B.\n" \
    "\n" \
    "Parameters:\n" \
    "  trans (str): 'T' = transpose (B = A^T), 'C' = conjugate transpose (B = A^H)\n" \
    "  job (str): 'U' = upper triangular only, 'L' = lower triangular only, other = full matrix\n" \
    "  a (ndarray): Input complex matrix (m x n, F-order, complex128)\n" \
    "\n" \
    "Returns:\n" \
    "  b (ndarray): Output transposed matrix (n x m, F-order, complex128)"

#define DOC_MA02BD "Reverse order of rows and/or columns of a matrix.\n" \
    "\n" \
    "Pre/post-multiplies matrix by permutation matrix P with ones\n" \
    "down the secondary diagonal.\n" \
    "\n" \
    "Parameters:\n" \
    "  side (str): 'L' (P*A), 'R' (A*P), 'B' (P*A*P)\n" \
    "  a (ndarray): Matrix (m x n, F-order), modified in place\n" \
    "\n" \
    "Returns:\n" \
    "  a: Permuted matrix"

#define DOC_MA02BZ "Reverse order of rows and/or columns of a complex matrix.\n" \
    "\n" \
    "Pre/post-multiplies complex matrix by permutation matrix P with ones\n" \
    "down the secondary diagonal.\n" \
    "\n" \
    "Parameters:\n" \
    "  side (str): 'L' (P*A), 'R' (A*P), 'B' (P*A*P)\n" \
    "  a (ndarray): Complex matrix (m x n, F-order, complex128), modified in place\n" \
    "\n" \
    "Returns:\n" \
    "  a: Permuted complex matrix"

#define DOC_MA02CD "Pertranspose the central band of a square matrix.\n" \
    "\n" \
    "Computes the pertranspose of the central band (KL subdiagonals, main\n" \
    "diagonal, KU superdiagonals). Equivalent to P*B'*P where P has ones\n" \
    "on the secondary diagonal.\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): Square matrix (n x n, F-order), modified in place\n" \
    "  kl (int): Number of subdiagonals to pertranspose (0 <= kl <= n-1)\n" \
    "  ku (int): Number of superdiagonals to pertranspose (0 <= ku <= n-1)\n" \
    "\n" \
    "Returns:\n" \
    "  a (ndarray): Matrix with central band pertransposed"

#define DOC_MA02CZ "Pertranspose the central band of a complex square matrix.\n" \
    "\n" \
    "Computes the pertranspose of the central band of a square complex matrix.\n" \
    "The pertranspose reverses elements along each antidiagonal within the\n" \
    "specified band (KL subdiagonals, main diagonal, KU superdiagonals).\n" \
    "This is equivalent to P*B'*P where B is the band matrix and P is\n" \
    "a permutation matrix with ones on the secondary diagonal.\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): Complex square matrix (n x n, F-order, complex128), modified in place\n" \
    "  kl (int): Number of subdiagonals to pertranspose (0 <= kl <= n-1)\n" \
    "  ku (int): Number of superdiagonals to pertranspose (0 <= ku <= n-1)\n" \
    "\n" \
    "Returns:\n" \
    "  a: Matrix with central band pertransposed"

#define DOC_MA02DD "Pack/unpack upper or lower triangle of symmetric matrix.\n" \
    "\n" \
    "Packs or unpacks the upper or lower triangle of a symmetric matrix.\n" \
    "The packed matrix is stored column-wise in a one-dimensional array.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'P' = pack, 'U' = unpack\n" \
    "  uplo (str): 'U' = upper triangular, 'L' = lower triangular\n" \
    "  a/ap (ndarray): For job='P': square matrix to pack (n x n, F-order)\n" \
    "                  For job='U': packed array (n*(n+1)/2 elements)\n" \
    "  n (int): Matrix dimension (required for job='U')\n" \
    "\n" \
    "Returns:\n" \
    "  For job='P': packed array (1D, n*(n+1)/2 elements)\n" \
    "  For job='U': unpacked matrix (n x n, F-order)"

#define DOC_MA02ED "Store by symmetry the upper or lower triangle of a symmetric matrix.\n" \
    "\n" \
    "Parameters:\n" \
    "  uplo (str): 'U' if upper triangle given, 'L' if lower triangle given\n" \
    "  a (ndarray): Symmetric matrix (n x n, F-order), modified in place\n" \
    "\n" \
    "Returns:\n" \
    "  a (ndarray): Completed symmetric matrix with both triangles"

#define DOC_MA02ES "Store by skew-symmetry the upper or lower triangle of a skew-symmetric matrix.\n" \
    "\n" \
    "Completes a skew-symmetric matrix by negating one triangle to fill the other.\n" \
    "Diagonal entries are set to zero.\n" \
    "\n" \
    "Parameters:\n" \
    "  uplo (str): 'U' if upper triangle given, 'L' if lower triangle given\n" \
    "  a (ndarray): Skew-symmetric matrix (n x n, F-order), modified in place\n" \
    "\n" \
    "Returns:\n" \
    "  a (ndarray): Completed skew-symmetric matrix (A = -A^T)"

#define DOC_MA02EZ "Store by (skew-)symmetry the upper or lower triangle of a complex matrix.\n" \
    "\n" \
    "Completes a (skew-)symmetric/Hermitian complex matrix by copying one\n" \
    "triangle to the other, optionally applying conjugation or negation.\n" \
    "\n" \
    "Parameters:\n" \
    "  uplo (str): 'U' (upper given), 'L' (lower given)\n" \
    "  trans (str): 'T' (transpose), 'C' (conjugate transpose)\n" \
    "  skew (str): 'G' (general), 'N' (symmetric/Hermitian), 'S' (skew-symmetric/Hermitian)\n" \
    "  a (ndarray): Complex square matrix (n x n, F-order, complex128), modified in place\n" \
    "\n" \
    "Returns:\n" \
    "  a: Completed (skew-)symmetric/Hermitian matrix\n" \
    "\n" \
    "Notes:\n" \
    "  For TRANS='C', SKEW='N': diagonal imaginary parts set to 0 (Hermitian)\n" \
    "  For TRANS='C', SKEW='S': diagonal real parts set to 0 (skew-Hermitian)"

#define DOC_MA02GD "Column interchanges on a matrix.\n" \
    "\n" \
    "Performs column swaps based on pivot indices.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Number of rows\n" \
    "  a (ndarray): Matrix to permute (n x m, F-order)\n" \
    "  k1 (int): First pivot index (1-based)\n" \
    "  k2 (int): Last pivot index (1-based)\n" \
    "  ipiv (ndarray): Pivot indices (int32)\n" \
    "  incx (int): Increment for IPIV traversal\n" \
    "\n" \
    "Returns:\n" \
    "  a: Permuted matrix"

#define DOC_MA02GZ "Column interchanges on a complex matrix.\n" \
    "\n" \
    "Performs a series of column swaps based on pivot indices. One column\n" \
    "interchange is initiated for each of columns K1 through K2 of A.\n" \
    "Column-oriented counterpart of LAPACK's ZLASWP (row swaps).\n" \
    "Complex version of MA02GD.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Number of rows of matrix A (n >= 0)\n" \
    "  a (ndarray): Complex matrix (n x m, F-order, complex128), modified in place\n" \
    "  k1 (int): First column for which interchange is done (1-based)\n" \
    "  k2 (int): Last column for which interchange is done (1-based)\n" \
    "  ipiv (ndarray): Pivot indices (int32). IPIV[k-1]=l means swap columns k and l\n" \
    "  incx (int): Increment between IPIV elements. Negative = reverse order.\n" \
    "\n" \
    "Returns:\n" \
    "  a: Permuted complex matrix"

#define DOC_MA02HD "Check if matrix equals scalar times identity-like matrix.\n" \
    "\n" \
    "Checks if A = DIAG*I, where I is an M-by-N matrix with ones on\n" \
    "the diagonal and zeros elsewhere.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): Part of matrix to check: 'U' (upper), 'L' (lower), other (all)\n" \
    "  a (ndarray): Input matrix (m x n, F-order)\n" \
    "  diag (float): Scalar to compare diagonal elements against\n" \
    "\n" \
    "Returns:\n" \
    "  bool: True if A = DIAG*I in specified region, False otherwise.\n" \
    "        Returns False if min(m,n) = 0."

#define DOC_MA02HZ "Check if complex matrix equals scalar times identity-like matrix.\n" \
    "\n" \
    "Checks if A = DIAG*I, where I is an M-by-N matrix with ones on\n" \
    "the diagonal and zeros elsewhere. A and DIAG are complex.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): Part of matrix to check: 'U' (upper), 'L' (lower), other (all)\n" \
    "  a (ndarray): Input complex matrix (m x n, F-order, complex128)\n" \
    "  diag (complex): Complex scalar to compare diagonal elements against\n" \
    "\n" \
    "Returns:\n" \
    "  bool: True if A = DIAG*I in specified region, False otherwise.\n" \
    "        Returns False if min(m,n) = 0."

#define DOC_MA02IZ "Compute norm of complex skew-Hamiltonian or Hamiltonian matrix.\n" \
    "\n" \
    "Computes the one norm, Frobenius norm, infinity norm, or max element\n" \
    "of a complex skew-Hamiltonian or Hamiltonian matrix:\n" \
    "    H = [A, G; Q, -A^H] for Hamiltonian (G=G^H, Q=Q^H)\n" \
    "    X = [A, G; Q, A^H]  for skew-Hamiltonian (G=-G^H, Q=-Q^H)\n" \
    "\n" \
    "Note: For these matrix types, the infinity norm equals the one norm.\n" \
    "\n" \
    "Parameters:\n" \
    "  typ (str): 'S' = skew-Hamiltonian, 'H' = Hamiltonian\n" \
    "  norm (str): '1'/'O' = one norm, 'F'/'E' = Frobenius, 'I' = infinity, 'M' = max\n" \
    "  a (ndarray): Complex matrix A (n x n, F-order, complex128)\n" \
    "  qg (ndarray): Complex matrix containing Q and G (n x n+1, F-order, complex128)\n" \
    "\n" \
    "Returns:\n" \
    "  float: The computed norm value"

#define DOC_MA02JD "Test if a matrix is an orthogonal symplectic matrix.\n" \
    "\n" \
    "Computes || Q^T Q - I ||_F for a matrix of the form:\n" \
    "    Q = [  op(Q1)   op(Q2) ]\n" \
    "        [ -op(Q2)   op(Q1) ]\n" \
    "\n" \
    "where Q1 and Q2 are N-by-N matrices. This residual can be used to\n" \
    "test whether Q is numerically an orthogonal symplectic matrix.\n" \
    "\n" \
    "Parameters:\n" \
    "  ltran1 (bool): If True, op(Q1) = Q1^T; otherwise op(Q1) = Q1\n" \
    "  ltran2 (bool): If True, op(Q2) = Q2^T; otherwise op(Q2) = Q2\n" \
    "  q1 (ndarray): Matrix Q1 (n x n, F-order)\n" \
    "  q2 (ndarray): Matrix Q2 (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  float: The computed residual || Q^T Q - I ||_F"

#define DOC_MA02JZ "Test if a complex matrix is a unitary symplectic matrix.\n" \
    "\n" \
    "Computes || Q^H Q - I ||_F for a complex matrix of the form:\n" \
    "    Q = [  op(Q1)   op(Q2) ]\n" \
    "        [ -op(Q2)   op(Q1) ]\n" \
    "\n" \
    "where Q1 and Q2 are N-by-N complex matrices. This residual can be used\n" \
    "to test whether Q is numerically a unitary symplectic matrix.\n" \
    "\n" \
    "This is the complex version of MA02JD.\n" \
    "\n" \
    "Parameters:\n" \
    "  ltran1 (bool): If True, op(Q1) = Q1'; otherwise op(Q1) = Q1\n" \
    "  ltran2 (bool): If True, op(Q2) = Q2'; otherwise op(Q2) = Q2\n" \
    "  q1 (ndarray): Complex matrix Q1 (n x n, F-order, complex128)\n" \
    "  q2 (ndarray): Complex matrix Q2 (n x n, F-order, complex128)\n" \
    "\n" \
    "Returns:\n" \
    "  float: The computed residual || Q^H Q - I ||_F"

#define DOC_MA02MD "Compute norms of a real skew-symmetric matrix.\n" \
    "\n" \
    "Computes the value of the one norm, or the Frobenius norm, or\n" \
    "the infinity norm, or the element of largest absolute value\n" \
    "of a real skew-symmetric matrix.\n" \
    "\n" \
    "Note: For skew-symmetric matrices, the infinity norm equals the one norm.\n" \
    "\n" \
    "Parameters:\n" \
    "  norm (str): Specifies the value to return:\n" \
    "              '1' or 'O' = one norm\n" \
    "              'F' or 'E' = Frobenius norm\n" \
    "              'I' = infinity norm\n" \
    "              'M' = max(abs(A(i,j)))\n" \
    "  uplo (str): 'U' = upper triangular part stored, 'L' = lower triangular part stored\n" \
    "  a (ndarray): Skew-symmetric matrix (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  float: The computed norm value"

#define DOC_MA02MZ "Compute norms of a complex skew-Hermitian matrix.\n" \
    "\n" \
    "Computes the value of the one norm, or the Frobenius norm, or\n" \
    "the infinity norm, or the element of largest absolute value\n" \
    "of a complex skew-Hermitian matrix.\n" \
    "\n" \
    "Note: For skew-Hermitian matrices, the infinity norm equals the one norm.\n" \
    "Diagonal elements are assumed to be pure imaginary (real parts = 0).\n" \
    "\n" \
    "Parameters:\n" \
    "  norm (str): Specifies the value to return:\n" \
    "              '1' or 'O' = one norm\n" \
    "              'F' or 'E' = Frobenius norm\n" \
    "              'I' = infinity norm\n" \
    "              'M' = max(abs(A(i,j)))\n" \
    "  uplo (str): 'U' = upper triangular part stored, 'L' = lower triangular part stored\n" \
    "  a (ndarray): Complex skew-Hermitian matrix (n x n, F-order, complex128)\n" \
    "\n" \
    "Returns:\n" \
    "  float: The computed norm value"

#define DOC_MA02NZ "Permute two rows and corresponding columns of a (skew-)symmetric/Hermitian complex matrix.\n" \
    "\n" \
    "Permutes rows K and L and the corresponding columns K and L of a\n" \
    "(skew-)symmetric/Hermitian complex matrix, stored in triangular form.\n" \
    "\n" \
    "Parameters:\n" \
    "  uplo (str): 'U' = upper triangular part stored, 'L' = lower triangular part stored\n" \
    "  trans (str): 'T' = use transposition, 'C' = use conjugate transposition\n" \
    "  skew (str): 'N' = symmetric/Hermitian, 'S' = skew-symmetric/skew-Hermitian\n" \
    "  k (int): Smaller index of pair to permute (1-based). If k=0, routine returns.\n" \
    "  l (int): Larger index of pair to permute (1-based). k <= l <= n.\n" \
    "  a (ndarray): Complex matrix (n x n, F-order, complex128). Modified in-place.\n" \
    "\n" \
    "Returns:\n" \
    "  a: The permuted matrix (modified in-place)"

#define DOC_MA02OD "Count zero rows of a real (skew-)Hamiltonian matrix.\n" \
    "\n" \
    "Computes the number of zero rows (and zero columns) of a real\n" \
    "(skew-)Hamiltonian matrix:\n" \
    "\n" \
    "      (  A    D   )\n" \
    "  H = (           )\n" \
    "      (  E  +/-A' )\n" \
    "\n" \
    "The matrix E is stored in the lower triangular part of DE (columns 1..M),\n" \
    "and D is stored in the upper triangular part of DE (columns 2..M+1).\n" \
    "\n" \
    "Parameters:\n" \
    "  skew (str): 'H' = Hamiltonian (D=D', E=E'),\n" \
    "              'S' = skew-Hamiltonian (D=-D', E=-E', diagonal assumed zero)\n" \
    "  a (ndarray): Matrix A (m x m, F-order)\n" \
    "  de (ndarray): Matrix DE (m x m+1, F-order) containing E (lower tri) and D (upper tri)\n" \
    "\n" \
    "Returns:\n" \
    "  int: Number of zero rows in H"

#define DOC_MA02OZ "Count zero rows of a complex (skew-)Hamiltonian matrix.\n" \
    "\n" \
    "Computes the number of zero rows (and zero columns) of a complex\n" \
    "(skew-)Hamiltonian matrix:\n" \
    "\n" \
    "      (  A    D   )\n" \
    "  H = (           )\n" \
    "      (  E  +/-A' )\n" \
    "\n" \
    "The matrix E is stored in the lower triangular part of DE (columns 1..M),\n" \
    "and D is stored in the upper triangular part of DE (columns 2..M+1).\n" \
    "\n" \
    "For Hamiltonian: D and E are Hermitian, real parts of diagonal are checked.\n" \
    "For skew-Hamiltonian: D and E are skew-Hermitian, imaginary parts of diagonal are checked.\n" \
    "\n" \
    "This is the complex version of MA02OD.\n" \
    "\n" \
    "Parameters:\n" \
    "  skew (str): 'H' = Hamiltonian (D=D^H, E=E^H),\n" \
    "              'S' = skew-Hamiltonian (D=-D^H, E=-E^H)\n" \
    "  a (ndarray): Complex matrix A (m x m, F-order, complex128)\n" \
    "  de (ndarray): Complex matrix DE (m x m+1, F-order, complex128)\n" \
    "\n" \
    "Returns:\n" \
    "  int: Number of zero rows in H"

#define DOC_MA02PD "Compute the number of zero rows and zero columns of a real matrix.\n" \
    "\n" \
    "Scans the M-by-N matrix A to count rows and columns that contain only zeros.\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): Input matrix (m x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  nzr (int): Number of zero rows\n" \
    "  nzc (int): Number of zero columns"

#define DOC_MA02PZ "Compute the number of zero rows and zero columns of a complex matrix.\n" \
    "\n" \
    "Scans the M-by-N complex matrix A to count rows and columns that contain only zeros.\n" \
    "Complex version of MA02PD.\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): Input complex matrix (m x n, F-order, complex128)\n" \
    "\n" \
    "Returns:\n" \
    "  nzr (int): Number of zero rows\n" \
    "  nzc (int): Number of zero columns"

#define DOC_MA02RD "Sort vector D and rearrange E with same permutation.\n" \
    "\n" \
    "Sorts the elements of an n-vector D in increasing (ID='I') or decreasing\n" \
    "(ID='D') order, and rearranges the elements of an n-vector E using the\n" \
    "same permutations.\n" \
    "\n" \
    "Uses Quick Sort with Insertion sort fallback for arrays of length <= 20.\n" \
    "Based on LAPACK DLASRT, but applies to E the same interchanges used for D.\n" \
    "\n" \
    "Parameters:\n" \
    "  id (str): 'I' for increasing order, 'D' for decreasing order\n" \
    "  d (ndarray): Vector to sort (n-element, F-order), modified in place\n" \
    "  e (ndarray): Vector to rearrange (n-element, F-order), modified in place\n" \
    "\n" \
    "Returns:\n" \
    "  d (ndarray): Sorted vector D\n" \
    "  e (ndarray): Rearranged vector E (same permutation as D)\n" \
    "  info (int): 0 on success, -i if i-th argument had illegal value"

#define DOC_MA02SD "Compute smallest nonzero absolute value of matrix elements.\n" \
    "\n" \
    "Finds the smallest nonzero |A(i,j)| in a real M-by-N matrix A.\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): Input matrix (m x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  result (float): Smallest nonzero absolute value. Returns 0 if M=0 or N=0.\n" \
    "                  Returns overflow value if all elements are zero."

#define DOC_MB01KD "Skew-symmetric rank-2k update.\n" \
    "\n" \
    "Computes C := alpha*A*B' - alpha*B*A' + beta*C (trans='N')\n" \
    "or C := alpha*A'*B - alpha*B'*A + beta*C (trans='T'/'C')\n" \
    "where C is skew-symmetric.\n" \
    "\n" \
    "Parameters:\n" \
    "  uplo (str): 'U' for strictly upper triangle, 'L' for strictly lower\n" \
    "  trans (str): 'N' for C=alpha*A*B'-alpha*B*A'+beta*C, 'T'/'C' for transpose form\n" \
    "  n (int): Order of matrix C\n" \
    "  k (int): If trans='N': columns of A,B. If trans='T': rows of A,B\n" \
    "  alpha (float): Scalar multiplier\n" \
    "  a (ndarray): Matrix A (F-order)\n" \
    "  b (ndarray): Matrix B (F-order)\n" \
    "  beta (float): Scalar multiplier for C\n" \
    "  c (ndarray): n-by-n skew-symmetric C (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (c, info): Updated C (strictly upper/lower triangle) and exit code"

#define DOC_MB01LD "Skew-symmetric matrix update: R = alpha*R + beta*op(A)*X*op(A)'.\n" \
    "\n" \
    "Computes R := alpha*R + beta*op(A)*X*op(A)' where R and X are skew-symmetric.\n" \
    "\n" \
    "Parameters:\n" \
    "  uplo (str): 'U' for strictly upper triangle, 'L' for strictly lower\n" \
    "  trans (str): 'N' for op(A)=A, 'T'/'C' for op(A)=A'\n" \
    "  m (int): Order of R and rows of op(A)\n" \
    "  n (int): Order of X and columns of op(A)\n" \
    "  alpha (float): Scalar for R\n" \
    "  beta (float): Scalar for product\n" \
    "  r (ndarray): m-by-m skew-symmetric R (F-order)\n" \
    "  a (ndarray): Matrix A (F-order)\n" \
    "  x (ndarray): n-by-n skew-symmetric X (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (r, info): Updated R (strictly upper/lower triangle) and exit code"

#define DOC_MB01MD "Skew-symmetric matrix-vector multiply: y := alpha*A*x + beta*y.\n" \
    "\n" \
    "Performs y := alpha*A*x + beta*y where A is n-by-n skew-symmetric (A = -A').\n" \
    "Modified version of BLAS DSYMV for skew-symmetric matrices.\n" \
    "\n" \
    "Parameters:\n" \
    "  uplo (str): 'U' for strictly upper triangle, 'L' for strictly lower\n" \
    "  n (int): Order of matrix A\n" \
    "  alpha (float): Scalar for A*x. If alpha=0, A is not referenced\n" \
    "  a (ndarray): n-by-n skew-symmetric A (F-order)\n" \
    "  x (ndarray): Input vector x\n" \
    "  incx (int): Increment for x. If incx<0, x accessed in reverse order\n" \
    "  beta (float): Scalar for y. If beta=0, y need not be set on input\n" \
    "  y (ndarray): Input/output vector y\n" \
    "  incy (int): Increment for y. If incy<0, y accessed in reverse order\n" \
    "\n" \
    "Returns:\n" \
    "  (y, info): Updated y and exit code (0=success, i=param i invalid)"

#define DOC_MB01ND "Skew-symmetric rank 2 update: A := alpha*x*y' - alpha*y*x' + A.\n" \
    "\n" \
    "Performs the skew-symmetric rank 2 operation where A is n-by-n skew-symmetric (A = -A').\n" \
    "Modified version of BLAS DSYR2 for skew-symmetric matrices.\n" \
    "\n" \
    "Parameters:\n" \
    "  uplo (str): 'U' for strictly upper triangle, 'L' for strictly lower\n" \
    "  n (int): Order of matrix A\n" \
    "  alpha (float): Scalar for rank-2 update. If alpha=0, x and y not referenced\n" \
    "  x (ndarray): Input vector x\n" \
    "  incx (int): Increment for x. If incx<0, x accessed in reverse order\n" \
    "  y (ndarray): Input vector y\n" \
    "  incy (int): Increment for y. If incy<0, y accessed in reverse order\n" \
    "  a (ndarray): n-by-n skew-symmetric A (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, info): Updated A and exit code (0=success, i=param i invalid)"

#define DOC_MB01OC "Symmetric rank 2k update with Hessenberg matrix.\n" \
    "\n" \
    "Performs one of the special symmetric rank 2k operations:\n" \
    "    R := alpha*R + beta*H*X + beta*X*H'     (trans='N')\n" \
    "    R := alpha*R + beta*H'*X + beta*X*H     (trans='T' or 'C')\n" \
    "\n" \
    "where R and X are N-by-N symmetric matrices and H is N-by-N upper Hessenberg.\n" \
    "\n" \
    "Parameters:\n" \
    "  uplo (str): 'U' for upper triangle, 'L' for lower triangle of R and X\n" \
    "  trans (str): 'N' for H*X + X*H', 'T'/'C' for H'*X + X*H\n" \
    "  n (int): Order of matrices R, H, X\n" \
    "  alpha (float): Scalar for R. If alpha=0, R need not be set\n" \
    "  beta (float): Scalar for products. If beta=0, H and X not referenced\n" \
    "  r (ndarray): n-by-n symmetric R (F-order), triangle per uplo\n" \
    "  h (ndarray): n-by-n upper Hessenberg H (F-order)\n" \
    "  x (ndarray): n-by-n symmetric X (F-order), triangle per uplo\n" \
    "\n" \
    "Returns:\n" \
    "  (r, info): Updated R (same triangle) and exit code (0=success, -i=param i invalid)"

#define DOC_MB01OD "Symmetric rank 2k update with Hessenberg, symmetric, and triangular matrices.\n" \
    "\n" \
    "Computes the matrix formula:\n" \
    "    R := alpha*R + beta*(op(H)*X*op(E)' + op(E)*X*op(H)')\n" \
    "\n" \
    "where R and X are N-by-N symmetric matrices, H is N-by-N upper Hessenberg,\n" \
    "E is N-by-N upper triangular, and op(M) is M or M'.\n" \
    "\n" \
    "Parameters:\n" \
    "  uplo (str): 'U' for upper triangle, 'L' for lower triangle of R and X\n" \
    "  trans (str): 'N' for op(M)=M, 'T'/'C' for op(M)=M'\n" \
    "  n (int): Order of matrices R, H, X, E\n" \
    "  alpha (float): Scalar for R. If alpha=0, R need not be set\n" \
    "  beta (float): Scalar for products. If beta=0, H and X not referenced\n" \
    "  r (ndarray): n-by-n symmetric R (F-order), triangle per uplo\n" \
    "  h (ndarray): n-by-n upper Hessenberg H (F-order)\n" \
    "  x (ndarray): n-by-n symmetric X (F-order), triangle per uplo\n" \
    "  e (ndarray): n-by-n upper triangular E (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (r, info): Updated R (same triangle) and exit code (0=success, -i=param i invalid)"

#define DOC_MB01OE "Symmetric rank 2k update with Hessenberg and triangular matrices.\n" \
    "\n" \
    "Performs one of the symmetric rank 2k operations:\n" \
    "    R := alpha*R + beta*H*E' + beta*E*H'     (trans='N')\n" \
    "    R := alpha*R + beta*H'*E + beta*E'*H     (trans='T' or 'C')\n" \
    "\n" \
    "where R is N-by-N symmetric, H is N-by-N upper Hessenberg, and E is N-by-N upper triangular.\n" \
    "\n" \
    "Parameters:\n" \
    "  uplo (str): 'U' for upper triangle, 'L' for lower triangle of R\n" \
    "  trans (str): 'N' for H*E' + E*H', 'T'/'C' for H'*E + E'*H\n" \
    "  n (int): Order of matrices R, H, E\n" \
    "  alpha (float): Scalar for R. If alpha=0, R need not be set\n" \
    "  beta (float): Scalar for products. If beta=0, H and E not referenced\n" \
    "  r (ndarray): n-by-n symmetric R (F-order), triangle per uplo\n" \
    "  h (ndarray): n-by-n upper Hessenberg H (F-order)\n" \
    "  e (ndarray): n-by-n upper triangular E (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (r, info): Updated R (same triangle) and exit code (0=success, -i=param i invalid)"

#define DOC_MB01OH "Symmetric rank 2k update with two Hessenberg matrices.\n" \
    "\n" \
    "Performs one of the symmetric rank 2k operations:\n" \
    "    R := alpha*R + beta*H*A' + beta*A*H'     (trans='N')\n" \
    "    R := alpha*R + beta*H'*A + beta*A'*H     (trans='T' or 'C')\n" \
    "\n" \
    "where R is N-by-N symmetric, and H and A are N-by-N upper Hessenberg matrices.\n" \
    "\n" \
    "Parameters:\n" \
    "  uplo (str): 'U' for upper triangle, 'L' for lower triangle of R\n" \
    "  trans (str): 'N' for H*A' + A*H', 'T'/'C' for H'*A + A'*H\n" \
    "  n (int): Order of matrices R, H, A\n" \
    "  alpha (float): Scalar for R. If alpha=0, R need not be set\n" \
    "  beta (float): Scalar for products. If beta=0, H and A not referenced\n" \
    "  r (ndarray): n-by-n symmetric R (F-order), triangle per uplo\n" \
    "  h (ndarray): n-by-n upper Hessenberg H (F-order)\n" \
    "  a (ndarray): n-by-n upper Hessenberg A (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (r, info): Updated R (same triangle) and exit code (0=success, -i=param i invalid)"

#define DOC_MB01OO "Compute P = op(H)*X*op(E)' or P' with H Hessenberg, X symmetric, E triangular.\n" \
    "\n" \
    "Computes either P or P', with P defined by the matrix formula:\n" \
    "    P = op(H)*X*op(E)'\n" \
    "\n" \
    "where H is an upper Hessenberg matrix, X is a symmetric matrix,\n" \
    "E is an upper triangular matrix, and op(M) is M or M'.\n" \
    "\n" \
    "Parameters:\n" \
    "  uplo (str): 'U' for upper triangular part of X given, 'L' for lower\n" \
    "  trans (str): 'N' for P=H*X*E', 'T'/'C' for P'=E'*X*H\n" \
    "  h (ndarray): n-by-n upper Hessenberg H (F-order)\n" \
    "  x (ndarray): n-by-n symmetric X (F-order), triangle per uplo\n" \
    "  e (ndarray): n-by-n upper triangular E (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (p, info): Output matrix P and exit code (0=success, -i=param i invalid)"

#define DOC_MB01OS "Compute P = H*X or P = X*H with H Hessenberg and X symmetric.\n" \
    "\n" \
    "Computes the matrix product:\n" \
    "    P := H*X   (trans='N')\n" \
    "    P := X*H   (trans='T' or 'C')\n" \
    "\n" \
    "where H is an N-by-N upper Hessenberg matrix and X is an N-by-N symmetric matrix.\n" \
    "\n" \
    "Parameters:\n" \
    "  uplo (str): 'U' for upper triangular part of X given, 'L' for lower\n" \
    "  trans (str): 'N' for P=H*X, 'T'/'C' for P=X*H\n" \
    "  h (ndarray): n-by-n upper Hessenberg H (F-order)\n" \
    "  x (ndarray): n-by-n symmetric X (F-order), triangle per uplo\n" \
    "\n" \
    "Returns:\n" \
    "  (p, info): Output matrix P and exit code (0=success, -i=param i invalid)"

#define DOC_MB01OT "Symmetric rank 2k update with two upper triangular matrices.\n" \
    "\n" \
    "Performs one of the symmetric rank 2k operations:\n" \
    "    R := alpha*R + beta*E*T' + beta*T*E'     (trans='N')\n" \
    "    R := alpha*R + beta*E'*T + beta*T'*E     (trans='T' or 'C')\n" \
    "\n" \
    "where R is N-by-N symmetric, and E and T are N-by-N upper triangular matrices.\n" \
    "\n" \
    "Parameters:\n" \
    "  uplo (str): 'U' for upper triangle, 'L' for lower triangle of R\n" \
    "  trans (str): 'N' for E*T' + T*E', 'T'/'C' for E'*T + T'*E\n" \
    "  n (int): Order of matrices R, E, T\n" \
    "  alpha (float): Scalar for R. If alpha=0, R need not be set\n" \
    "  beta (float): Scalar for products. If beta=0, E and T not referenced\n" \
    "  r (ndarray): n-by-n symmetric R (F-order), triangle per uplo\n" \
    "  e (ndarray): n-by-n upper triangular E (F-order)\n" \
    "  t (ndarray): n-by-n upper triangular T (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (r, info): Updated R (same triangle) and exit code (0=success, -i=param i invalid)"

#define DOC_MB01PD "Scale matrix to safe numerical range or undo scaling.\n" \
    "\n" \
    "Scales matrix so its norm is in [SMLNUM, BIGNUM] or undoes such scaling.\n" \
    "Preserves ANRM between scale and undo operations.\n" \
    "\n" \
    "Parameters:\n" \
    "  scun (str): 'S' to scale, 'U' to undo scaling\n" \
    "  type (str): Matrix storage type ('G', 'L', 'U', 'H', 'B', 'Q', 'Z')\n" \
    "  m (int): Number of rows\n" \
    "  n (int): Number of columns\n" \
    "  kl (int): Lower bandwidth (for B/Q/Z types)\n" \
    "  ku (int): Upper bandwidth (for B/Q/Z types)\n" \
    "  anrm (float): Norm of initial matrix (>= 0)\n" \
    "  nbl (int): Number of diagonal blocks (0 = no block structure)\n" \
    "  nrows (ndarray or None): Block sizes (int32)\n" \
    "  a (ndarray): Matrix array (column-major, shape (m,n))\n" \
    "\n" \
    "Returns:\n" \
    "  (a, info): Modified matrix and exit code"

#define DOC_MB01QD "Multiply matrix by scalar CTO/CFROM without overflow/underflow.\n" \
    "\n" \
    "Parameters:\n" \
    "  type (str): Matrix storage type ('G', 'L', 'U', 'H', 'B', 'Q', 'Z')\n" \
    "  m (int): Number of rows\n" \
    "  n (int): Number of columns\n" \
    "  kl (int): Lower bandwidth\n" \
    "  ku (int): Upper bandwidth\n" \
    "  cfrom (float): Denominator scalar\n" \
    "  cto (float): Numerator scalar\n" \
    "  a (ndarray): Matrix array (column-major, shape (m,n))\n" \
    "  nrows (ndarray, optional): Block sizes\n" \
    "\n" \
    "Returns:\n" \
    "  (a, info): Modified matrix and exit code"

#define DOC_MB01RB "Block triangular symmetric rank-k update (BLAS 3 version).\n" \
    "\n" \
    "Computes R = alpha*R + beta*op(A)*B or R = alpha*R + beta*B*op(A)\n" \
    "where only the specified triangle is computed using block algorithm.\n" \
    "\n" \
    "Parameters:\n" \
    "  side (str): 'L' for left (R = alpha*R + beta*op(A)*B)\n" \
    "              'R' for right (R = alpha*R + beta*B*op(A))\n" \
    "  uplo (str): 'U' for upper triangle, 'L' for lower triangle\n" \
    "  trans (str): 'N' for op(A)=A, 'T'/'C' for op(A)=A'\n" \
    "  m (int): Order of matrix R\n" \
    "  n (int): Dimension for product\n" \
    "  alpha (float): Scalar multiplier for R\n" \
    "  beta (float): Scalar multiplier for product\n" \
    "  r (ndarray): m-by-m matrix R (F-order), modified in place\n" \
    "  a (ndarray): Matrix A (F-order)\n" \
    "  b (ndarray): Matrix B (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (r, info): Updated R (triangle only) and exit code"

#define DOC_MB01RD "Symmetric rank-k matrix update with symmetric matrices.\n" \
    "\n" \
    "Computes R = alpha*R + beta*op(A)*X*op(A)' where R and X are symmetric.\n" \
    "\n" \
    "Parameters:\n" \
    "  uplo (str): 'U' for upper, 'L' for lower triangle stored\n" \
    "  trans (str): 'N' for op(A)=A, 'T'/'C' for op(A)=A'\n" \
    "  m (int): Order of R and rows of op(A)\n" \
    "  n (int): Order of X and columns of op(A)\n" \
    "  alpha (float): Scalar multiplier for R\n" \
    "  beta (float): Scalar multiplier for quadratic term\n" \
    "  r (ndarray): m-by-m symmetric matrix R (F-order)\n" \
    "  a (ndarray): General matrix A (F-order)\n" \
    "  x (ndarray): n-by-n symmetric matrix X (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (r, info): Updated R and exit code"

#define DOC_MB01RH "Symmetric matrix expression with Hessenberg matrix.\n" \
    "\n" \
    "Computes R := alpha*R + beta*op(H)*X*op(H)' where R and X are symmetric\n" \
    "and H is upper Hessenberg.\n" \
    "\n" \
    "Parameters:\n" \
    "  uplo (str): 'U' for upper, 'L' for lower triangle stored\n" \
    "  trans (str): 'N' for op(H)=H, 'T'/'C' for op(H)=H'\n" \
    "  n (int): Order of matrices R, H, X\n" \
    "  alpha (float): Scalar multiplier for R\n" \
    "  beta (float): Scalar multiplier for quadratic form\n" \
    "  r (ndarray): n-by-n symmetric matrix R (F-order)\n" \
    "  h (ndarray): n-by-n upper Hessenberg matrix H (F-order)\n" \
    "  x (ndarray): n-by-n symmetric matrix X (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (r, info): Updated R and exit code"

#define DOC_MB01RT "Symmetric matrix expression with upper triangular matrix.\n" \
    "\n" \
    "Computes R := alpha*R + beta*op(E)*X*op(E)' where R and X are symmetric\n" \
    "and E is upper triangular.\n" \
    "\n" \
    "Parameters:\n" \
    "  uplo (str): 'U' for upper, 'L' for lower triangle stored\n" \
    "  trans (str): 'N' for op(E)=E, 'T'/'C' for op(E)=E'\n" \
    "  n (int): Order of matrices R, E, X\n" \
    "  alpha (float): Scalar multiplier for R\n" \
    "  beta (float): Scalar multiplier for quadratic form\n" \
    "  r (ndarray): n-by-n symmetric matrix R (F-order)\n" \
    "  e (ndarray): n-by-n upper triangular matrix E (F-order)\n" \
    "  x (ndarray): n-by-n symmetric matrix X (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (r, info): Updated R and exit code"

#define DOC_MB01RU "Symmetric matrix update: R = alpha*R + beta*op(A)*X*op(A)'.\n" \
    "\n" \
    "Computes R := alpha*R + beta*op(A)*X*op(A)' where R and X are symmetric.\n" \
    "\n" \
    "Parameters:\n" \
    "  uplo (str): 'U' for upper triangle, 'L' for lower triangle\n" \
    "  trans (str): 'N' for op(A)=A, 'T'/'C' for op(A)=A'\n" \
    "  m (int): Order of R, rows of op(A)\n" \
    "  n (int): Order of X, columns of op(A)\n" \
    "  alpha (float): Scalar for R\n" \
    "  beta (float): Scalar for product\n" \
    "  r (ndarray): m-by-m symmetric R (F-order), modified in place\n" \
    "  a (ndarray): Matrix A (F-order)\n" \
    "  x (ndarray): n-by-n symmetric X (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (r, info): Updated R (triangle only) and exit code"

#define DOC_MB01RW "Symmetric matrix transformation (BLAS 2 version).\n" \
    "\n" \
    "Computes A := op(Z)*A*op(Z)' where A is symmetric.\n" \
    "\n" \
    "Parameters:\n" \
    "  uplo (str): 'U' for upper, 'L' for lower triangle stored\n" \
    "  trans (str): 'N' for op(Z)=Z, 'T' for op(Z)=Z'\n" \
    "  m (int): Order of result\n" \
    "  n (int): Order of input A\n" \
    "  a (ndarray): max(m,n)-by-max(m,n) symmetric matrix A (F-order)\n" \
    "  z (ndarray): Transformation matrix Z (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, info): Transformed A and exit code"

#define DOC_MB01RX "Triangular symmetric rank-k update.\n" \
    "\n" \
    "Computes R = alpha*R + beta*op(A)*B or R = alpha*R + beta*B*op(A)\n" \
    "where only the specified triangle is computed.\n" \
    "\n" \
    "Parameters:\n" \
    "  side (str): 'L' for left (R = alpha*R + beta*op(A)*B)\n" \
    "              'R' for right (R = alpha*R + beta*B*op(A))\n" \
    "  uplo (str): 'U' for upper triangle, 'L' for lower triangle\n" \
    "  trans (str): 'N' for op(A)=A, 'T'/'C' for op(A)=A'\n" \
    "  m (int): Order of matrix R\n" \
    "  n (int): Dimension for product\n" \
    "  alpha (float): Scalar multiplier for R\n" \
    "  beta (float): Scalar multiplier for product\n" \
    "  r (ndarray): m-by-m matrix R (F-order), modified in place\n" \
    "  a (ndarray): Matrix A (F-order)\n" \
    "  b (ndarray): Matrix B (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (r, info): Updated R (triangle only) and exit code"

#define DOC_MB01RY "Hessenberg matrix product: R = alpha*R + beta*op(H)*B or beta*B*op(H).\n" \
    "\n" \
    "Computes upper or lower triangle of the product.\n" \
    "\n" \
    "Parameters:\n" \
    "  side (str): 'L' for op(H)*B, 'R' for B*op(H)\n" \
    "  uplo (str): 'U' for upper triangle, 'L' for lower triangle\n" \
    "  trans (str): 'N' for op(H)=H, 'T'/'C' for op(H)=H'\n" \
    "  m (int): Order of matrices\n" \
    "  alpha (float): Scalar for R\n" \
    "  beta (float): Scalar for product\n" \
    "  r (ndarray): m-by-m matrix R (F-order), modified in place\n" \
    "  h (ndarray): m-by-m upper Hessenberg H (F-order)\n" \
    "  b (ndarray): m-by-m matrix B (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (r, info): Updated R (triangle only) and exit code"

#define DOC_MB01SD "Scale rows or columns of a matrix by a diagonal matrix.\n" \
    "\n" \
    "Computes one of:\n" \
    "  A := diag(R) * A        (jobs='R', row scaling)\n" \
    "  A := A * diag(C)        (jobs='C', column scaling)\n" \
    "  A := diag(R) * A * diag(C)  (jobs='B', both)\n" \
    "\n" \
    "Parameters:\n" \
    "  jobs (str): 'R' for row, 'C' for column, 'B' for both\n" \
    "  a (ndarray): M-by-N matrix A (F-order), modified in-place\n" \
    "  r (ndarray): Row scale factors, dimension M (not used if jobs='C')\n" \
    "  c (ndarray): Column scale factors, dimension N (not used if jobs='R')\n" \
    "\n" \
    "Returns:\n" \
    "  a: The scaled matrix"

#define DOC_MB01SS "Scale a symmetric matrix using diagonal scaling factors.\n" \
    "\n" \
    "Computes one of:\n" \
    "  A := diag(D)*A*diag(D)           (jobs='D', scaling with D)\n" \
    "  A := inv(diag(D))*A*inv(diag(D)) (jobs='I', scaling with inv(D))\n" \
    "\n" \
    "where A is an N-by-N symmetric matrix stored in upper or lower triangular form.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobs (str): 'D' for scaling with D, 'I' for scaling with inv(D)\n" \
    "  uplo (str): 'U' for upper triangle stored, 'L' for lower triangle stored\n" \
    "  a (ndarray): N-by-N symmetric matrix A (F-order), modified in-place\n" \
    "  d (ndarray): Diagonal scaling factors, dimension N\n" \
    "\n" \
    "Returns:\n" \
    "  a: The scaled matrix (same triangle as input)"

#define DOC_MB01TD "Product of upper quasi-triangular matrices B := A * B.\n" \
    "\n" \
    "Computes A * B where A and B are upper quasi-triangular matrices\n" \
    "(block upper triangular with 1x1 or 2x2 diagonal blocks) with the\n" \
    "same structure. Result is returned in B.\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): N-by-N upper quasi-triangular matrix A (F-order)\n" \
    "  b (ndarray): N-by-N upper quasi-triangular matrix B (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (b, info): Product A*B in B, exit code (0=success, 1=structure mismatch)"

#define DOC_MB01UD "Hessenberg matrix product: B = alpha*op(H)*A or alpha*A*op(H).\n" \
    "\n" \
    "Parameters:\n" \
    "  side (str): 'L' for B=alpha*op(H)*A, 'R' for B=alpha*A*op(H)\n" \
    "  trans (str): 'N' for op(H)=H, 'T'/'C' for op(H)=H'\n" \
    "  m (int): Rows of A and B\n" \
    "  n (int): Columns of A and B\n" \
    "  alpha (float): Scalar multiplier\n" \
    "  h (ndarray): Upper Hessenberg matrix (F-order)\n" \
    "  a (ndarray): m-by-n matrix A (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (b, info): Output matrix B and exit code"

#define DOC_MB01UW "In-place Hessenberg matrix product: A := alpha*op(H)*A or alpha*A*op(H).\n" \
    "\n" \
    "Parameters:\n" \
    "  side (str): 'L' for A:=alpha*op(H)*A, 'R' for A:=alpha*A*op(H)\n" \
    "  trans (str): 'N' for op(H)=H, 'T'/'C' for op(H)=H'\n" \
    "  m (int): Rows of A\n" \
    "  n (int): Columns of A\n" \
    "  alpha (float): Scalar multiplier (alpha=0 sets A to zero)\n" \
    "  h (ndarray): Upper Hessenberg matrix (F-order)\n" \
    "  a (ndarray): m-by-n matrix A (F-order), modified in-place\n" \
    "\n" \
    "Returns:\n" \
    "  (a, info): Modified matrix A and exit code"

#define DOC_MB01UX "In-place quasi-triangular matrix product: A := alpha*op(T)*A or alpha*A*op(T).\n" \
    "\n" \
    "Computes A := alpha*op(T)*A (side='L') or A := alpha*A*op(T) (side='R')\n" \
    "where T is a quasi-triangular matrix (upper or lower) and op(T) is T or T'.\n" \
    "\n" \
    "Parameters:\n" \
    "  side (str): 'L' for A:=alpha*op(T)*A, 'R' for A:=alpha*A*op(T)\n" \
    "  uplo (str): 'U' for upper, 'L' for lower quasi-triangular T\n" \
    "  trans (str): 'N' for op(T)=T, 'T'/'C' for op(T)=T'\n" \
    "  m (int): Rows of A\n" \
    "  n (int): Columns of A\n" \
    "  alpha (float): Scalar multiplier (alpha=0 sets A to zero)\n" \
    "  t (ndarray): Quasi-triangular matrix T (F-order)\n" \
    "  a (ndarray): m-by-n matrix A (F-order), modified in-place\n" \
    "\n" \
    "Returns:\n" \
    "  (a, info): Modified matrix A and exit code"

#define DOC_MB01UY "Compute matrix product T := alpha*op(T)*A or T := alpha*A*op(T).\n" \
    "\n" \
    "Parameters:\n" \
    "  side (str): 'L' or 'R' - triangular matrix position\n" \
    "  uplo (str): 'U' or 'L' - upper/lower triangular\n" \
    "  trans (str): 'N'/'T'/'C' - transpose option\n" \
    "  m (int): Number of rows of A\n" \
    "  n (int): Number of columns of A\n" \
    "  alpha (float): Scalar multiplier\n" \
    "  t (ndarray): Triangular matrix (F-order), overwritten with result\n" \
    "  a (ndarray): Matrix A (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (t, info): Result matrix and exit code"

#define DOC_MB01UZ "Compute complex matrix product T := alpha*op(T)*A or T := alpha*A*op(T).\n" \
    "\n" \
    "Computes T := alpha*op(T)*A (side='L') or T := alpha*A*op(T) (side='R')\n" \
    "where T is a triangular complex matrix and op(T) is T, T', or conj(T').\n" \
    "\n" \
    "Parameters:\n" \
    "  side (str): 'L' for T:=alpha*op(T)*A, 'R' for T:=alpha*A*op(T)\n" \
    "  uplo (str): 'U' for upper triangular, 'L' for lower triangular T\n" \
    "  trans (str): 'N' for op(T)=T, 'T' for op(T)=T', 'C' for op(T)=conj(T')\n" \
    "  m (int): Rows of A\n" \
    "  n (int): Columns of A\n" \
    "  alpha (complex): Scalar multiplier (alpha=0 sets T to zero)\n" \
    "  t (ndarray): Triangular complex matrix T (F-order), overwritten with result\n" \
    "  a (ndarray): m-by-n complex matrix A (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (t, info): Result matrix T (m-by-n) and exit code"

#define DOC_MB01WD "Residuals of Lyapunov or Stein equations for Cholesky factored solutions.\n" \
    "\n" \
    "Computes:\n" \
    "  R = alpha*(op(A)'*op(T)'*op(T) + op(T)'*op(T)*op(A)) + beta*R (DICO='C')\n" \
    "or\n" \
    "  R = alpha*(op(A)'*op(T)'*op(T)*op(A) - op(T)'*op(T)) + beta*R (DICO='D')\n" \
    "\n" \
    "where R is symmetric, T is triangular, A is general or Hessenberg,\n" \
    "and op(M) = M or M'.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' continuous-time, 'D' discrete-time\n" \
    "  uplo (str): 'U' upper triangular parts, 'L' lower triangular parts\n" \
    "  trans (str): 'N' for op(M)=M, 'T'/'C' for op(M)=M'\n" \
    "  hess (str): 'F' A is full, 'H' A is Hessenberg\n" \
    "  n (int): Order of R, A, T\n" \
    "  alpha (float): Scalar. If 0, A and T not referenced.\n" \
    "  beta (float): Scalar. If 0, R need not be set on entry.\n" \
    "  r (ndarray): Symmetric matrix R (F-order), overwritten with result\n" \
    "  a (ndarray): Matrix A (F-order), overwritten with intermediate product\n" \
    "  t (ndarray): Triangular matrix T (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, r, info): Modified A, result R, and exit code (0=success)"

#define DOC_MB01XD "Compute U'*U or L*L' for triangular matrix (block algorithm).\n" \
    "\n" \
    "Computes the matrix product U'*U or L*L', where U and L are upper and\n" \
    "lower triangular matrices respectively. Result overwrites input triangle.\n" \
    "Uses BLAS 3 block algorithm for large matrices, otherwise MB01XY.\n" \
    "This is a counterpart of LAPACK DLAUUM which computes U*U' or L'*L.\n" \
    "\n" \
    "Parameters:\n" \
    "  uplo (str): 'U' for U'*U (upper triangular), 'L' for L*L' (lower triangular)\n" \
    "  a (ndarray): N-by-N triangular matrix (F-order), overwritten with result\n" \
    "\n" \
    "Returns:\n" \
    "  (a, info): Result matrix (same triangle) and exit code (0=success)"

#define DOC_MB01XY "Compute U'*U or L*L' for triangular matrix.\n" \
    "\n" \
    "Computes the matrix product U'*U or L*L', where U and L are upper and\n" \
    "lower triangular matrices respectively. Result overwrites input triangle.\n" \
    "This is a counterpart of LAPACK DLAUU2 which computes U*U' or L'*L.\n" \
    "\n" \
    "Parameters:\n" \
    "  uplo (str): 'U' for U'*U (upper triangular), 'L' for L*L' (lower triangular)\n" \
    "  a (ndarray): N-by-N triangular matrix (F-order), overwritten with result\n" \
    "\n" \
    "Returns:\n" \
    "  (a, info): Result matrix (same triangle) and exit code (0=success)"

#define DOC_MB01YD "Symmetric rank k operation with banded matrix.\n" \
    "\n" \
    "Performs C := alpha*op(A)*op(A)' + beta*C where C is n-by-n symmetric\n" \
    "and A has L nonzero codiagonals (UPLO='U': L subdiagonals, UPLO='L': L superdiagonals).\n" \
    "This is a specialization of DSYRK for banded matrices.\n" \
    "\n" \
    "Parameters:\n" \
    "  uplo (str): 'U' upper triangle of C, 'L' lower triangle\n" \
    "  trans (str): 'N' for op(A)=A, 'T'/'C' for op(A)=A'\n" \
    "  n (int): Order of C\n" \
    "  k (int): Columns of op(A)\n" \
    "  l (int): Number of nonzero codiagonals in A\n" \
    "  alpha (float): Scalar for A*A' term\n" \
    "  beta (float): Scalar for C term\n" \
    "  a (ndarray): Matrix A (F-order), banded structure per UPLO\n" \
    "  c (ndarray): Symmetric matrix C (F-order), updated in place\n" \
    "\n" \
    "Returns:\n" \
    "  (c, info): Updated C and exit code (0=success)"

#define DOC_MB01ZD "Hessenberg-triangular matrix product.\n" \
    "\n" \
    "Computes H := alpha*op(T)*H (SIDE='L') or H := alpha*H*op(T) (SIDE='R'),\n" \
    "where H is m-by-n upper/lower Hessenberg-like matrix with L nonzero\n" \
    "subdiagonals/superdiagonals, T is triangular, and op(T) is T or T'.\n" \
    "\n" \
    "Parameters:\n" \
    "  side (str): 'L' for left multiply, 'R' for right multiply\n" \
    "  uplo (str): 'U' for upper triangular T and upper Hessenberg H\n" \
    "              'L' for lower triangular T and lower Hessenberg H\n" \
    "  trans (str): 'N' for op(T)=T, 'T'/'C' for op(T)=T'\n" \
    "  diag (str): 'U' for unit triangular T, 'N' for non-unit\n" \
    "  m (int): Rows of H\n" \
    "  n (int): Columns of H\n" \
    "  l (int): Number of nonzero sub/superdiagonals in H\n" \
    "  alpha (float): Scalar multiplier\n" \
    "  t (ndarray): Triangular matrix T (F-order)\n" \
    "  h (ndarray): Hessenberg-like matrix H (F-order), overwritten with result\n" \
    "\n" \
    "Returns:\n" \
    "  (h, info): Result matrix and exit code (0=success)"

#define DOC_MB02CD "Cholesky factorization of positive definite block Toeplitz matrix.\n" \
    "\n" \
    "Computes the Cholesky factor and the generator and/or the Cholesky factor\n" \
    "of the inverse of a symmetric positive definite (s.p.d.) block Toeplitz\n" \
    "matrix T, defined by either its first block row or column.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'G' = generator only\n" \
    "             'R' = generator and Cholesky factor of T\n" \
    "             'L' = generator and Cholesky factor of inv(T)\n" \
    "             'A' = all (generator, L, R)\n" \
    "             'O' = Cholesky factor of T only\n" \
    "  typet (str): 'R' = T given by first block row (R upper, L lower)\n" \
    "               'C' = T given by first block column (R lower, L upper)\n" \
    "  k (int): Block size, k >= 0\n" \
    "  n (int): Number of blocks, n >= 0\n" \
    "  t (ndarray): First block row (k x n*k) or column (n*k x k), F-order\n" \
    "\n" \
    "Returns:\n" \
    "  (g, r, l, cs, info): Generator, Cholesky of T, Cholesky of inv(T),\n" \
    "    transformation info, exit code. Returns None for unrequested outputs.\n" \
    "    info=1 means matrix is not positive definite."

#define DOC_MB02CU "Bring first part of generator to proper form (block or rank-deficient).\n" \
    "\n" \
    "Brings the first part of a generator in proper form using orthogonal\n" \
    "transformations and modified hyperbolic rotations. Used for computing\n" \
    "Cholesky factor of symmetric positive semi-definite block Toeplitz\n" \
    "matrices and for solving associated linear systems.\n" \
    "\n" \
    "Parameters:\n" \
    "  typeg (str): 'D' = column oriented, rank deficient\n" \
    "               'C' = column oriented, not rank deficient\n" \
    "               'R' = row oriented, not rank deficient\n" \
    "  k (int): Number of rows in A1 to process, k >= 0\n" \
    "  p (int): Columns of positive generator, p >= k\n" \
    "  q (int): Columns in B. If typeg='D': q >= k; else q >= 0\n" \
    "  nb (int): Block size for typeg='C' or 'R'. nb <= 0 means unblocked.\n" \
    "  a1 (ndarray): K-by-K triangular matrix (F-order)\n" \
    "  a2 (ndarray): Additional part of positive generator (F-order)\n" \
    "  b (ndarray): Negative generator (F-order)\n" \
    "  tol (float, optional): Tolerance for rank determination (typeg='D' only)\n" \
    "\n" \
    "Returns:\n" \
    "  (a1, a2, b, rnk, ipvt, cs, info): Transformed matrices, rank,\n" \
    "    pivot indices, rotation info, and exit code.\n" \
    "    info=1 means matrix is not numerically positive definite."

#define DOC_MB02CV "Apply MB02CU transformations to other generator columns/rows.\n" \
    "\n" \
    "Applies the Householder transformations and hyperbolic rotations\n" \
    "computed by MB02CU to other columns/rows of the generator in F1, F2, G.\n" \
    "\n" \
    "Parameters:\n" \
    "  typeg (str): 'D' = column rank-deficient, 'C' = column, 'R' = row\n" \
    "  strucg (str): 'T' = triangular structure, 'N' = no structure\n" \
    "  k (int): Rows in A1 to process, k >= 0\n" \
    "  n (int): Rows in F1 (typeg='D','C') or columns (typeg='R'), n >= 0\n" \
    "  p (int): Columns of positive generator, p >= k\n" \
    "  q (int): Columns in B, q >= k for typeg='D'\n" \
    "  nb (int): Block size for typeg='C','R'\n" \
    "  rnk (int): Rank from MB02CU for typeg='D'\n" \
    "  a1 (ndarray): K-by-K matrix from MB02CU (typeg='D' only)\n" \
    "  a2 (ndarray): Matrix A2 from MB02CU\n" \
    "  b (ndarray): Matrix B from MB02CU\n" \
    "  f1 (ndarray): First part of positive generator\n" \
    "  f2 (ndarray): Second part of positive generator\n" \
    "  g (ndarray): Negative generator part\n" \
    "  cs (ndarray): Rotation/Householder info from MB02CU\n" \
    "\n" \
    "Returns:\n" \
    "  (f1, f2, g, info): Transformed generators and exit code"

#define DOC_MB02CX "Bring first blocks of generator to proper form.\n" \
    "\n" \
    "Brings the first blocks of a generator into proper form using\n" \
    "QR/LQ decomposition, Householder transformations, and modified\n" \
    "hyperbolic rotations. Used for Toeplitz solvers.\n" \
    "\n" \
    "Parameters:\n" \
    "  typet (str): 'R' = row-wise, 'C' = column-wise\n" \
    "  p (int): Rows/columns in A (positive generator), p >= 0\n" \
    "  q (int): Rows/columns in B (negative generator), q >= 0\n" \
    "  k (int): Columns/rows to process, p >= k >= 0\n" \
    "  a (ndarray): Positive generator (p x k) or (k x p), F-order\n" \
    "  b (ndarray): Negative generator (q x k) or (k x q), F-order\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, cs, info): Transformed generators, rotation info, exit code\n" \
    "  info=1 means matrix is not positive definite"

#define DOC_MB02CY "Apply hyperbolic transformations to generator columns/rows.\n" \
    "\n" \
    "Applies transformations computed by MB02CX on other columns/rows\n" \
    "of the generator contained in arrays A and B.\n" \
    "\n" \
    "Parameters:\n" \
    "  typet (str): 'R' = row-wise, 'C' = column-wise\n" \
    "  strucg (str): 'T' = triangular structure, 'N' = no special structure\n" \
    "  p (int): Rows/columns in A (positive generator), p >= 0\n" \
    "  q (int): Rows/columns in B (negative generator), q >= 0\n" \
    "  n (int): Columns/rows to process, n >= 0\n" \
    "  k (int): Columns/rows in H, p >= k >= 0\n" \
    "  a (ndarray): Positive generator (p x n) or (n x p), F-order\n" \
    "  b (ndarray): Negative generator (q x n) or (n x q), F-order\n" \
    "  h (ndarray): Householder info from MB02CX (q x k) or (k x q), F-order\n" \
    "  cs (ndarray): Rotation/Householder info from MB02CX\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, info): Transformed generators and exit code"

#define DOC_MB02DD "Update Cholesky factorization of positive definite block Toeplitz matrix.\n" \
    "\n" \
    "Updates the Cholesky factor and the generator and/or the Cholesky factor\n" \
    "of the inverse of a symmetric positive definite (s.p.d.) block Toeplitz\n" \
    "matrix T, given info from MB02CD and additional blocks.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'R' = update generator G and new columns/rows for R\n" \
    "             'A' = update G, new columns/rows for R and L\n" \
    "             'O' = only new columns/rows for R\n" \
    "  typet (str): 'R' = T given by first block row (R upper, L lower)\n" \
    "               'C' = T given by first block column (R lower, L upper)\n" \
    "  k (int): Block size, k >= 0\n" \
    "  m (int): Number of additional blocks in TA, m >= 0\n" \
    "  n (int): Number of initial blocks in T, n >= 0\n" \
    "  ta (ndarray): Additional blocks (k x m*k) or (m*k x k), F-order\n" \
    "  t (ndarray): Transformation info from MB02CD, F-order\n" \
    "  g (ndarray): Generator from MB02CD, extended for output\n" \
    "  r (ndarray): Last block column/row of previous R, extended for output\n" \
    "  cs (ndarray): Transformation info from MB02CD, extended for output\n" \
    "\n" \
    "Returns:\n" \
    "  (ta, g, r, l, cs, info): Updated transformation info, generator,\n" \
    "    Cholesky of T, Cholesky of inv(T), transformation info, exit code.\n" \
    "    info=1 means matrix is not positive definite."

#define DOC_MB02ED "Solve symmetric positive definite block Toeplitz system.\n" \
    "\n" \
    "Solves T*X = B or X*T = B where T is a symmetric positive definite\n" \
    "block Toeplitz matrix defined by its first block row or column.\n" \
    "\n" \
    "Parameters:\n" \
    "  typet (str): 'R' = first block row, solve X*T = B\n" \
    "               'C' = first block column, solve T*X = B\n" \
    "  k (int): Block size, k >= 0\n" \
    "  n (int): Number of blocks, n >= 0\n" \
    "  nrhs (int): Number of right-hand sides, nrhs >= 0\n" \
    "  t (ndarray): Block Toeplitz data (k x n*k) or (n*k x k), F-order\n" \
    "  b (ndarray): Right-hand side (nrhs x n*k) or (n*k x nrhs), F-order\n" \
    "\n" \
    "Returns:\n" \
    "  (x, t_out, info): Solution matrix, modified T, exit code\n" \
    "  info=1 means matrix is not positive definite"

#define DOC_MB02FD "Incomplete Cholesky factor of positive definite block Toeplitz matrix.\n" \
    "\n" \
    "Computes the incomplete Cholesky (ICC) factor of a symmetric positive\n" \
    "definite block Toeplitz matrix T. Supports incremental computation.\n" \
    "\n" \
    "Parameters:\n" \
    "  typet (str): 'R' = first block row, R is upper trapezoidal\n" \
    "               'C' = first block column, R is lower trapezoidal\n" \
    "  k (int): Block size, k >= 0\n" \
    "  n (int): Number of blocks, n >= 0\n" \
    "  p (int): Previously computed block rows/columns, 0 <= p <= n\n" \
    "  s (int): Block rows/columns to compute, 0 <= s <= n-p\n" \
    "  t (ndarray): Block Toeplitz data, F-order\n" \
    "               If P=0: first block row/column of T\n" \
    "               If P>0: negative generator from previous call\n" \
    "\n" \
    "Returns:\n" \
    "  (r, info): ICC factor R (upper/lower trapezoidal), exit code\n" \
    "  info=1 means matrix is not positive definite"

#define DOC_MB02GD "Cholesky factorization of banded symmetric positive definite block Toeplitz matrix.\n" \
    "\n" \
    "Computes the Cholesky factor of a banded s.p.d. block Toeplitz matrix.\n" \
    "Supports incremental block-by-block computation.\n" \
    "\n" \
    "Parameters:\n" \
    "  typet (str): 'R' = first block row, factor is upper triangular\n" \
    "               'C' = first block column, factor is lower triangular\n" \
    "  triu (str): 'N' = last block has no special structure\n" \
    "              'T' = last block is triangular\n" \
    "  k (int): Block size, k >= 0\n" \
    "  n (int): Number of blocks. n >= 1 for triu='N', n >= 2 for triu='T'\n" \
    "  nl (int): Lower block bandwidth (nl+1 nonzero blocks in first column)\n" \
    "  p (int): Previously computed block rows/columns, 0 <= p <= n\n" \
    "  s (int): Block rows/columns to compute, 0 <= s <= n-p\n" \
    "  t (ndarray): Block Toeplitz data (k x (nl+1)*k) or ((nl+1)*k x k), F-order\n" \
    "\n" \
    "Returns:\n" \
    "  (rb, info): Cholesky factor in banded storage format, exit code\n" \
    "  info=1 means matrix is not positive definite"

#define DOC_MB02HD "Cholesky factorization of T'T for banded block Toeplitz matrix.\n" \
    "\n" \
    "Computes R*R' = T'*T where T is a banded K*M-by-L*N block Toeplitz matrix.\n" \
    "The lower triangular factor R is returned in band storage format.\n" \
    "Supports incremental block-by-block computation.\n" \
    "\n" \
    "Parameters:\n" \
    "  triu (str): 'N' = TC/TR have no special structure\n" \
    "              'T' = last blocks are triangular\n" \
    "  k (int): Number of rows in blocks of T, k >= 0\n" \
    "  l (int): Number of columns in blocks of T, l >= 0\n" \
    "  m (int): Number of blocks in first block column of T, m >= 1\n" \
    "  ml (int): Lower block bandwidth, 0 <= ml < m\n" \
    "  n (int): Number of blocks in first block row of T, n >= 1\n" \
    "  nu (int): Upper block bandwidth, 0 <= nu < n\n" \
    "  p (int): Previously computed block columns of R, p >= 0\n" \
    "  s (int): Block columns of R to compute, s >= 0\n" \
    "  tc (ndarray): First block column data ((ml+1)*k x l, F-order)\n" \
    "  tr (ndarray): Blocks 2 to nu+1 of first block row (k x nu*l, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (rb, info): Lower triangular factor R in band storage, exit code\n" \
    "  info=1 means full rank condition not satisfied"

#define DOC_MB02ID "Solve over/underdetermined linear systems with block Toeplitz matrix.\n" \
    "\n" \
    "Solves overdetermined or underdetermined real linear systems involving an\n" \
    "M*K-by-N*L block Toeplitz matrix T specified by its first block column and row.\n" \
    "Assumes T has full rank.\n" \
    "\n" \
    "Options:\n" \
    "  JOB='O': Least squares - minimize ||B - T*X||\n" \
    "  JOB='U': Minimum norm - solve T'*X = C\n" \
    "  JOB='A': Solve both problems\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'O' overdetermined, 'U' underdetermined, 'A' both\n" \
    "  k (int): Number of rows in blocks of T, k >= 0\n" \
    "  l (int): Number of columns in blocks of T, l >= 0\n" \
    "  m (int): Number of blocks in first block column of T, m >= 0\n" \
    "  n (int): Number of blocks in first block row of T, 0 <= n*l <= m*k\n" \
    "  tc (ndarray): First block column of T (M*K x L, F-order)\n" \
    "  tr (ndarray): Blocks 2 to N of first block row (K x (N-1)*L, F-order)\n" \
    "  b (ndarray): RHS matrix B for overdetermined system (M*K x RB, F-order)\n" \
    "  c (ndarray): RHS matrix C for underdetermined system (N*L x RC, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (b_out, c_out, info):\n" \
    "    - b_out: Overdetermined solution (N*L x RB) if job='O' or 'A'\n" \
    "    - c_out: Underdetermined solution (M*K x RC) if job='U' or 'A'\n" \
    "    - info: 0=success, <0=invalid arg, 1=T not full rank"

#define DOC_MB02JD "Full QR factorization of a block Toeplitz matrix of full rank.\n" \
    "\n" \
    "Computes lower triangular R and orthogonal Q such that T = Q * R^T,\n" \
    "where T is a K*M-by-L*N block Toeplitz matrix with blocks of size (K,L).\n" \
    "Assumes first MIN(M*K, N*L) columns of T have full rank.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'Q' = compute Q and R, 'R' = only compute R\n" \
    "  k (int): Number of rows in one block of T, k >= 0\n" \
    "  l (int): Number of columns in one block of T, l >= 0\n" \
    "  m (int): Number of blocks in one block column of T, m >= 0\n" \
    "  n (int): Number of blocks in one block row of T, n >= 0\n" \
    "  p (int): Previously computed block columns of R, p >= 0\n" \
    "  s (int): Number of block columns of R to compute, s >= 0\n" \
    "  tc (ndarray): First block column of T (M*K x L, F-order)\n" \
    "  tr (ndarray): First block row without leading block (K x (N-1)*L, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (q, r, info): Factor Q (M*K x cols), lower triangular factor R, exit code\n" \
    "  info=1 means full rank condition numerically violated"

#define DOC_MB02JX "Low rank QR factorization with column pivoting of a block Toeplitz matrix.\n" \
    "\n" \
    "Computes T*P = Q*R^T where R is lower trapezoidal, P is a block permutation\n" \
    "matrix, and Q^T*Q = I. The number of columns in R (RNK) equals the numerical\n" \
    "rank of T. Only columns in the same block of T are permuted.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'Q' = compute Q and R, 'R' = only compute R\n" \
    "  k (int): Number of rows in one block of T, k >= 0\n" \
    "  l (int): Number of columns in one block of T, l >= 0\n" \
    "  m (int): Number of blocks in one block column of T, m >= 0\n" \
    "  n (int): Number of blocks in one block row of T, n >= 0\n" \
    "  tc (ndarray): First block column of T (M*K x L, F-order)\n" \
    "  tr (ndarray): First block row without leading block (K x (N-1)*L, F-order)\n" \
    "  tol1 (float): Diagonal tolerance (< 0 for default)\n" \
    "  tol2 (float): Offdiagonal tolerance (< 0 for default)\n" \
    "\n" \
    "Returns:\n" \
    "  (rnk, q, r, jpvt, info):\n" \
    "    - rnk: Numerical rank of T\n" \
    "    - q: Factor Q (M*K x RNK, F-order) if job='Q'\n" \
    "    - r: Lower trapezoidal factor R (N*L x RNK, F-order)\n" \
    "    - jpvt: Column permutation (jpvt[j]=k means col j of T*P was col k of T)\n" \
    "    - info: 0=success, 1=generator indefinite, 2=columns not linearly dependent"

#define DOC_MB02KD "Block Toeplitz matrix-matrix product.\n" \
    "\n" \
    "Computes C = alpha*op(T)*B + beta*C where T is a block Toeplitz matrix\n" \
    "specified by its first block column TC and first block row TR.\n" \
    "\n" \
    "Parameters:\n" \
    "  ldblk (str): 'C' = T(1,1) in first block of TC, 'R' = in first block of TR\n" \
    "  trans (str): 'N' = op(T)=T, 'T'/'C' = op(T)=T'\n" \
    "  k (int): Number of rows in blocks of T, k >= 0\n" \
    "  l (int): Number of columns in blocks of T, l >= 0\n" \
    "  m (int): Number of blocks in first block column, m >= 0\n" \
    "  n (int): Number of blocks in first block row, n >= 0\n" \
    "  r (int): Number of columns in B and C, r >= 0\n" \
    "  alpha (float): Scalar multiplier\n" \
    "  beta (float): Scalar for C. If zero, C not referenced on entry\n" \
    "  tc (ndarray): First block column of T (M*K x L or (M-1)*K x L, F-order)\n" \
    "  tr (ndarray): First block row of T (K x (N-1)*L or K x N*L, F-order)\n" \
    "  b (ndarray): Input matrix B (N*L x R or M*K x R, F-order)\n" \
    "  c (ndarray, optional): Input matrix C (for beta != 0)\n" \
    "\n" \
    "Returns:\n" \
    "  (c, info): Result matrix C = alpha*op(T)*B + beta*C, exit code"

#define DOC_MB02MD "Total Least Squares solution using SVD approach.\n" \
    "\n" \
    "Solves the TLS problem AX = B where both A and B are inaccurate.\n" \
    "Computes minimum norm solution for overdetermined, determined,\n" \
    "or underdetermined systems.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'R' compute RANK only, 'T' compute TOL only,\n" \
    "             'B' compute both, 'N' compute neither\n" \
    "  m (int): Number of rows in A and B, m >= 0\n" \
    "  n (int): Number of columns in A, n >= 0\n" \
    "  l (int): Number of columns in B, l >= 0\n" \
    "  c (ndarray): Matrix [A|B] (m x (n+l), F-order)\n" \
    "  tol (float): Tolerance (or sdev for job='T'/'B')\n" \
    "  rank (int, optional): Input rank for job='T'/'N'\n" \
    "\n" \
    "Returns:\n" \
    "  (c_out, s, x, rank, rcond, iwarn, info):\n" \
    "    - c_out: Transformed right singular vectors\n" \
    "    - s: Singular values (descending order)\n" \
    "    - x: TLS solution (n x l)\n" \
    "    - rank: Computed/adjusted rank\n" \
    "    - rcond: Reciprocal condition number of F\n" \
    "    - iwarn: 0=ok, 1=multiplicity, 2=F singular\n" \
    "    - info: Exit code"

#define DOC_MB02ND "Total Least Squares solution using Partial SVD approach.\n" \
    "\n" \
    "Solves the TLS problem AX = B using partial singular value decomposition\n" \
    "where both A and B are inaccurate. More efficient than full SVD for\n" \
    "large problems when only a subset of singular values is needed.\n" \
    "\n" \
    "Parameters:\n" \
    "  m (int): Number of rows in A and B, m >= 0\n" \
    "  n (int): Number of columns in A, n >= 0\n" \
    "  l (int): Number of columns in B, l >= 0\n" \
    "  rank (int): If < 0, compute rank. Otherwise, specify rank. rank <= min(m,n)\n" \
    "  theta (float): If rank < 0, threshold for singular values (>= 0).\n" \
    "                 If rank >= 0 and theta < 0, compute theta.\n" \
    "  c (ndarray): Matrix [A|B] (max(m,n+l) x (n+l), F-order)\n" \
    "  tol (float): Tolerance for singular value multiplicity (0 for default)\n" \
    "  reltol (float): Relative tolerance for bisection (0 for default)\n" \
    "\n" \
    "Returns:\n" \
    "  (x, rank, theta, q, inul, iwarn, info):\n" \
    "    - x: TLS solution (n x l)\n" \
    "    - rank: Computed/adjusted rank\n" \
    "    - theta: Computed singular value bound\n" \
    "    - q: Bidiagonal matrix elements (2*p-1 where p=min(m,n+l))\n" \
    "    - inul: Boolean list indicating singular subspace columns\n" \
    "    - iwarn: 0=ok, 1=multiplicity, 2=F singular\n" \
    "    - info: Exit code (0=success, 1=max iterations, 2=rank too large)"

#define DOC_MB02NY "Separate zero singular value of bidiagonal submatrix.\n" \
    "\n" \
    "Annihilates superdiagonal elements E(i-1) and/or E(i) when diagonal\n" \
    "entry Q(i) is negligible, using Givens rotations.\n" \
    "\n" \
    "Parameters:\n" \
    "  updatu (bool): True to update U with left Givens rotations\n" \
    "  updatv (bool): True to update V with right Givens rotations\n" \
    "  m (int): Number of rows of matrix U, m >= 0\n" \
    "  n (int): Number of rows of matrix V, n >= 0\n" \
    "  i (int): Index of negligible diagonal entry Q(i), 1-based\n" \
    "  k (int): Last index of bidiagonal submatrix, 1-based\n" \
    "  q (ndarray): Diagonal entries of J (p,), p = min(m,n)\n" \
    "  e (ndarray): Superdiagonal entries of J (p-1,)\n" \
    "  u (ndarray): Left transformation matrix (m x p, F-order)\n" \
    "  v (ndarray): Right transformation matrix (n x p, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (q, e, u, v): Transformed diagonal, superdiagonal, and matrices"

#define DOC_MB02OD "Solve triangular matrix equation with condition estimation.\n" \
    "\n" \
    "Solves op(A)*X = alpha*B or X*op(A) = alpha*B where A is triangular.\n" \
    "Only solves if reciprocal condition number RCOND > TOL.\n" \
    "\n" \
    "Parameters:\n" \
    "  side (str): 'L' for op(A)*X = alpha*B, 'R' for X*op(A) = alpha*B\n" \
    "  uplo (str): 'U' for upper triangular, 'L' for lower triangular\n" \
    "  trans (str): 'N' for op(A)=A, 'T'/'C' for op(A)=A'\n" \
    "  diag (str): 'U' for unit triangular, 'N' for non-unit\n" \
    "  norm (str): '1' or 'O' for 1-norm, 'I' for infinity-norm\n" \
    "  alpha (float): Scalar multiplier\n" \
    "  a (ndarray): Triangular matrix (k x k, F-order) where k=m if side='L'\n" \
    "  b (ndarray): RHS matrix B (m x n, F-order), overwritten with solution X\n" \
    "  tol (float, optional): Tolerance. If <= 0, uses k*k*eps.\n" \
    "\n" \
    "Returns:\n" \
    "  (x, rcond, info): Solution X, reciprocal condition number, exit code\n" \
    "  info=1 means matrix is singular (rcond <= tol)"

#define DOC_MB02PD "Solve linear system op(A)*X = B with LU factorization.\n" \
    "\n" \
    "Uses LU factorization with optional equilibration and\n" \
    "iterative refinement for improved accuracy.\n" \
    "\n" \
    "Parameters:\n" \
    "  fact (str): 'N' factor A, 'E' equilibrate+factor, 'F' factored\n" \
    "  trans (str): 'N' solve A*X=B, 'T'/'C' solve A'*X=B\n" \
    "  a (ndarray): N-by-N coefficient matrix A (F-order)\n" \
    "  b (ndarray): N-by-NRHS right-hand side B (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (x, ferr, berr, rcond, info): Solution X, forward/backward error,\n" \
    "                                condition number, exit code"

#define DOC_MB02QD "Linear least squares solution using complete orthogonal factorization.\n" \
    "\n" \
    "Computes a solution to minimize ||A*X - B|| for rank-deficient matrices.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'L' standard least squares, 'F' with free elements in Y\n" \
    "  iniper (str): 'P' use initial permutation in jpvt, 'N' no initial perm\n" \
    "  m (int): Number of rows of A\n" \
    "  n (int): Number of columns of A\n" \
    "  nrhs (int): Number of right-hand sides\n" \
    "  rcond (float): Tolerance for rank determination, 0 <= rcond <= 1\n" \
    "  svlmax (float): Estimate of largest singular value, use 0.0 if unknown\n" \
    "  a (ndarray): M-by-N matrix A (F-order)\n" \
    "  b (ndarray): M-by-NRHS right-hand side B (F-order)\n" \
    "  y (ndarray, optional): Free elements for job='F' ((n-rank)*nrhs,)\n" \
    "  jpvt (ndarray, optional): Initial permutation for iniper='P' (n,)\n" \
    "\n" \
    "Returns:\n" \
    "  (x, rank, sval, jpvt, info): Solution X (N-by-NRHS), effective rank,\n" \
    "                               singular value estimates [3], permutation, exit code"

#define DOC_MB02RD "Solve linear system with Hessenberg LU factorization.\n" \
    "\n" \
    "Solves H*X=B or H'*X=B using LU from mb02sd.\n" \
    "\n" \
    "Parameters:\n" \
    "  trans (str): 'N' for H*X=B, 'T'/'C' for H'*X=B\n" \
    "  n (int): Order of H\n" \
    "  nrhs (int): Number of RHS columns\n" \
    "  h (ndarray): LU factors from mb02sd (n x n, F-order)\n" \
    "  ipiv (ndarray): Pivots from mb02sd (n)\n" \
    "  b (ndarray): RHS matrix (n x nrhs, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (x, info): Solution X, exit code"

#define DOC_MB02RZ "Solve complex Hessenberg system using LU factorization.\n" \
    "\n" \
    "Solves H*X=B, H'*X=B, or H^H*X=B using factorization from mb02sz.\n" \
    "\n" \
    "Parameters:\n" \
    "  trans (str): 'N' (no transpose), 'T' (transpose), 'C' (conjugate transpose)\n" \
    "  h (ndarray): LU factors from mb02sz (n x n, F-order)\n" \
    "  ipiv (ndarray): Pivot indices from mb02sz (n elements)\n" \
    "  b (ndarray): Right-hand side matrix (n x nrhs, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (x, info): Solution matrix and exit code (0=success)"

#define DOC_MB02SD "LU factorization of upper Hessenberg matrix.\n" \
    "\n" \
    "Computes H = P*L*U for upper Hessenberg H.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Order of H\n" \
    "  h (ndarray): Upper Hessenberg matrix (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (h, ipiv, info): LU factors, pivots (1-based), exit code"

#define DOC_MB02SZ "LU factorization of complex upper Hessenberg matrix.\n" \
    "\n" \
    "Computes LU factorization H = P*L*U where:\n" \
    "  - P is permutation matrix\n" \
    "  - L is unit lower bidiagonal\n" \
    "  - U is upper triangular\n" \
    "\n" \
    "Parameters:\n" \
    "  h (ndarray): Complex upper Hessenberg matrix (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (h_lu, ipiv, info):\n" \
    "    - h_lu: LU factors (L stored below diagonal, U on/above diagonal)\n" \
    "    - ipiv: Pivot indices (1-based, n elements)\n" \
    "    - info: Exit code (0=success, >0=singular U)"

#define DOC_MB02TD "Estimate reciprocal condition number of upper Hessenberg matrix.\n" \
    "\n" \
    "Uses LU factorization from mb02sd to estimate 1/cond(H).\n" \
    "\n" \
    "Parameters:\n" \
    "  norm (str): '1' or 'O' for 1-norm, 'I' for infinity-norm\n" \
    "  n (int): Order of matrix H\n" \
    "  hnorm (float): Norm of original matrix H (before factorization)\n" \
    "  h (ndarray): LU factors from mb02sd (n x n, F-order)\n" \
    "  ipiv (ndarray): Pivot indices from mb02sd (n elements)\n" \
    "\n" \
    "Returns:\n" \
    "  (rcond, info): Reciprocal condition estimate and exit code (0=success)"

#define DOC_MB02TZ "Estimate reciprocal condition number of complex Hessenberg matrix.\n" \
    "\n" \
    "Uses LU factorization from mb02sz to estimate 1/cond(H).\n" \
    "\n" \
    "Parameters:\n" \
    "  norm (str): '1' or 'O' for 1-norm, 'I' for infinity-norm\n" \
    "  hnorm (float): Norm of original matrix H (before factorization)\n" \
    "  h (ndarray): LU factors from mb02sz (n x n, F-order)\n" \
    "  ipiv (ndarray): Pivot indices from mb02sz (n elements)\n" \
    "\n" \
    "Returns:\n" \
    "  (rcond, info): Reciprocal condition estimate and exit code (0=success)"

#define DOC_MB02UD "Minimum norm least squares solution using SVD.\n" \
    "\n" \
    "Solves op(R)*X = alpha*B (side='L') or X*op(R) = alpha*B (side='R')\n" \
    "where R is upper triangular, using singular value decomposition.\n" \
    "\n" \
    "Parameters:\n" \
    "  fact (str): 'N' to compute SVD, 'F' if SVD already available\n" \
    "  side (str): 'L' for left, 'R' for right\n" \
    "  trans (str): 'N' for op(R)=R, 'T'/'C' for op(R)=R'\n" \
    "  jobp (str): 'P' to compute/use pseudoinverse, 'N' otherwise\n" \
    "  m (int): Number of rows of B\n" \
    "  n (int): Number of columns of B\n" \
    "  alpha (float): Scalar multiplier\n" \
    "  rcond (float): Rank threshold (not used if fact='F')\n" \
    "  r (ndarray): L-by-L upper triangular matrix R (F-order)\n" \
    "  b (ndarray): M-by-N matrix B (F-order)\n" \
    "  q (ndarray, optional): Q matrix (required if fact='F')\n" \
    "  sv (ndarray, optional): Singular values (required if fact='F')\n" \
    "  rank (int, optional): Rank (required if fact='F')\n" \
    "  rp (ndarray, optional): Pseudoinverse (if fact='F' and jobp='P')\n" \
    "  ldwork (int, optional): Workspace size\n" \
    "\n" \
    "Returns:\n" \
    "  (x, q, sv, rank, rp, info): Solution, Q matrix, singular values, rank, pseudoinverse, exit code"

#define DOC_MB02UU "Solve linear system using LU factorization with complete pivoting.\n" \
    "\n" \
    "Solves A*x = scale*rhs using the LU factorization from mb02uv.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Order of matrix A\n" \
    "  a (ndarray): LU factors from mb02uv (n x n, F-order)\n" \
    "  rhs (ndarray): Right-hand side vector (n,)\n" \
    "  ipiv (ndarray): Row pivot indices from mb02uv (n,)\n" \
    "  jpiv (ndarray): Column pivot indices from mb02uv (n,)\n" \
    "\n" \
    "Returns:\n" \
    "  (x, scale): Solution vector and scale factor (0 < scale <= 1)"

#define DOC_MB02UV "LU factorization with complete pivoting.\n" \
    "\n" \
    "Computes A = P*L*U*Q where P, Q are permutation matrices.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Order of matrix A\n" \
    "  a (ndarray): n-by-n matrix A (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a_lu, ipiv, jpiv, info): LU factors, row pivots, column pivots, exit code"

#define DOC_MB02UW "Solve 1x1 or 2x2 linear system with scaling and perturbation.\n" \
    "\n" \
    "Solves A*X = s*B or A'*X = s*B where A is 1x1 or 2x2. The scale factor\n" \
    "s is computed to prevent overflow. Near-singular A is perturbed.\n" \
    "\n" \
    "Parameters:\n" \
    "  ltrans (bool): True for A', False for A\n" \
    "  par (ndarray): Machine parameters [eps, sfmin, smin]\n" \
    "  a (ndarray): Matrix A (n x n, F-order), n=1 or 2\n" \
    "  b (ndarray): RHS matrix B (n x m, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (x, scale, iwarn): Solution, scale factor, warning (1=A perturbed)"

#define DOC_MB02VD "Solve X * op(A) = B using LU factorization.\n" \
    "\n" \
    "Parameters:\n" \
    "  trans (str): 'N' for X*A=B, 'T'/'C' for X*A'=B\n" \
    "  a (ndarray): Matrix A (n x n, F-order), overwritten with LU factors\n" \
    "  b (ndarray): RHS matrix B (m x n, F-order), overwritten with solution X\n" \
    "\n" \
    "Returns:\n" \
    "  (x, ipiv, info): Solution X, pivots, exit code"

#define DOC_MB02WD "Solve symmetric linear system using conjugate gradient.\n" \
    "\n" \
    "Solves A*x = b for symmetric positive definite A using the\n" \
    "preconditioned conjugate gradient method.\n" \
    "\n" \
    "Parameters:\n" \
    "  form (str): 'U' (upper triangular), 'L' (lower triangular)\n" \
    "  itmax (int): Maximum iterations\n" \
    "  a (ndarray): Symmetric matrix A (n x n, F-order)\n" \
    "  b (ndarray): Right-hand side vector (n,)\n" \
    "  x (ndarray): Initial guess (n,), modified in place\n" \
    "  tol (float): Convergence tolerance\n" \
    "\n" \
    "Returns:\n" \
    "  (x, iterations, residual, iwarn, info): Solution, iteration count,\n" \
    "    residual norm, warning code, exit code"

#define DOC_MB02XD "Solve A'*A*X = B using Cholesky factorization.\n" \
    "\n" \
    "Solves a set of systems A'*A*X = B with A'*A positive definite\n" \
    "using Cholesky factorization. The routine computes A'*A and\n" \
    "solves via symmetric Gaussian elimination.\n" \
    "\n" \
    "Parameters:\n" \
    "  stor (str): 'F' full storage, 'P' packed storage for A'*A\n" \
    "  uplo (str): 'U' upper triangle, 'L' lower triangle\n" \
    "  m (int): Rows of A (>= 0)\n" \
    "  n (int): Columns of A, rows of X and B (>= 0)\n" \
    "  nrhs (int): Number of right-hand sides (>= 0)\n" \
    "  a (ndarray): M-by-N matrix A (F-order)\n" \
    "  b (ndarray): N-by-NRHS right-hand side, overwritten with solution (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (ata, info):\n" \
    "    ata: N-by-N Cholesky factor of A'*A (or packed form)\n" \
    "    info: 0=success, >0=not positive definite (leading minor i)"

#define DOC_MB02YD "Solve augmented system A*x = b, D*x = 0 in least squares sense.\n" \
    "\n" \
    "Parameters:\n" \
    "  cond (str): 'E' = estimate condition, 'N' = check zeros, 'U' = use rank\n" \
    "  n (int): Order of matrix R\n" \
    "  r (ndarray): Upper triangular matrix R (n x n, F-order)\n" \
    "  ipvt (ndarray): Permutation vector (1-based indices)\n" \
    "  diag (ndarray): Diagonal elements of D\n" \
    "  qtb (ndarray): First n elements of Q'*b\n" \
    "  rank (int): Input rank (COND='U') or 0 otherwise\n" \
    "  tol (float): Tolerance for rank determination (COND='E')\n" \
    "\n" \
    "Returns:\n" \
    "  (x, rank, info): Solution vector, estimated rank, exit code"

#define DOC_MB03AB "Compute Givens rotations for Wilkinson shift polynomial.\n" \
    "\n" \
    "Computes two Givens rotations (C1,S1) and (C2,S2) such that the orthogonal\n" \
    "matrix Z makes the first column of the real Wilkinson shift polynomial of\n" \
    "a product of matrices in periodic upper Hessenberg form parallel to the\n" \
    "first unit vector.\n" \
    "\n" \
    "Parameters:\n" \
    "  shft (str): Shift type: 'C'=complex conjugate, 'D'=real identical,\n" \
    "              'R'=real distinct, 'S'=single real shift\n" \
    "  k (int): Number of factors (K >= 1)\n" \
    "  n (int): Order of factors (N >= 2 for single, N >= 3 for double shift)\n" \
    "  amap (ndarray): Factor access map (K,), 1-based indices\n" \
    "  s (ndarray): Signature array (K,), each entry 1 or -1\n" \
    "  sinv (int): Signature multiplier\n" \
    "  a (ndarray): Matrix factors (N x N x K, F-order)\n" \
    "  w1 (float): Real part of first eigenvalue (unused for SHFT='S')\n" \
    "  w2 (float): Second eigenvalue (real) or imaginary part (complex)\n" \
    "\n" \
    "Returns:\n" \
    "  (c1, s1, c2, s2): Givens rotation parameters.\n" \
    "    For SHFT='S': c2=1, s2=0"

#define DOC_MB03AD "Compute Givens rotations for Wilkinson shift polynomial (implicit shifts).\n" \
    "\n" \
    "Computes two Givens rotations (C1,S1) and (C2,S2) such that the orthogonal\n" \
    "matrix Z makes the first column of the real Wilkinson double/single shift\n" \
    "polynomial parallel to the first unit vector. Unlike MB03AB, this routine\n" \
    "computes the shifts implicitly from the trailing submatrices.\n" \
    "\n" \
    "Parameters:\n" \
    "  shft (str): Shift type: 'D'=double shift (N>2), 'S'=single real shift\n" \
    "  k (int): Number of factors (K >= 1)\n" \
    "  n (int): Order of factors (N >= 2 for single, N >= 3 for double shift)\n" \
    "  amap (ndarray): Factor access map (K,), 1-based indices\n" \
    "  s (ndarray): Signature array (K,), each entry 1 or -1\n" \
    "  sinv (int): Signature multiplier\n" \
    "  a (ndarray): Matrix factors (N x N x K, F-order) in periodic Hessenberg form\n" \
    "\n" \
    "Returns:\n" \
    "  (c1, s1, c2, s2): Givens rotation parameters.\n" \
    "    For SHFT='S': c2=1, s2=0"

#define DOC_MB03AE "Compute Givens rotations for Wilkinson shift polynomial (variant).\n" \
    "\n" \
    "Computes two Givens rotations (C1,S1) and (C2,S2) such that the orthogonal\n" \
    "matrix Z makes the first column of the real Wilkinson double/single shift\n" \
    "polynomial parallel to the first unit vector. Unlike MB03AB, this routine\n" \
    "computes shifts from the trailing 2x2 submatrix rather than explicit values.\n" \
    "\n" \
    "Parameters:\n" \
    "  shft (str): 'D'=double shift (assumes N>2), 'S'=single real shift\n" \
    "  k (int): Number of factors (K >= 1)\n" \
    "  n (int): Order of factors (N >= 2)\n" \
    "  amap (ndarray): Factor access map (K,), 1-based indices\n" \
    "  s (ndarray): Signature array (K,), each entry 1 or -1\n" \
    "  sinv (int): Signature multiplier\n" \
    "  a (ndarray): Matrix factors (N x N x K, F-order) in periodic Hessenberg form\n" \
    "\n" \
    "Returns:\n" \
    "  (c1, s1, c2, s2): Givens rotation parameters.\n" \
    "    For SHFT='S' or N==2: c2=1, s2=0"

#define DOC_MB03AG "Compute Givens rotations for Wilkinson shift polynomial (full evaluation).\n" \
    "\n" \
    "Computes two Givens rotations (C1,S1) and (C2,S2) using full evaluation of\n" \
    "the matrix product and DLAHQR for eigenvalue computation. Makes the first\n" \
    "column of the Wilkinson shift polynomial parallel to the first unit vector.\n" \
    "The Hessenberg matrix is the first factor (AMAP[0]).\n" \
    "\n" \
    "More robust but slower. Used when convergence difficulties are encountered\n" \
    "for small order matrices (N, K <= 6).\n" \
    "\n" \
    "Parameters:\n" \
    "  shft (str): 'D'=double shift (N>2), 'S'=single real shift\n" \
    "  k (int): Number of factors (K >= 1)\n" \
    "  n (int): Order of factors (N >= 2)\n" \
    "  amap (ndarray): Factor access map (K,), 1-based indices. AMAP[0] is Hessenberg.\n" \
    "  s (ndarray): Signature array (K,), each entry 1 or -1\n" \
    "  sinv (int): Signature multiplier\n" \
    "  a (ndarray): Matrix factors (N x N x K, F-order) in periodic Hessenberg form\n" \
    "\n" \
    "Returns:\n" \
    "  (c1, s1, c2, s2): Givens rotation parameters.\n" \
    "    For SHFT='S': c2=1, s2=0"

#define DOC_MB03AH "Compute Givens rotations for real Wilkinson shift polynomial (partial evaluation).\n" \
    "\n" \
    "Computes two Givens rotations (C1,S1) and (C2,S2) such that the orthogonal\n" \
    "matrix Q makes the first column of the real Wilkinson double/single shift\n" \
    "polynomial parallel to the first unit vector. The Hessenberg matrix is the\n" \
    "last factor (AMAP[K-1]).\n" \
    "\n" \
    "Uses partial evaluation of the matrix product (trailing 2x2 and first two\n" \
    "columns). Called when convergence difficulties are encountered. For double\n" \
    "shift with two real eigenvalues, both shifts equal the eigenvalue with\n" \
    "minimum modulus.\n" \
    "\n" \
    "Parameters:\n" \
    "  shft (str): 'D'=double shift (assumes N>2), 'S'=single real shift\n" \
    "  k (int): Number of factors (K >= 1)\n" \
    "  n (int): Order of factors (N >= 2)\n" \
    "  amap (ndarray): Factor access map (K,), 1-based indices. AMAP[K-1] is Hessenberg.\n" \
    "  s (ndarray): Signature array (K,), each entry 1 or -1\n" \
    "  sinv (int): Signature multiplier\n" \
    "  a (ndarray): Matrix factors (N x N x K, F-order) in periodic Hessenberg form\n" \
    "\n" \
    "Returns:\n" \
    "  (c1, s1, c2, s2, info): Givens rotation parameters and exit code.\n" \
    "    For SHFT='S' or N==2: c2=1, s2=0. info=0 always (no error checking)."

#define DOC_MB03AI "Compute Givens rotations for Wilkinson shift polynomial (full evaluation).\n" \
    "\n" \
    "Computes two Givens rotations (C1,S1) and (C2,S2) using full evaluation of\n" \
    "the trailing submatrix elements. Makes the first column of the Wilkinson\n" \
    "shift polynomial parallel to the first unit vector.\n" \
    "\n" \
    "Parameters:\n" \
    "  shft (str): 'D'=double shift, 'S'=single real shift\n" \
    "  k (int): Number of factors (K >= 1)\n" \
    "  n (int): Order of factors (N >= 2)\n" \
    "  amap (ndarray): Factor access map (K,), 1-based indices\n" \
    "  s (ndarray): Signature array (K,), each entry 1 or -1\n" \
    "  sinv (int): Signature multiplier\n" \
    "  a (ndarray): Matrix factors (N x N x K, F-order) in periodic Hessenberg form\n" \
    "\n" \
    "Returns:\n" \
    "  (c1, s1, c2, s2): Givens rotation parameters.\n" \
    "    For SHFT='S' or N==2: c2=1, s2=0"

#define DOC_MB03BA "Compute maps for Hessenberg index and signature array.\n" \
    "\n" \
    "Auxiliary routine for periodic QZ algorithms. Computes suitable maps\n" \
    "(AMAP, QMAP) based on Hessenberg index H and signature array S.\n" \
    "\n" \
    "Parameters:\n" \
    "  k (int): Number of factors (K >= 1)\n" \
    "  h (int): Index corresponding to A_1 (1-based)\n" \
    "  s (ndarray): Signature array, dimension (K). Each entry must be 1 or -1.\n" \
    "\n" \
    "Returns:\n" \
    "  (smult, amap, qmap): Signature multiplier, factor access map, Q access map"

#define DOC_MB03BC "Product SVD of 2x2 triangular factors.\n" \
    "\n" \
    "Computes Givens rotations for the product singular value decomposition of K-1\n" \
    "triangular factors corresponding to a 2-by-2 product of K factors in upper\n" \
    "Hessenberg-triangular form.\n" \
    "\n" \
    "For a product of 2-by-2 triangular matrices\n" \
    "  A = A(:,:,2)^S(2) * A(:,:,3)^S(3) * ... * A(:,:,K)^S(K)\n" \
    "\n" \
    "Givens rotations are computed so that the product is diagonalized.\n" \
    "\n" \
    "Parameters:\n" \
    "  k (int): Number of factors (K >= 1)\n" \
    "  amap (ndarray): Factor access map (K,), 1-based indices\n" \
    "  s (ndarray): Signature array (K,), each entry 1 or -1\n" \
    "  sinv (int): Signature multiplier\n" \
    "  a (ndarray): Matrix factors (2 x 2 x K, F-order), upper triangular\n" \
    "  macpar (ndarray): Machine parameters (5,):\n" \
    "    - MACPAR(0): overflow threshold (DLAMCH('O'))\n" \
    "    - MACPAR(1): underflow threshold (DLAMCH('U'))\n" \
    "    - MACPAR(2): safe minimum (DLAMCH('S'))\n" \
    "    - MACPAR(3): relative precision (DLAMCH('E'))\n" \
    "    - MACPAR(4): base (DLAMCH('B'))\n" \
    "\n" \
    "Returns:\n" \
    "  (a, cv, sv):\n" \
    "    - a: Modified triangular factors (product is diagonal)\n" \
    "    - cv: Cosines of Givens rotations (K,)\n" \
    "    - sv: Sines of Givens rotations (K,)"

#define DOC_MB03BD "Compute eigenvalues of periodic Hessenberg matrix product.\n" \
    "\n" \
    "Finds eigenvalues of the generalized matrix product:\n" \
    "  A(:,:,1)^S(1) * A(:,:,2)^S(2) * ... * A(:,:,K)^S(K)\n" \
    "where A(:,:,H) is upper Hessenberg and other factors are upper triangular.\n" \
    "\n" \
    "Uses double-shift periodic QZ algorithm. Can reduce to periodic Schur form.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'E' eigenvalues only, 'S' periodic Schur, 'T' standardized Schur\n" \
    "  defl (str): 'C' careful deflation, 'A' aggressive deflation\n" \
    "  compq (str): 'N' no Q, 'U' update Q, 'I' initialize Q, 'P' partial\n" \
    "  k (int): Number of factors (K >= 1)\n" \
    "  n (int): Order of each factor (N >= 0)\n" \
    "  h (int): Hessenberg index (1 <= H <= K)\n" \
    "  ilo (int): Lower index for active block (1-based)\n" \
    "  ihi (int): Upper index for active block (1-based)\n" \
    "  s (ndarray): Signature array (K,), each entry 1 or -1\n" \
    "  a (ndarray): Matrix factors (N x N x K, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, q, alphar, alphai, beta, scal, iwarn, info):\n" \
    "    - a: Transformed matrices\n" \
    "    - q: Orthogonal factors (if COMPQ != 'N')\n" \
    "    - alphar, alphai: Scaled eigenvalue real/imaginary parts\n" \
    "    - beta: Eigenvalue indicator (0 = infinite)\n" \
    "    - scal: Eigenvalue scaling factors\n" \
    "    - iwarn: Warning code\n" \
    "    - info: Exit code (0 = success)\n" \
    "\n" \
    "Eigenvalue formula: (alphar + alphai*j) / beta * 2^scal"

#define DOC_MB03BE "Apply periodic QZ iterations to 2x2 matrix product.\n" \
    "\n" \
    "Applies at most 20 iterations of a real single shifted periodic QZ algorithm\n" \
    "to the 2-by-2 product of matrices stored in array A. Drives the subdiagonal\n" \
    "element A(2,1) of the first factor toward zero.\n" \
    "\n" \
    "Parameters:\n" \
    "  k (int): Number of factors (K >= 1)\n" \
    "  amap (ndarray): Factor access map (K,), 1-based indices\n" \
    "  s (ndarray): Signature array (K,), each entry 1 or -1\n" \
    "  sinv (int): Signature multiplier\n" \
    "  a (ndarray): Matrix factors (2 x 2 x K, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  a: Modified matrix factors with quasi-triangular first factor"

#define DOC_MB03BF "Apply periodic QZ iterations to 2x2 matrix product with ULP tolerance.\n" \
    "\n" \
    "Applies at most 20 iterations of a real single shifted periodic QZ algorithm\n" \
    "to the 2-by-2 product of matrices stored in array A. Drives the subdiagonal\n" \
    "element A(2,1) of the Hessenberg factor (last one) toward zero.\n" \
    "\n" \
    "Parameters:\n" \
    "  k (int): Number of factors (K >= 1)\n" \
    "  amap (ndarray): Factor access map (K,), 1-based indices\n" \
    "  s (ndarray): Signature array (K,), each entry 1 or -1\n" \
    "  sinv (int): Signature multiplier\n" \
    "  a (ndarray): Matrix factors (2 x 2 x K, F-order)\n" \
    "  ulp (float): Machine precision for convergence test\n" \
    "\n" \
    "Returns:\n" \
    "  a: Modified matrix factors with quasi-triangular Hessenberg factor"

#define DOC_MB03BG "Compute eigenvalues of 2x2 trailing submatrix of matrix product.\n" \
    "\n" \
    "Computes the eigenvalues of the 2-by-2 trailing submatrix of the product\n" \
    "A(:,:,1)^S(1) * A(:,:,2)^S(2) * ... * A(:,:,K)^S(K), where A(:,:,AMAP(K))\n" \
    "is upper Hessenberg and A(:,:,AMAP(i)) for i < K are upper triangular.\n" \
    "\n" \
    "Parameters:\n" \
    "  k (int): Number of factors (K >= 1)\n" \
    "  n (int): Order of factors (N >= 2)\n" \
    "  amap (ndarray): Factor access map (K,), 1-based indices\n" \
    "  s (ndarray): Signature array (K,), each entry 1 or -1\n" \
    "  sinv (int): Signature multiplier\n" \
    "  a (ndarray): Matrix factors (N x N x K, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (wr, wi): Real and imaginary parts of the two eigenvalues"

#define DOC_MB03BZ "Complex periodic QZ algorithm for eigenvalues of generalized matrix products.\n" \
    "\n" \
    "Computes eigenvalues of the complex generalized matrix product:\n" \
    "  A(:,:,1)^S(1) * A(:,:,2)^S(2) * ... * A(:,:,K)^S(K), S(1) = 1\n" \
    "\n" \
    "where A(:,:,1) is upper Hessenberg and A(:,:,i), i=2,...,K are upper triangular.\n" \
    "Can optionally reduce A to periodic Schur form.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'E' eigenvalues only, 'S' Schur form + eigenvalues\n" \
    "  compq (str): 'N' no Q, 'V' update Q, 'I' initialize Q to identity\n" \
    "  k (int): Number of factors (K >= 1)\n" \
    "  n (int): Order of factors (N >= 0)\n" \
    "  ilo (int): Lower bound of active block (1-based)\n" \
    "  ihi (int): Upper bound of active block (1-based)\n" \
    "  s (ndarray): Signature array (K,), S(1)=1, others +1 or -1\n" \
    "  a (ndarray): Matrix factors (N x N x K, F-order, complex128)\n" \
    "  q (ndarray, optional): Initial Q matrices (N x N x K) if COMPQ='V'\n" \
    "\n" \
    "Returns:\n" \
    "  (a, q, alpha, beta, scal, info):\n" \
    "    - a: Transformed matrices (Schur form if JOB='S')\n" \
    "    - q: Unitary factors (if COMPQ != 'N')\n" \
    "    - alpha: Scaled eigenvalues (complex, dimension N)\n" \
    "    - beta: Infinite eigenvalue indicator (complex, dimension N)\n" \
    "    - scal: Scaling exponents (int, dimension N)\n" \
    "    - info: Exit code (0=success, <0=param error, >0=convergence failure)\n" \
    "\n" \
    "Eigenvalue formula: alpha(i) / beta(i) * BASE^scal(i)"

#define DOC_MB03CD "Exchange eigenvalues in block triangular pencils.\n" \
    "\n" \
    "Computes orthogonal matrices Q1, Q2, Q3 for a real 2-by-2, 3-by-3, or 4-by-4\n" \
    "regular block upper triangular pencil aAB - bD such that the eigenvalues in\n" \
    "Spec(A11*B11, D11) and Spec(A22*B22, D22) are exchanged.\n" \
    "\n" \
    "For UPLO='U': Upper block triangular, eigenvalues are exchanged.\n" \
    "For UPLO='L': Lower block triangular, eigenvalues are NOT exchanged.\n" \
    "\n" \
    "Parameters:\n" \
    "  uplo (str): 'U' upper block triangular, 'L' lower block triangular\n" \
    "  n1 (int): Size of upper left block (N1 <= 2)\n" \
    "  n2 (int): Size of lower right block (N2 <= 2)\n" \
    "  prec (float): Machine precision (from DLAMCH)\n" \
    "  a (ndarray): Matrix A of pencil (n1+n2 x n1+n2, F-order)\n" \
    "  b (ndarray): Matrix B of pencil (n1+n2 x n1+n2, F-order)\n" \
    "  d (ndarray): Matrix D of pencil (n1+n2 x n1+n2, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, d, q1, q2, q3, n1, n2, info):\n" \
    "    - a, b, d: Transformed matrices\n" \
    "    - q1, q2, q3: Orthogonal transformation matrices\n" \
    "    - n1, n2: Exchanged block sizes (if UPLO='U')\n" \
    "    - info: Exit code (0=success, 1-5=algorithm error)"

#define DOC_MB03CZ "Exchange eigenvalues of complex 2x2 upper triangular pencil.\n" \
    "\n" \
    "Computes unitary matrices Q1, Q2, Q3 for a complex 2-by-2 regular pencil\n" \
    "aAB - bD, with A, B, D upper triangular, such that Q3' A Q2, Q2' B Q1,\n" \
    "Q3' D Q1 are still upper triangular, but the eigenvalues are in reversed order.\n" \
    "\n" \
    "The matrices Q1, Q2, Q3 are represented by:\n" \
    "  Q = [[co, si], [-conj(si), co]]\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): Complex 2x2 upper triangular matrix A (F-order, complex128)\n" \
    "  b (ndarray): Complex 2x2 upper triangular matrix B (F-order, complex128)\n" \
    "  d (ndarray): Complex 2x2 upper triangular matrix D (F-order, complex128)\n" \
    "\n" \
    "Returns:\n" \
    "  (co1, si1, co2, si2, co3, si3): Unitary matrix parameters\n" \
    "    - co1, co2, co3: Cosines (real)\n" \
    "    - si1, si2, si3: Sines (complex)"

#define DOC_MB03DD "Exchange eigenvalues of real block upper triangular pencil.\n" \
    "\n" \
    "Computes orthogonal matrices Q1 and Q2 for a real 2-by-2, 3-by-3, or 4-by-4\n" \
    "regular block upper triangular pencil aA - bB such that the pencil\n" \
    "a(Q2' A Q1) - b(Q2' B Q1) is still in block upper triangular form, but\n" \
    "the eigenvalues in Spec(A11, B11), Spec(A22, B22) are exchanged.\n" \
    "\n" \
    "Optionally, upper triangularizes the real regular pencil in block lower\n" \
    "triangular form while keeping eigenvalues in the same diagonal position.\n" \
    "\n" \
    "Parameters:\n" \
    "  uplo (str): 'U' upper block triangular (eigenvalues exchanged),\n" \
    "              'T' upper block triangular with B triangular (eigenvalues exchanged),\n" \
    "              'L' lower block triangular (eigenvalues NOT exchanged)\n" \
    "  n1 (int): Size of upper left block (N1 <= 2)\n" \
    "  n2 (int): Size of lower right block (N2 <= 2)\n" \
    "  prec (float): Machine precision (from DLAMCH)\n" \
    "  a (ndarray): Matrix A of pencil (n1+n2 x n1+n2, F-order)\n" \
    "  b (ndarray): Matrix B of pencil (n1+n2 x n1+n2, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, q1, q2, n1, n2, info):\n" \
    "    - a, b: Transformed matrices\n" \
    "    - q1, q2: Orthogonal transformation matrices\n" \
    "    - n1, n2: Exchanged block sizes (if UPLO='U'/'T' and INFO=0)\n" \
    "    - info: Exit code (0=success, 3=QZ failed, 4=DHGEQZ error, 5=DTGSEN reorder failed)"

#define DOC_MB03DZ "Exchange eigenvalues of complex 2x2 upper triangular pencil.\n" \
    "\n" \
    "Computes unitary matrices Q1 and Q2 for a complex 2-by-2 regular pencil\n" \
    "aA - bB with A, B upper triangular, such that Q2' (aA - bB) Q1 is still\n" \
    "upper triangular but the eigenvalues are in reversed order.\n" \
    "\n" \
    "The matrices Q1 and Q2 are represented by:\n" \
    "     (  CO1  SI1  )       (  CO2  SI2  )\n" \
    "Q1 = (            ), Q2 = (            ).\n" \
    "     ( -SI1' CO1  )       ( -SI2' CO2  )\n" \
    "\n" \
    "where ' denotes conjugate transpose.\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): Complex 2x2 upper triangular matrix A (F-order)\n" \
    "  b (ndarray): Complex 2x2 upper triangular matrix B (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (co1, si1, co2, si2):\n" \
    "    - co1 (float): Cosine of Q1\n" \
    "    - si1 (complex): Sine of Q1\n" \
    "    - co2 (float): Cosine of Q2\n" \
    "    - si2 (complex): Sine of Q2"

#define DOC_MB03ED "Compute orthogonal matrices for 2x2 or 4x4 skew-Hamiltonian/Hamiltonian pencil.\n" \
    "\n" \
    "Computes orthogonal matrices Q1, Q2, Q3 for a real 2-by-2 or 4-by-4 regular\n" \
    "pencil aAB - bD, such that Q3' A Q2 and Q2' B Q1 are upper triangular,\n" \
    "Q3' D Q1 is upper quasi-triangular, and eigenvalues with negative real parts\n" \
    "(if any) are allocated on top.\n" \
    "\n" \
    "The pencil has the form:\n" \
    "    ( A11  0  ) ( B11  0  )     (  0  D12 )\n" \
    "a * (         ) (         ) - b (         )\n" \
    "    (  0  A22 ) (  0  B22 )     ( D21  0  )\n" \
    "\n" \
    "where A11, A22, B11, B22, D12 are upper triangular.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Order of the input pencil, N = 2 or N = 4\n" \
    "  prec (float): Machine precision (relative machine precision * base)\n" \
    "  a (ndarray): N-by-N upper triangular matrix A of pencil (F-order)\n" \
    "  b (ndarray): N-by-N upper triangular matrix B of pencil (F-order)\n" \
    "  d (ndarray): N-by-N matrix D of pencil (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (d, q1, q2, q3, info):\n" \
    "    - d: Transformed D (Schur form if N=4, unchanged if N=2)\n" \
    "    - q1: First orthogonal transformation matrix (N x N)\n" \
    "    - q2: Second orthogonal transformation matrix (N x N)\n" \
    "    - q3: Third orthogonal transformation matrix (N x N)\n" \
    "    - info: Exit code (0=success, 1=QZ failed, 2=other DGGES error)"

#define DOC_MB03FD "Reduce 2x2 or 4x4 block diagonal skew-Hamiltonian/Hamiltonian pencil to Schur form.\n" \
    "\n" \
    "Computes orthogonal matrices Q1 and Q2 for a real 2-by-2 or 4-by-4 regular\n" \
    "pencil aA - bB where:\n" \
    "  A = [A11 0; 0 A22] (block diagonal, upper triangular blocks)\n" \
    "  B = [0 B12; B21 0] (anti-block-diagonal)\n" \
    "\n" \
    "such that Q2' A Q1 is upper triangular, Q2' B Q1 is upper quasi-triangular,\n" \
    "and eigenvalues with negative real parts (if any) are allocated on top.\n" \
    "\n" \
    "For N=2 with real eigenvalues, computes Givens rotations. For N=4, uses\n" \
    "DGGES (generalized Schur decomposition) with balancing fallback.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Order of the pencil, N = 2 or N = 4\n" \
    "  prec (float): Machine precision (relative machine precision * base)\n" \
    "  a (ndarray): N-by-N block diagonal matrix A (F-order)\n" \
    "  b (ndarray): N-by-N anti-block-diagonal matrix B (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, q1, q2, info):\n" \
    "    - a: Transformed A (upper triangular if N=4, unchanged if N=2)\n" \
    "    - b: Transformed B (quasi-triangular if N=4, unchanged if N=2)\n" \
    "    - q1: First orthogonal transformation matrix (N x N)\n" \
    "    - q2: Second orthogonal transformation matrix (N x N)\n" \
    "    - info: Exit code (0=success, 1=QZ failed in DGGES, 2=other DGGES error)"

#define DOC_MB03FZ "Eigenvalues and deflating subspace of complex skew-Hamiltonian/Hamiltonian pencil.\n" \
    "\n" \
    "Computes eigenvalues of a complex N-by-N skew-Hamiltonian/Hamiltonian pencil\n" \
    "aS - bH with S = J*Z'*J'*Z and H = [[B,F],[G,-B']] where J = [[0,I],[-I,0]].\n" \
    "\n" \
    "Optionally computes orthonormal basis of right deflating subspace (Q) and\n" \
    "companion subspace (U) corresponding to eigenvalues with strictly negative\n" \
    "real part.\n" \
    "\n" \
    "Parameters:\n" \
    "  compq (str): 'N'=no Q, 'C'=compute deflating subspace Q\n" \
    "  compu (str): 'N'=no U, 'C'=compute companion subspace U\n" \
    "  orth (str): Orthogonalization method: 'P'=QR with pivoting, 'S'=SVD, 'Q'=QR\n" \
    "  n (int): Order of pencil (n >= 0, must be even)\n" \
    "  z (ndarray): N-by-N complex matrix Z (F-order)\n" \
    "  b (ndarray): N/2-by-N/2 complex matrix B (F-order)\n" \
    "  fg (ndarray): N/2-by-(N/2+1) packed F and G (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (z, b, fg, neig, d, c, q, u, alphar, alphai, beta, info):\n" \
    "    Transformed matrices, number of negative eigenvalues, output matrices,\n" \
    "    deflating/companion subspaces, eigenvalue components, exit code.\n" \
    "    Eigenvalue = (alphar + i*alphai) / beta"

#define DOC_MB03GD "Exchange eigenvalues of 2x2 or 4x4 skew-Hamiltonian/Hamiltonian pencil (factored).\n" \
    "\n" \
    "Computes orthogonal matrix Q and orthogonal symplectic matrix U for a real\n" \
    "regular 2-by-2 or 4-by-4 skew-Hamiltonian/Hamiltonian pencil a J B' J' B - b D\n" \
    "with B = [[B11,B12],[0,B22]], D = [[D11,D12],[0,-D11']], J = [[0,I],[-I,0]].\n" \
    "\n" \
    "Such that J Q' J' D Q and U' B Q keep block triangular form, but eigenvalues\n" \
    "are reordered.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Order of pencil, N = 2 or N = 4\n" \
    "  b (ndarray): N-by-N upper block triangular matrix B (F-order)\n" \
    "  d (ndarray): (N/2)-by-N array, first block row of Hamiltonian D (F-order)\n" \
    "  macpar (ndarray): Machine parameters [DLAMCH('P'), DLAMCH('S')]. Not used for N=2.\n" \
    "\n" \
    "Returns:\n" \
    "  (q, u, info):\n" \
    "    - q (ndarray): N-by-N orthogonal transformation matrix Q\n" \
    "    - u (ndarray): N-by-N orthogonal symplectic transformation matrix U\n" \
    "    - info (int): 0=success, 1=B11 or B22 numerically singular"

#define DOC_MB03GZ "Compute unitary matrices for 2x2 skew-Hamiltonian/Hamiltonian pencil.\n" \
    "\n" \
    "Computes a unitary matrix Q and a unitary symplectic matrix U for a complex\n" \
    "regular 2-by-2 skew-Hamiltonian/Hamiltonian pencil aS - bH with S = J Z' J' Z,\n" \
    "where Z and H are upper triangular.\n" \
    "\n" \
    "The matrices Q and U are represented by:\n" \
    "  Q = [[CO1, SI1], [-conj(SI1), CO1]]\n" \
    "  U = [[CO2, SI2], [-conj(SI2), CO2]]\n" \
    "\n" \
    "Such that U' Z Q and (J Q J')' H Q are both upper triangular, but the\n" \
    "eigenvalues are in reversed order.\n" \
    "\n" \
    "Parameters:\n" \
    "  z11 (complex): Upper left element of Z\n" \
    "  z12 (complex): Upper right element of Z\n" \
    "  z22 (complex): Lower right element of Z\n" \
    "  h11 (complex): Upper left element of H\n" \
    "  h12 (complex): Upper right element of H\n" \
    "\n" \
    "Returns:\n" \
    "  (co1, si1, co2, si2):\n" \
    "    - co1 (float): Cosine element of Q\n" \
    "    - si1 (complex): Sine element of Q\n" \
    "    - co2 (float): Cosine element of U\n" \
    "    - si2 (complex): Sine element of U"

#define DOC_MB03HD "Exchange eigenvalues of 2x2 or 4x4 skew-Hamiltonian/Hamiltonian pencil.\n" \
    "\n" \
    "Computes an orthogonal matrix Q for a real regular 2-by-2 or 4-by-4\n" \
    "skew-Hamiltonian/Hamiltonian pencil in structured Schur form:\n" \
    "\n" \
    "              ( A11 A12  )     ( B11  B12  )\n" \
    "  aA - bB = a (          ) - b (           )\n" \
    "              (  0  A11' )     (  0  -B11' )\n" \
    "\n" \
    "such that J Q' J' (aA - bB) Q is still in structured Schur form but the\n" \
    "eigenvalues are exchanged.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Order of the pencil, N = 2 or N = 4\n" \
    "  a (ndarray): N/2-by-N array. First block row of skew-Hamiltonian A.\n" \
    "               Only (1,1), (1,2), (1,4), (2,2) referenced for N=4. Not used for N=2.\n" \
    "  b (ndarray): N/2-by-N array. First block row of Hamiltonian B.\n" \
    "  macpar (ndarray): Machine parameters [DLAMCH('P'), DLAMCH('S')]. Not used for N=2.\n" \
    "\n" \
    "Returns:\n" \
    "  (q, info):\n" \
    "    - q (ndarray): N-by-N orthogonal transformation matrix Q\n" \
    "    - info (int): 0=success, 1=B11 nearly singular (perturbed values used)"

#define DOC_MB03HZ "Exchange eigenvalues of complex 2x2 skew-Hamiltonian/Hamiltonian pencil.\n" \
    "\n" \
    "Computes a unitary matrix Q for a complex regular 2-by-2\n" \
    "skew-Hamiltonian/Hamiltonian pencil aS - bH with\n" \
    "\n" \
    "    (  S11  S12  )        (  H11  H12  )\n" \
    "S = (            ),   H = (            ),\n" \
    "    (   0   S11' )        (   0  -H11' )\n" \
    "\n" \
    "such that J Q' J' (aS - bH) Q is upper triangular but the eigenvalues\n" \
    "are in reversed order. The matrix Q is represented by\n" \
    "\n" \
    "    (  CO  SI  )\n" \
    "Q = (          ).\n" \
    "    ( -SI' CO  )\n" \
    "\n" \
    "The notation M' denotes the conjugate transpose of the matrix M.\n" \
    "\n" \
    "Parameters:\n" \
    "  s11 (complex): Upper left element of skew-Hamiltonian matrix S\n" \
    "  s12 (complex): Upper right element of skew-Hamiltonian matrix S\n" \
    "  h11 (complex): Upper left element of Hamiltonian matrix H\n" \
    "  h12 (complex): Upper right element of Hamiltonian matrix H\n" \
    "\n" \
    "Returns:\n" \
    "  (co, si):\n" \
    "    - co (float): Upper left element of Q (real)\n" \
    "    - si (complex): Upper right element of Q"

#define DOC_MB03ID "Reorder eigenvalues of real skew-Hamiltonian/Hamiltonian pencil (factored version).\n" \
    "\n" \
    "Moves eigenvalues with strictly negative real parts of an N-by-N real\n" \
    "skew-Hamiltonian/Hamiltonian pencil aS - bH in structured Schur form to the\n" \
    "leading principal subpencil, while keeping the triangular form.\n" \
    "\n" \
    "On entry: S = J Z' J' Z, J = [[0, I], [-I, 0]], Z = [[A, D], [0, C]], H = [[B, F], [0, -B']]\n" \
    "where A is upper triangular, B is upper quasi-triangular, C is lower triangular.\n" \
    "\n" \
    "Parameters:\n" \
    "  compq (str): 'N'=no Q, 'I'=init Q to identity, 'U'=update Q\n" \
    "  compu (str): 'N'=no U, 'I'=init U to identity, 'U'=update U\n" \
    "  n (int): Order of the pencil (n >= 0, even)\n" \
    "  a (ndarray): Upper triangular matrix A (n/2 x n/2, F-order)\n" \
    "  c (ndarray): Lower triangular matrix C (n/2 x n/2, F-order)\n" \
    "  d (ndarray): Matrix D (n/2 x n/2, F-order)\n" \
    "  b (ndarray): Upper quasi-triangular matrix B (n/2 x n/2, F-order)\n" \
    "  f (ndarray): Symmetric matrix F upper part (n/2 x n/2, F-order)\n" \
    "  q (ndarray, optional): Orthogonal matrix Q (n x n, F-order)\n" \
    "  u1 (ndarray, optional): Upper left block of orthogonal symplectic U (n/2 x n/2, F-order)\n" \
    "  u2 (ndarray, optional): Upper right block of orthogonal symplectic U (n/2 x n/2, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, c, d, b, f, q, u1, u2, neig, info):\n" \
    "    a, c, d, b, f: Transformed matrices\n" \
    "    q: Orthogonal matrix Q\n" \
    "    u1, u2: Blocks of orthogonal symplectic U\n" \
    "    neig (int): Number of eigenvalues with negative real part\n" \
    "    info (int): 0 = success, <0 = -i means i-th argument invalid, 1=QZ failed, 2=MB03CD error, 3=MB03GD error"

#define DOC_MB03IZ "Reorder eigenvalues of complex skew-Hamiltonian/Hamiltonian pencil.\n" \
    "\n" \
    "Moves eigenvalues with strictly negative real parts of an N-by-N complex\n" \
    "skew-Hamiltonian/Hamiltonian pencil aS - bH in structured Schur form to the\n" \
    "leading principal subpencil, while keeping the triangular form.\n" \
    "\n" \
    "Parameters:\n" \
    "  compq (str): 'N'=no Q, 'I'=init Q to identity, 'U'=update Q\n" \
    "  compu (str): 'N'=no U, 'I'=init U to identity, 'U'=update U\n" \
    "  n (int): Order of the pencil (n >= 0, even)\n" \
    "  a (ndarray): Upper triangular matrix A (n/2 x n/2, complex, F-order)\n" \
    "  c (ndarray): Lower triangular matrix C (n/2 x n/2, complex, F-order)\n" \
    "  d (ndarray): Matrix D (n/2 x n/2, complex, F-order)\n" \
    "  b (ndarray): Upper triangular matrix B (n/2 x n/2, complex, F-order)\n" \
    "  f (ndarray): Hermitian matrix F upper part (n/2 x n/2, complex, F-order)\n" \
    "  tol (float): Tolerance for eigenvalue sign (tol <= 0 uses default)\n" \
    "  q (ndarray, optional): Unitary matrix Q (n x n, complex, F-order)\n" \
    "  u1 (ndarray, optional): Upper left block of U (n/2 x n/2, complex, F-order)\n" \
    "  u2 (ndarray, optional): Upper right block of U (n/2 x n/2, complex, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, c, d, b, f, q, u1, u2, neig, info):\n" \
    "    a, c, d, b, f: Transformed matrices\n" \
    "    q: Unitary matrix Q\n" \
    "    u1, u2: Blocks of unitary symplectic U\n" \
    "    neig (int): Number of eigenvalues with negative real part\n" \
    "    info (int): 0 = success, <0 = -i means i-th argument invalid"

#define DOC_MB03JD "Reorder eigenvalues of real skew-Hamiltonian/Hamiltonian pencil.\n" \
    "\n" \
    "Moves eigenvalues with strictly negative real parts of an N-by-N real\n" \
    "skew-Hamiltonian/Hamiltonian pencil aS - bH in structured Schur form to the\n" \
    "leading principal subpencil, while keeping the triangular form.\n" \
    "\n" \
    "Parameters:\n" \
    "  compq (str): 'N'=no Q, 'I'=init Q to identity, 'U'=update Q\n" \
    "  n (int): Order of the pencil (n >= 0, even)\n" \
    "  a (ndarray): Upper triangular matrix A (n/2 x n/2, F-order)\n" \
    "  d (ndarray): Skew-symmetric matrix D upper part (n/2 x n/2, F-order)\n" \
    "  b (ndarray): Upper quasi-triangular matrix B (n/2 x n/2, F-order)\n" \
    "  f (ndarray): Symmetric matrix F upper part (n/2 x n/2, F-order)\n" \
    "  q (ndarray, optional): Orthogonal matrix Q (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, d, b, f, q, neig, info):\n" \
    "    a, d, b, f: Transformed matrices\n" \
    "    q: Orthogonal matrix Q\n" \
    "    neig (int): Number of eigenvalues with negative real part\n" \
    "    info (int): 0 = success, <0 = -i means i-th argument invalid"

#define DOC_MB03JP "Reorder eigenvalues of real skew-Hamiltonian/Hamiltonian pencil (panel variant).\n" \
    "\n" \
    "Panel-based blocked variant of MB03JD for better performance on large matrices.\n" \
    "Moves eigenvalues with strictly negative real parts of an N-by-N real\n" \
    "skew-Hamiltonian/Hamiltonian pencil aS - bH in structured Schur form to the\n" \
    "leading principal subpencil, while keeping the triangular form.\n" \
    "\n" \
    "Parameters:\n" \
    "  compq (str): 'N'=no Q, 'I'=init Q to identity, 'U'=update Q\n" \
    "  n (int): Order of the pencil (n >= 0, even)\n" \
    "  a (ndarray): Upper triangular matrix A (n/2 x n/2, F-order)\n" \
    "  d (ndarray): Skew-symmetric matrix D upper part (n/2 x n/2, F-order)\n" \
    "  b (ndarray): Upper quasi-triangular matrix B (n/2 x n/2, F-order)\n" \
    "  f (ndarray): Symmetric matrix F upper part (n/2 x n/2, F-order)\n" \
    "  q (ndarray, optional): Orthogonal matrix Q (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, d, b, f, q, neig, info):\n" \
    "    a, d, b, f: Transformed matrices\n" \
    "    q: Orthogonal matrix Q\n" \
    "    neig (int): Number of eigenvalues with negative real part\n" \
    "    info (int): 0 = success, <0 = -i means i-th argument invalid"

#define DOC_MB03JZ "Reorder eigenvalues of complex skew-Hamiltonian/Hamiltonian pencil.\n" \
    "\n" \
    "Moves eigenvalues with strictly negative real parts of an N-by-N complex\n" \
    "skew-Hamiltonian/Hamiltonian pencil aS - bH in structured Schur form to the\n" \
    "leading principal subpencil, while keeping the triangular form.\n" \
    "\n" \
    "On entry: S = [[A, D], [0, A']], H = [[B, F], [0, -B']]\n" \
    "where A, B are upper triangular, D is skew-Hermitian, F is Hermitian.\n" \
    "\n" \
    "Parameters:\n" \
    "  compq (str): 'N'=no Q, 'I'=init Q to identity, 'U'=update Q\n" \
    "  n (int): Order of the pencil (n >= 0, even)\n" \
    "  a (ndarray): Upper triangular matrix A (n/2 x n/2, F-order, complex)\n" \
    "  d (ndarray): Skew-Hermitian matrix D upper part (n/2 x n/2, F-order, complex)\n" \
    "  b (ndarray): Upper triangular matrix B (n/2 x n/2, F-order, complex)\n" \
    "  f (ndarray): Hermitian matrix F upper part (n/2 x n/2, F-order, complex)\n" \
    "  q (ndarray, optional): Unitary matrix Q (n x n, F-order, complex)\n" \
    "  tol (float, optional): Tolerance for eigenvalue sign (default: MIN(N,10)*EPS)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, d, b, f, q, neig, info):\n" \
    "    a, d, b, f: Transformed matrices (complex)\n" \
    "    q: Unitary matrix Q (complex)\n" \
    "    neig (int): Number of eigenvalues with negative real part\n" \
    "    info (int): 0 = success, <0 = -i means i-th argument invalid"

#define DOC_MB03KA "Move diagonal blocks in generalized periodic Schur form.\n" \
    "\n" \
    "Reorders diagonal blocks of formal matrix product T22_K^S(K) * ... * T22_1^S(1)\n" \
    "by moving the block at row index IFST to row index ILST.\n" \
    "\n" \
    "Parameters:\n" \
    "  compq (str): 'N'=no Q, 'U'=update Q, 'W'=selective via WHICHQ\n" \
    "  whichq (ndarray): Array(K) specifying Q_k computation when COMPQ='W' (int32)\n" \
    "  ws (bool): If True, perform strong stability tests\n" \
    "  k (int): Period of sequences (K >= 2)\n" \
    "  nc (int): Number of core eigenvalues\n" \
    "  kschur (int): Index where T22_kschur is upper quasi-triangular (1-based)\n" \
    "  ifst (int): Starting row index of block to move (1-based)\n" \
    "  ilst (int): Target row index for block (1-based)\n" \
    "  n (ndarray): Array(K) of matrix dimensions (int32)\n" \
    "  ni (ndarray): Array(K) of T11_k dimensions (int32)\n" \
    "  s (ndarray): Array(K) of signatures +1 or -1 (int32)\n" \
    "  t (ndarray): Flattened array containing K matrices T_k at positions IXT(k)\n" \
    "  ldt (ndarray): Array(K) of leading dimensions for T_k (int32)\n" \
    "  ixt (ndarray): Array(K) of 1-based start indices for T_k in t (int32)\n" \
    "  q (ndarray): Flattened array containing K matrices Q_k at positions IXQ(k)\n" \
    "  ldq (ndarray): Array(K) of leading dimensions for Q_k (int32)\n" \
    "  ixq (ndarray): Array(K) of 1-based start indices for Q_k in q (int32)\n" \
    "  tol (ndarray): Array(3) [c, EPS, SMLNUM] tolerance parameters\n" \
    "  ldwork (int, optional): Workspace size. -1 for query.\n" \
    "\n" \
    "Returns:\n" \
    "  (t, q, ifst, ilst, info): Updated T matrices, updated Q matrices, final IFST, final ILST, exit code\n" \
    "  info=0: success, info=1: reordering failed, info=-21: workspace too small"

#define DOC_MB03KB "Swap pairs of adjacent diagonal blocks in generalized periodic Schur form.\n" \
    "\n" \
    "Reorders diagonal blocks of formal matrix product T22_K^S(K) * ... * T22_1^S(1)\n" \
    "by swapping pairs of adjacent diagonal blocks of sizes 1 and/or 2.\n" \
    "\n" \
    "Parameters:\n" \
    "  compq (str): 'N'=no Q, 'U'=update Q, 'W'=selective via WHICHQ\n" \
    "  whichq (ndarray): Array(K) specifying Q_k computation when COMPQ='W' (int32)\n" \
    "  ws (bool): If True, perform strong stability tests\n" \
    "  k (int): Period of sequences (K >= 2)\n" \
    "  nc (int): Number of core eigenvalues\n" \
    "  kschur (int): Index where T22_kschur is upper quasi-triangular (1-based)\n" \
    "  j1 (int): First row/column of first block to swap (1-based)\n" \
    "  n1 (int): Order of first block (0, 1, or 2)\n" \
    "  n2 (int): Order of second block (0, 1, or 2)\n" \
    "  n (ndarray): Array(K) of matrix dimensions (int32)\n" \
    "  ni (ndarray): Array(K) of T11_k dimensions (int32)\n" \
    "  s (ndarray): Array(K) of signatures +1 or -1 (int32)\n" \
    "  t (ndarray): Flattened array containing K matrices T_k at positions IXT(k)\n" \
    "  ldt (ndarray): Array(K) of leading dimensions for T_k (int32)\n" \
    "  ixt (ndarray): Array(K) of 1-based start indices for T_k in t (int32)\n" \
    "  q (ndarray): Flattened array containing K matrices Q_k at positions IXQ(k)\n" \
    "  ldq (ndarray): Array(K) of leading dimensions for Q_k (int32)\n" \
    "  ixq (ndarray): Array(K) of 1-based start indices for Q_k in q (int32)\n" \
    "  tol (ndarray): Array(3) [c, EPS, SMLNUM] tolerance parameters\n" \
    "  ldwork (int, optional): Workspace size. -1 for query.\n" \
    "\n" \
    "Returns:\n" \
    "  (t, q, info): Updated T matrices, updated Q matrices, exit code\n" \
    "  info=0: success, info=1: swap rejected, info=-22: workspace too small"

#define DOC_MB03KC "Reduce 2x2 formal matrix product to periodic Hessenberg-triangular form.\n" \
    "\n" \
    "Reduces a 2-by-2 general, formal matrix product A of length K,\n" \
    "  A_K^s(K) * A_{K-1}^s(K-1) * ... * A_1^s(1),\n" \
    "to periodic Hessenberg-triangular form using K-periodic sequence of\n" \
    "elementary reflectors (Householder matrices).\n" \
    "\n" \
    "Each reflector H_k = I - tau_k * v_k * v_k' is constructed such that\n" \
    "all T_k are upper triangular except T_{khess} which is full.\n" \
    "\n" \
    "Parameters:\n" \
    "  k (int): Number of matrices in sequence (k >= 2)\n" \
    "  khess (int): Index for Hessenberg matrix (1 <= khess <= k)\n" \
    "  n (int): Order of extended matrices (n = 3 or n = 4)\n" \
    "  r (int): Starting row/column for 2x2 submatrices (r = 1 or r = n-1)\n" \
    "  s (ndarray): Signature array of length k (+1 or -1 each, int32)\n" \
    "  a (ndarray): Array of k N-by-N matrices (N x N x K, F-order)\n" \
    "  lda (int): Leading dimension of matrices (lda >= n)\n" \
    "\n" \
    "Returns:\n" \
    "  (v, tau, a):\n" \
    "    - v (ndarray): Array of length 2*k containing K 2-vectors v_k\n" \
    "    - tau (ndarray): Array of length k containing tau_k values\n" \
    "    - a (ndarray): Transformed matrices Te_k (N x N x K, F-order)"

#define DOC_MB03KD "Reorder diagonal blocks in generalized periodic Schur form.\n" \
    "\n" \
    "Reorders the diagonal blocks of a formal matrix product\n" \
    "T22_K^S(K) * T22_K-1^S(K-1) * ... * T22_1^S(1) of length K,\n" \
    "such that the M selected eigenvalues end up in the leading part.\n" \
    "\n" \
    "Parameters:\n" \
    "  compq (str): 'N'=no Q, 'I'=identity Q, 'U'=update Q\n" \
    "  strong (str): 'N'=no strong tests, 'S'=strong stability tests\n" \
    "  k (int): Period of matrix sequences (K >= 2)\n" \
    "  nc (int): Number of core eigenvalues\n" \
    "  kschur (int): Index for quasi-triangular factor (1 <= KSCHUR <= K)\n" \
    "  n (ndarray): Dimensions array (K,) int32\n" \
    "  ni (ndarray): T11 dimensions array (K,) int32\n" \
    "  s (ndarray): Signatures array (K,) int32 (+1 or -1)\n" \
    "  select (ndarray): Eigenvalue selection (NC,) bool\n" \
    "  t (ndarray): Packed T matrices (F-order)\n" \
    "  ldt (ndarray): Leading dimensions (K,) int32\n" \
    "  ixt (ndarray): Start indices in t, 1-based (K,) int32\n" \
    "  ldq (ndarray): Q leading dimensions (K,) int32\n" \
    "  ixq (ndarray): Start indices in q, 1-based (K,) int32\n" \
    "  tol (float): Tolerance (>= 10)\n" \
    "  q (ndarray, optional): Packed Q matrices (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (t, q, m, info):\n" \
    "    t: Reordered T matrices\n" \
    "    q: Orthogonal transformation matrices\n" \
    "    m: Number of selected eigenvalues reordered\n" \
    "    info: 0=success, <0=param error, 1=reordering failed"

#define DOC_MB03KE "Solve small periodic Sylvester-like equations.\n" \
    "\n" \
    "Solves periodic Sylvester-like equations (PSLE):\n" \
    "  op(A(i))*X(i)   + isgn*X(i+1)*op(B(i)) = -scale*C(i), S(i) =  1\n" \
    "  op(A(i))*X(i+1) + isgn*X(i)  *op(B(i)) = -scale*C(i), S(i) = -1\n" \
    "\n" \
    "for i = 1, ..., K, where op(A) means A or A**T, for the K-periodic\n" \
    "matrix sequence X(i) = X(i+K). A(i) are M-by-M and B(i) are N-by-N,\n" \
    "with 1 <= M, N <= 2.\n" \
    "\n" \
    "Parameters:\n" \
    "  trana (bool): If True, op(A) = A**T; otherwise op(A) = A\n" \
    "  tranb (bool): If True, op(B) = B**T; otherwise op(B) = B\n" \
    "  isgn (int): Sign variant: 1 or -1\n" \
    "  k (int): Period of sequences (k >= 2)\n" \
    "  m (int): Order of A matrices (1 <= m <= 2)\n" \
    "  n (int): Order of B matrices (1 <= n <= 2)\n" \
    "  s (ndarray): Array of K signatures (+1 or -1 each, int32)\n" \
    "  a (ndarray): Array of K M-by-M matrices (length M*M*K)\n" \
    "  b (ndarray): Array of K N-by-N matrices (length N*N*K)\n" \
    "  c (ndarray): Array of K M-by-N matrices C(i) (length M*N*K)\n" \
    "  ldwork (int, optional): Workspace size. -1 for query.\n" \
    "\n" \
    "Returns:\n" \
    "  (x, scale, info):\n" \
    "    - x (ndarray): Solution sequence X(i) (length M*N*K)\n" \
    "    - scale (float): Scale factor (<= 1) to avoid overflow\n" \
    "    - info (int): 0=success, -21=ldwork too small, 1=scaled"

#define DOC_MB03LD "Eigenvalues and right deflating subspace of real skew-Hamiltonian/Hamiltonian pencil.\n" \
    "\n" \
    "Computes eigenvalues of aS - bH with S = [[A,D],[E,A']] (D,E skew-symmetric)\n" \
    "and H = [[B,F],[G,-B']] (F,G symmetric). Optionally computes orthonormal\n" \
    "basis of right deflating subspace for eigenvalues with negative real part.\n" \
    "\n" \
    "Parameters:\n" \
    "  compq (str): 'N'=eigenvalues only, 'C'=compute deflating subspace\n" \
    "  orth (str): 'P'=QR with pivoting, 'S'=SVD (only if compq='C')\n" \
    "  n (int): Order of pencil (n >= 0, must be even)\n" \
    "  a (ndarray): N/2-by-N/2 matrix A\n" \
    "  de (ndarray): N/2-by-(N/2+1) packed D and E matrices\n" \
    "  b (ndarray): N/2-by-N/2 matrix B\n" \
    "  fg (ndarray): N/2-by-(N/2+1) packed F and G matrices\n" \
    "\n" \
    "Returns:\n" \
    "  (a, de, b, fg, neig, q, alphar, alphai, beta, info):\n" \
    "    a, de, b, fg: Modified matrices\n" \
    "    neig: Number of eigenvalues with negative real part\n" \
    "    q: N-by-NEIG deflating subspace (if compq='C' and neig>0)\n" \
    "    alphar, alphai, beta: Eigenvalue components (lambda=alpha/beta)\n" \
    "    info: 0=success, 1=MB04BD/HD QZ failed, 2=MB04HD/MB03DD failed, 3=singular, 4=SVD failed, 5=eigenvalues may be inaccurate"

#define DOC_MB03LF "Eigenvalues and deflating subspace of skew-Hamiltonian/Hamiltonian pencil in factored form.\n" \
    "\n" \
    "Computes eigenvalues of aS - bH with S = J*Z'*J'*Z (factored) and H = [[B,F],[G,-B']].\n" \
    "Optionally computes orthogonal basis of right deflating subspace (compq='C') and\n" \
    "companion subspace (compu='C') for eigenvalues with strictly negative real part.\n" \
    "\n" \
    "Parameters:\n" \
    "  compq (str): 'N'=no deflating subspace, 'C'=compute and store in Q\n" \
    "  compu (str): 'N'=no companion subspace, 'C'=compute and store in U\n" \
    "  orth (str): 'P'=QR with pivoting, 'S'=SVD (only if compq='C' or compu='C')\n" \
    "  z (ndarray): N-by-N factor Z in S = J*Z'*J'*Z\n" \
    "  b (ndarray): N/2-by-N/2 matrix B\n" \
    "  fg (ndarray): N/2-by-(N/2+1) packed G (lower tri) and F (upper tri cols 2:N/2+1)\n" \
    "  n (int, optional): Order of pencil (derived from z if not given)\n" \
    "\n" \
    "Returns:\n" \
    "  (z_out, neig, q, u, alphar, alphai, beta, iwarn, info):\n" \
    "    z_out: Transformed upper triangular matrix Z11\n" \
    "    neig: Number of eigenvalues with negative real part\n" \
    "    q: N-by-NEIG deflating subspace (if compq='C')\n" \
    "    u: N-by-NEIG companion subspace (if compu='C')\n" \
    "    alphar, alphai, beta: Eigenvalue components (lambda=alpha/beta)\n" \
    "    iwarn: 0=ok, 1=some eigenvalues may be unreliable\n" \
    "    info: 0=success, 1=QZ failed, 2=QZ iteration failed, 3=singular, 4=SVD failed"

#define DOC_MB03LP "Eigenvalues and right deflating subspace of real skew-Hamiltonian/Hamiltonian pencil (panel-based).\n" \
    "\n" \
    "Same as MB03LD but applies transformations on panels of columns for better\n" \
    "performance on large matrices. Uses MB04BP instead of MB04BD internally.\n" \
    "\n" \
    "Computes eigenvalues of aS - bH with S = [[A,D],[E,A']] (D,E skew-symmetric)\n" \
    "and H = [[B,F],[G,-B']] (F,G symmetric). Optionally computes orthonormal\n" \
    "basis of right deflating subspace for eigenvalues with negative real part.\n" \
    "\n" \
    "Parameters:\n" \
    "  compq (str): 'N'=eigenvalues only, 'C'=compute deflating subspace\n" \
    "  orth (str): 'P'=QR with pivoting, 'S'=SVD (only if compq='C')\n" \
    "  a (ndarray): N/2-by-N/2 matrix A\n" \
    "  de (ndarray): N/2-by-(N/2+1) packed D and E matrices\n" \
    "  b (ndarray): N/2-by-N/2 matrix B\n" \
    "  fg (ndarray): N/2-by-(N/2+1) packed F and G matrices\n" \
    "\n" \
    "Returns:\n" \
    "  (a, de, b, fg, neig, q, alphar, alphai, beta, info):\n" \
    "    a, de, b, fg: Modified matrices\n" \
    "    neig: Number of eigenvalues with negative real part\n" \
    "    q: 2N-by-NEIG deflating subspace (if compq='C' and neig>0), or None\n" \
    "    alphar, alphai, beta: Eigenvalue components (lambda=alpha/beta)\n" \
    "    info: 0=success, 1=MB04BP/HD QZ failed, 2=MB04HD/MB03DD failed, 3=singular, 4=SVD failed, 5=eigenvalues may be inaccurate"

#define DOC_MB03LZ "Eigenvalues and right deflating subspace of complex skew-Hamiltonian/Hamiltonian pencil.\n" \
    "\n" \
    "Computes eigenvalues of aS - bH with S = [[A,D],[E,A']] (D,E skew-Hermitian)\n" \
    "and H = [[B,F],[G,-B']] (F,G Hermitian). Optionally computes orthonormal\n" \
    "basis of right deflating subspace for eigenvalues with negative real part.\n" \
    "\n" \
    "Parameters:\n" \
    "  compq (str): 'N'=eigenvalues only, 'C'=compute deflating subspace\n" \
    "  orth (str): 'P'=QR with pivoting, 'S'=SVD (only if compq='C')\n" \
    "  n (int): Order of pencil (n >= 0, must be even)\n" \
    "  a (ndarray): N/2-by-N/2 complex matrix A\n" \
    "  de (ndarray): N/2-by-(N/2+1) packed D and E matrices\n" \
    "  b (ndarray): N/2-by-N/2 complex matrix B\n" \
    "  fg (ndarray): N/2-by-(N/2+1) packed F and G matrices\n" \
    "\n" \
    "Returns:\n" \
    "  (a, de, b, fg, q, alphar, alphai, beta, neig, info):\n" \
    "    a, de, b, fg: Modified matrices (if compq='C')\n" \
    "    q: N-by-NEIG deflating subspace (if compq='C' and neig>0)\n" \
    "    alphar, alphai, beta: Eigenvalue components (lambda=alpha/beta)\n" \
    "    neig: Number of eigenvalues with negative real part\n" \
    "    info: 0=success, 1=MB04FD failed, 2=ZHGEQZ failed, 3=SVD failed, 4=singular"

#define DOC_MB03MD "Compute upper bound for L singular values of bidiagonal matrix.\n" \
    "\n" \
    "Computes an upper bound THETA using bisection such that the bidiagonal\n" \
    "matrix J has precisely L singular values <= THETA + TOL.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Order of bidiagonal matrix J (n >= 0)\n" \
    "  l (int): Number of singular values <= bound (0 <= l <= n)\n" \
    "  theta (float): Initial estimate (negative for default)\n" \
    "  q (ndarray): Diagonal elements (n,)\n" \
    "  e (ndarray): Superdiagonal elements (n-1,)\n" \
    "  q2 (ndarray): Squared diagonal elements (n,)\n" \
    "  e2 (ndarray): Squared superdiagonal elements (n-1,)\n" \
    "  pivmin (float): Minimum pivot value\n" \
    "  tol (float): Tolerance for singular value coincidence\n" \
    "  reltol (float): Relative tolerance for bisection\n" \
    "\n" \
    "Returns:\n" \
    "  (theta, l, iwarn, info):\n" \
    "    - theta (float): Computed upper bound\n" \
    "    - l (int): May be increased if multiplicity > 1\n" \
    "    - iwarn (int): 0=ok, 1=L increased due to coinciding singular values\n" \
    "    - info (int): 0=success, <0=parameter error"

#define DOC_MB03OD "Incremental rank estimation for QR factorization.\n" \
    "\n" \
    "Parameters:\n" \
    "  m (int): Number of rows\n" \
    "  n (int): Number of columns\n" \
    "  a (ndarray): Matrix array (column-major, shape (m,n))\n" \
    "  rcond (float): Rank threshold\n" \
    "  svlmax (float): Parent matrix singular value estimate\n" \
    "  jobqr (str, optional): 'Q' = perform QR (default), 'N' = use existing\n" \
    "\n" \
    "Returns:\n" \
    "  (jpvt, rank, sval, info): Pivot indices, rank, singular values, exit code"

#define DOC_MB03OY "Matrix rank determination by incremental condition estimation.\n" \
    "\n" \
    "Parameters:\n" \
    "  m (int): Number of rows\n" \
    "  n (int): Number of columns\n" \
    "  a (ndarray): Matrix array (column-major, shape (m,n))\n" \
    "  rcond (float): Threshold for rank determination\n" \
    "  svlmax (float): Estimate of largest singular value\n" \
    "\n" \
    "Returns:\n" \
    "  (a, rank, info, sval, jpvt, tau): QR factorization results"

#define DOC_MB03PD "Rank-revealing RQ factorization with row pivoting.\n" \
    "\n" \
    "Computes (optionally) a rank-revealing RQ factorization P*A = R*Q with\n" \
    "row pivoting, determining effective rank via incremental condition estimation.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobrq (str): 'R' = perform RQ factorization, 'N' = use existing\n" \
    "  m (int): Number of rows (>= 0)\n" \
    "  n (int): Number of columns (>= 0)\n" \
    "  a (ndarray): Input matrix (m x n, F-order)\n" \
    "  rcond (float): Reciprocal condition threshold (>= 0)\n" \
    "  svlmax (float): Max singular value estimate (>= 0)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, rank, info, sval, jpvt, tau):\n" \
    "    - a: RQ factorization result\n" \
    "    - rank: Effective rank (order of R22)\n" \
    "    - info: Exit code (0 = success, <0 = param error)\n" \
    "    - sval: Singular value estimates (3 values)\n" \
    "    - jpvt: Row pivot indices (1-based)\n" \
    "    - tau: Elementary reflector factors"

#define DOC_MB03PY "Rank-revealing RQ factorization with row pivoting.\n" \
    "\n" \
    "Computes truncated RQ factorization P*A = R*Q with row pivoting,\n" \
    "determining effective rank via incremental condition estimation.\n" \
    "\n" \
    "Parameters:\n" \
    "  m (int): Number of rows (>= 0)\n" \
    "  n (int): Number of columns (>= 0)\n" \
    "  a (ndarray): Input matrix (m x n, F-order)\n" \
    "  rcond (float): Reciprocal condition threshold (0 <= rcond <= 1)\n" \
    "  svlmax (float): Max singular value estimate (>= 0)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, rank, info, sval, jpvt, tau):\n" \
    "    - a: RQ factorization result\n" \
    "    - rank: Effective rank\n" \
    "    - info: Exit code (0 = success)\n" \
    "    - sval: Singular value estimates (3 values)\n" \
    "    - jpvt: Row pivot indices (1-based)\n" \
    "    - tau: Elementary reflector factors"

#define DOC_MB03QD "Reorder eigenvalues in quasi-triangular matrix.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous, 'D' for discrete\n" \
    "  stdom (str): 'S' for stable, 'U' for unstable\n" \
    "  jobu (str): 'I' to initialize U, 'U' to update U\n" \
    "  a (ndarray): Quasi-triangular matrix (n x n, F-order)\n" \
    "  nlow (int): Lowest block index (1-based)\n" \
    "  nsup (int): Highest block index (1-based)\n" \
    "  alpha (float): Stability boundary\n" \
    "\n" \
    "Returns:\n" \
    "  (a, u, ndim, info): Reordered A, updated U, invariant subspace dim, exit code"

#define DOC_MB03QG "Reorder diagonal blocks of upper quasi-triangular matrix pencil.\n" \
    "\n" \
    "Reorders the diagonal blocks of a principal subpencil of an upper\n" \
    "quasi-triangular matrix pencil A-lambda*E together with their\n" \
    "generalized eigenvalues, by constructing orthogonal similarity\n" \
    "transformations UT and VT.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  stdom (str): 'S' for stability domain, 'U' for instability domain\n" \
    "  jobu (str): 'I' to initialize U to identity, 'U' to update given U\n" \
    "  jobv (str): 'I' to initialize V to identity, 'U' to update given V\n" \
    "  a (ndarray): N-by-N Schur form matrix (F-order, modified in place)\n" \
    "  e (ndarray): N-by-N upper triangular matrix (F-order, modified in place)\n" \
    "  nlow (int): Lower boundary index for subpencil (1-based)\n" \
    "  nsup (int): Upper boundary index for subpencil (1-based)\n" \
    "  alpha (float): Boundary of domain of interest (>=0 for DICO='D')\n" \
    "  u (ndarray, optional): N-by-N transformation matrix (when jobu='U')\n" \
    "  v (ndarray, optional): N-by-N transformation matrix (when jobv='U')\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, u, v, ndim, info):\n" \
    "    a (ndarray): Reordered Schur form matrix\n" \
    "    e (ndarray): Reordered upper triangular matrix\n" \
    "    u (ndarray): Accumulated transformation matrix UT\n" \
    "    v (ndarray): Accumulated transformation matrix VT\n" \
    "    ndim (int): Number of eigenvalues in domain of interest\n" \
    "    info (int): 0=success, <0=param error, 1=block boundary error, 2=swap failed"

#define DOC_MB03QV "Compute eigenvalues of upper quasi-triangular matrix pencil.\n" \
    "\n" \
    "Computes generalized eigenvalues of (S, T) where S is upper\n" \
    "quasi-triangular and T is upper triangular.\n" \
    "\n" \
    "Parameters:\n" \
    "  s (ndarray): Upper quasi-triangular matrix S (n x n, F-order)\n" \
    "  t (ndarray): Upper triangular matrix T (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (alphar, alphai, beta, info):\n" \
    "    alphar (ndarray): Real parts of eigenvalue numerators\n" \
    "    alphai (ndarray): Imaginary parts of eigenvalue numerators\n" \
    "    beta (ndarray): Eigenvalue denominators\n" \
    "    info (int): 0 = success, <0 = -i means i-th argument invalid"

#define DOC_MB03QW "Reduce 2-by-2 diagonal block pair of quasi-triangular pencil.\n" \
    "\n" \
    "Computes eigenvalues of a selected 2-by-2 diagonal block pair of an upper\n" \
    "quasi-triangular pencil, reduces it to standard form, and splits it if\n" \
    "eigenvalues are real.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Order of matrices A, E, U, V (n >= 2)\n" \
    "  l (int): Position of the 2-by-2 block (1 <= l < n)\n" \
    "  a (ndarray): Upper quasi-triangular matrix A (n x n, F-order)\n" \
    "  e (ndarray): Upper triangular matrix E (n x n, F-order)\n" \
    "  u (ndarray): Transformation matrix U (n x n, F-order)\n" \
    "  v (ndarray): Transformation matrix V (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, u, v, alphar, alphai, beta, info):\n" \
    "    a (ndarray): Transformed matrix A\n" \
    "    e (ndarray): Transformed matrix E\n" \
    "    u (ndarray): Updated transformation matrix U*UT\n" \
    "    v (ndarray): Updated transformation matrix V*VT\n" \
    "    alphar (ndarray): Real parts of eigenvalue numerators (2,)\n" \
    "    alphai (ndarray): Imaginary parts of eigenvalue numerators (2,)\n" \
    "    beta (ndarray): Eigenvalue denominators (2,)\n" \
    "    info (int): 0 = success, <0 = -i means i-th argument invalid"

#define DOC_MB03QX "Compute eigenvalues of upper quasi-triangular matrix.\n" \
    "\n" \
    "Extracts eigenvalues from diagonal and 2x2 blocks of quasi-triangular matrix.\n" \
    "\n" \
    "Parameters:\n" \
    "  t (ndarray): Upper quasi-triangular matrix (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (wr, wi, info): Real parts, imaginary parts, exit code"

#define DOC_MB03QY "Process 2x2 diagonal block of quasi-triangular matrix.\n" \
    "\n" \
    "Computes eigenvalues of a selected 2x2 diagonal block, reduces to\n" \
    "standard form, and splits in case of real eigenvalues.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Order of matrices A and U. N >= 2.\n" \
    "  l (int): Position of block (1 <= L < N). 1-based.\n" \
    "  a (ndarray): Upper quasi-triangular matrix (n x n, F-order)\n" \
    "  u (ndarray): Transformation matrix (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, u, e1, e2, info): Transformed matrices, eigenvalue parts, exit code"

#define DOC_MB03RD "Reduce real Schur form matrix to block-diagonal form.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobx (str): 'N' no accumulation, 'U' accumulate transformations\n" \
    "  sort (str): 'N'/'C'/'S'/'B' - sorting/clustering options\n" \
    "  a (ndarray): Real Schur form matrix (n x n, F-order)\n" \
    "  pmax (float): Max element bound for transformations (>= 1.0)\n" \
    "  x (ndarray, optional): Transformation matrix (n x n, F-order)\n" \
    "  tol (float, optional): Clustering tolerance\n" \
    "\n" \
    "Returns:\n" \
    "  (a, x, nblcks, blsize, wr, wi, info): Block-diagonal A, transformation X,\n" \
    "   number of blocks, block sizes, real/imag eigenvalues, exit code"

#define DOC_MB03RW "Solve complex Sylvester equation -AX + XB = C.\n" \
    "\n" \
    "Solves the Sylvester equation -AX + XB = C where A (M-by-M) and B (N-by-N)\n" \
    "are complex upper triangular matrices in Schur form. Aborts if any element\n" \
    "of X exceeds PMAX in absolute value.\n" \
    "\n" \
    "Parameters:\n" \
    "  m (int): Order of A, number of rows of C and X (M >= 0)\n" \
    "  n (int): Order of B, number of columns of C and X (N >= 0)\n" \
    "  pmax (float): Upper bound for absolute value of X elements\n" \
    "  a (ndarray): M-by-M complex upper triangular matrix A (F-order)\n" \
    "  b (ndarray): N-by-N complex upper triangular matrix B (F-order)\n" \
    "  c (ndarray): M-by-N RHS matrix C (F-order, modified in place)\n" \
    "\n" \
    "Returns:\n" \
    "  (x, info):\n" \
    "    x (ndarray): Solution matrix X (M-by-N, complex)\n" \
    "    info (int): 0=success, 1=element of X exceeds PMAX, 2=perturbed values used"

#define DOC_MB03RX "Reorder diagonal blocks of a real Schur form matrix.\n" \
    "\n" \
    "Moves the diagonal block at position KU to position KL using\n" \
    "orthogonal similarity transformations (LAPACK DTREXC).\n" \
    "\n" \
    "Parameters:\n" \
    "  jobv (str): 'N' no accumulation, 'V' accumulate in X\n" \
    "  kl (int): Target position (1-based)\n" \
    "  ku (int): Source position (1-based)\n" \
    "  a (ndarray): Real Schur matrix (n x n, F-order)\n" \
    "  x (ndarray): Transformation matrix (n x n, F-order)\n" \
    "  wr (ndarray): Real parts of eigenvalues (n,)\n" \
    "  wi (ndarray): Imaginary parts of eigenvalues (n,)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, x, wr, wi, ku_out): Reordered matrix, transformation, eigenvalues, new ku"

#define DOC_MB03RY "Solve Sylvester equation -AX + XB = C with norm bound.\n" \
    "\n" \
    "Solves -AX + XB = C where A (M-by-M) and B (N-by-N) are in real Schur form.\n" \
    "Aborts if any elementary submatrix of X has infinity norm > PMAX.\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): M-by-M matrix in real Schur form, column-major\n" \
    "  b (ndarray): N-by-N matrix in real Schur form, column-major\n" \
    "  c (ndarray): M-by-N right-hand side matrix (modified in-place)\n" \
    "  pmax (float): Upper bound for infinity norm of elementary submatrices\n" \
    "\n" \
    "Returns:\n" \
    "  (x, info): Solution X (same array as c), exit code (0=ok, 1=norm exceeded)"

#define DOC_MB03RZ "Reduce complex Schur form matrix to block-diagonal form.\n" \
    "\n" \
    "Reduces an upper triangular complex matrix A (Schur form) to a\n" \
    "block-diagonal form using well-conditioned non-unitary similarity\n" \
    "transformations. The condition numbers of the transformations are\n" \
    "roughly bounded by PMAX.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobx (str): N = no transformation accumulation, U = accumulate in X\n" \
    "  sort (str): N = no reordering, S = cluster eigenvalues,\n" \
    "              C = closest-neighbour, B = both\n" \
    "  a (ndarray): N-by-N complex upper triangular matrix (F-order)\n" \
    "  pmax (float): Upper bound for transformation elements (>= 1.0)\n" \
    "  tol (float): Tolerance for eigenvalue clustering\n" \
    "  x (ndarray, optional): N-by-N transformation matrix (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a_out, x_out, nblcks, blsize, w, info):\n" \
    "    a_out (ndarray): Block-diagonal matrix (N-by-N, complex)\n" \
    "    x_out (ndarray): Accumulated transformation (N-by-N, complex)\n" \
    "    nblcks (int): Number of diagonal blocks\n" \
    "    blsize (ndarray): Block sizes (nblcks,)\n" \
    "    w (ndarray): Eigenvalues (N,)\n" \
    "    info (int): 0=success, <0=param error"

#define DOC_MB03SD "Eigenvalues of a square-reduced Hamiltonian matrix.\n" \
    "\n" \
    "Computes the eigenvalues of an N-by-N square-reduced Hamiltonian matrix H'\n" \
    "where eigenvalues of H' are the square roots of eigenvalues of A'' = A'^2 + G'Q'.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobscl (str): 'N' = no balancing, 'S' = scale for equilibration\n" \
    "  a (ndarray): N-by-N upper left block A' of H' (F-order)\n" \
    "  qg (ndarray): N-by-(N+1) array with Q' (lower tri col 0) and G' (upper tri cols 1:N+1)\n" \
    "\n" \
    "Returns:\n" \
    "  (wr, wi, info):\n" \
    "    wr (ndarray): Real parts of N eigenvalues with non-negative real part\n" \
    "    wi (ndarray): Imaginary parts of N eigenvalues\n" \
    "    info (int): 0=success, <0=param error, >0=DHSEQR failed at i-th eigenvalue"

#define DOC_MB03TD "Reorder diagonal blocks of (skew-)Hamiltonian Schur form.\n" \
    "\n" \
    "Reorders a matrix X in skew-Hamiltonian Schur form (X = [A G; 0 A^T], G = -G^T)\n" \
    "or Hamiltonian Schur form (X = [A G; 0 -A^T], G = G^T) so that selected\n" \
    "eigenvalues appear in leading diagonal blocks of A.\n" \
    "\n" \
    "Parameters:\n" \
    "  typ (str): 'S'=skew-Hamiltonian, 'H'=Hamiltonian\n" \
    "  compu (str): 'U'=update U1 and U2, 'N'=do not update\n" \
    "  select (ndarray): Boolean array (n,), eigenvalue selection\n" \
    "  lower (ndarray): Boolean array (n,), controls which eigenvalue copy\n" \
    "  a (ndarray): N-by-N upper quasi-triangular matrix in Schur form (F-order)\n" \
    "  g (ndarray): N-by-N symmetric (H) or skew-symmetric (S) matrix (F-order)\n" \
    "  u1 (ndarray): N-by-N matrix U1 (F-order, if compu='U')\n" \
    "  u2 (ndarray): N-by-N matrix U2 (F-order, if compu='U')\n" \
    "\n" \
    "Returns:\n" \
    "  (a, g, u1, u2, wr, wi, m, info):\n" \
    "    a (ndarray): Reordered matrix A in Schur form\n" \
    "    g (ndarray): Updated matrix G\n" \
    "    u1 (ndarray): Updated matrix U1\n" \
    "    u2 (ndarray): Updated matrix U2\n" \
    "    wr (ndarray): Real parts of eigenvalues (n,)\n" \
    "    wi (ndarray): Imaginary parts of eigenvalues (n,)\n" \
    "    m (int): Dimension of specified invariant subspace\n" \
    "    info (int): 0=success, <0=param error, 1=reordering failed"

#define DOC_MB03TS "Swap diagonal blocks in (skew-)Hamiltonian Schur form.\n" \
    "\n" \
    "Swaps diagonal blocks A11 and A22 of order 1 or 2 in the upper quasi-triangular\n" \
    "matrix A contained in a skew-Hamiltonian or Hamiltonian matrix.\n" \
    "\n" \
    "Parameters:\n" \
    "  isham (bool): True=Hamiltonian, False=skew-Hamiltonian matrix\n" \
    "  wantu (bool): True=update U1 and U2, False=do not update\n" \
    "  a (ndarray): N-by-N upper quasi-triangular matrix in Schur form (F-order)\n" \
    "  g (ndarray): N-by-N symmetric (ISHAM) or skew-symmetric (!ISHAM) matrix (F-order)\n" \
    "  u1 (ndarray): N-by-N matrix U1 (F-order, if wantu)\n" \
    "  u2 (ndarray): N-by-N matrix U2 (F-order, if wantu)\n" \
    "  j1 (int): Index of first row of first block A11 (1-based)\n" \
    "  n1 (int): Order of first block A11 (0, 1, or 2)\n" \
    "  n2 (int): Order of second block A22 (0, 1, or 2)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, g, u1, u2, info):\n" \
    "    a (ndarray): Reordered matrix A\n" \
    "    g (ndarray): Updated matrix G\n" \
    "    u1 (ndarray): Updated matrix U1\n" \
    "    u2 (ndarray): Updated matrix U2\n" \
    "    info (int): 0=success, 1=swap rejected (result would be far from Schur form)"

#define DOC_MB03UD "Singular value decomposition of upper triangular matrix.\n" \
    "\n" \
    "Computes SVD: A = Q*S*P' where Q, P are orthogonal and S is diagonal\n" \
    "with non-negative singular values in descending order.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Order of matrix A\n" \
    "  a (ndarray): Upper triangular matrix A (n x n, F-order)\n" \
    "               If jobp='V', returns P' on exit\n" \
    "  jobq (str, optional): 'V' to compute Q, 'N' otherwise (default 'N')\n" \
    "  jobp (str, optional): 'V' to compute P', 'N' otherwise (default 'N')\n" \
    "  ldwork (int, optional): Workspace size (-1 for query, default auto)\n" \
    "\n" \
    "Returns:\n" \
    "  (sv, p, q, info): Singular values, right vectors P', left vectors Q, exit code\n" \
    "  - sv: array of singular values (descending order)\n" \
    "  - p: P' matrix if jobp='V', else None\n" \
    "  - q: Q matrix if jobq='V', else None\n" \
    "  - info: 0=success, <0=invalid param, >0=convergence failure"

#define DOC_MB03VD "Reduce product of p matrices to periodic Hessenberg form.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Order of square matrices\n" \
    "  p (int): Number of matrices in product\n" \
    "  ilo (int): Lower index for reduction (1-based)\n" \
    "  ihi (int): Upper index for reduction (1-based)\n" \
    "  a (ndarray): 3D matrix array (n,n,p), column-major\n" \
    "\n" \
    "Returns:\n" \
    "  (a, tau, info): Reduced matrices, tau factors, exit code"

#define DOC_MB03VW "Reduce periodic product to Hessenberg-triangular form (unblocked).\n" \
    "\n" \
    "Reduces A(:,:,1)^S(1) * ... * A(:,:,K)^S(K) to Hessenberg-triangular form.\n" \
    "H-th matrix becomes upper Hessenberg, others upper triangular.\n" \
    "\n" \
    "Parameters:\n" \
    "  compq (str): 'N'=no Q, 'U'=update Q, 'I'=init Q to I, 'P'=use qind\n" \
    "  qind (ndarray, optional): Array(k) for partial Q generation with compq='P'\n" \
    "  triu (str): 'N'=only neg signature, 'A'=all N-1 matrices triangularized\n" \
    "  n (int): Order of each factor\n" \
    "  k (int): Number of factors\n" \
    "  h (int): Which factor becomes Hessenberg (1..k or auto)\n" \
    "  ilo (int): Lower active submatrix bound (1-based)\n" \
    "  ihi (int): Upper active submatrix bound (1-based)\n" \
    "  s (ndarray): Signature array(k) with values 1 or -1\n" \
    "  a (ndarray): 3D array (n,n,k) of factors\n" \
    "  q (ndarray, optional): 3D array (n,n,k) for Q matrices (compq='U')\n" \
    "\n" \
    "Returns:\n" \
    "  (a, q, h, info): Reduced factors, orthogonal matrices, Hessenberg index, exit code"

#define DOC_MB03VY "Generate orthogonal matrices from periodic Hessenberg reduction.\n" \
    "\n" \
    "Generates Q_1, Q_2, ..., Q_p orthogonal matrices from reflectors\n" \
    "produced by mb03vd.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Order of square matrices\n" \
    "  p (int): Number of transformation matrices\n" \
    "  ilo (int): Lower index from mb03vd (1-based)\n" \
    "  ihi (int): Upper index from mb03vd (1-based)\n" \
    "  a (ndarray): 3D array (n,n,p) with Householder vectors from mb03vd\n" \
    "  tau (ndarray): 2D array (n-1,p) with scalar factors from mb03vd\n" \
    "\n" \
    "Returns:\n" \
    "  (q, info): Orthogonal matrices Q_j, exit code"

#define DOC_MB03WA "Swap adjacent diagonal blocks in periodic real Schur form.\n" \
    "\n" \
    "Swaps adjacent diagonal blocks A11*B11 and A22*B22 of size 1-by-1 or 2-by-2\n" \
    "in an upper (quasi) triangular matrix product A*B by orthogonal equivalence\n" \
    "transformation.\n" \
    "\n" \
    "(A, B) must be in periodic real Schur canonical form, i.e., A is block upper\n" \
    "triangular with 1-by-1 and 2-by-2 diagonal blocks, and B is upper triangular.\n" \
    "\n" \
    "Optionally, the matrices Q and Z of generalized Schur vectors are updated:\n" \
    "    Q(in) * A(in) * Z(in)' = Q(out) * A(out) * Z(out)'\n" \
    "    Z(in) * B(in) * Q(in)' = Z(out) * B(out) * Q(out)'\n" \
    "\n" \
    "Parameters:\n" \
    "  wantq (bool): True to update Q, False if Q not required\n" \
    "  wantz (bool): True to update Z, False if Z not required\n" \
    "  n1 (int): Order of first block A11*B11 (0, 1, or 2)\n" \
    "  n2 (int): Order of second block A22*B22 (0, 1, or 2)\n" \
    "  a (ndarray): (N1+N2)-by-(N1+N2) matrix A (F-order)\n" \
    "  b (ndarray): (N1+N2)-by-(N1+N2) matrix B (F-order)\n" \
    "  q (ndarray): (N1+N2)-by-(N1+N2) orthogonal matrix Q (if wantq)\n" \
    "  z (ndarray): (N1+N2)-by-(N1+N2) orthogonal matrix Z (if wantz)\n" \
    "\n" \
    "Returns:\n" \
    "  (a_out, b_out, q_out, z_out, info):\n" \
    "    a_out: Reordered matrix A\n" \
    "    b_out: Reordered matrix B\n" \
    "    q_out: Updated orthogonal matrix Q (if wantq)\n" \
    "    z_out: Updated orthogonal matrix Z (if wantz)\n" \
    "    info: 0=success, 1=swap rejected (would be too far from Schur form)"

#define DOC_MB03WD "Compute periodic Schur decomposition of matrix product.\n" \
    "\n" \
    "Computes Schur decomposition and eigenvalues of H = H_1*H_2*...*H_p\n" \
    "with H_1 upper Hessenberg, H_2...H_p upper triangular.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'E' = eigenvalues only, 'S' = full Schur form\n" \
    "  compz (str): 'N' = no Z, 'I' = init Z to I, 'V' = update Z\n" \
    "  n (int): Order of matrices\n" \
    "  p (int): Number of matrices\n" \
    "  ilo (int): Lower index (1-based)\n" \
    "  ihi (int): Upper index (1-based)\n" \
    "  iloz (int): Lower Z row index (1-based)\n" \
    "  ihiz (int): Upper Z row index (1-based)\n" \
    "  h (ndarray): 3D array (n,n,p), column-major\n" \
    "  z (ndarray, optional): 3D array for COMPZ='V'\n" \
    "\n" \
    "Returns:\n" \
    "  (h, z, wr, wi, info): Schur form, transforms, eigenvalues, exit code"

#define DOC_MB03WX "Compute eigenvalues of a product of matrices in periodic Schur form.\n" \
    "\n" \
    "Computes eigenvalues of T = T_1*T_2*...*T_p where T_1 is upper\n" \
    "quasi-triangular (real Schur form) and T_2, ..., T_p are upper triangular.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Order of matrix T (n >= 0)\n" \
    "  p (int): Number of matrices in product (p >= 1)\n" \
    "  t (ndarray): 3D array (ldt1, ldt2, p). T(*,*,1) is upper quasi-triangular,\n" \
    "               T(*,*,j) for j > 1 are upper triangular.\n" \
    "\n" \
    "Returns:\n" \
    "  (wr, wi, info): Real and imaginary parts of eigenvalues, exit code"

#define DOC_MB03XD "Compute eigenvalues of Hamiltonian matrix.\n" \
    "\n" \
    "Computes eigenvalues of H = [A, G; Q, -A'] where G=G' and Q=Q'.\n" \
    "Uses symplectic URV and periodic Schur decompositions.\n" \
    "\n" \
    "Parameters:\n" \
    "  balanc (str): 'N' = none, 'P' = permute, 'S' = scale, 'B' = both\n" \
    "  job (str): 'E' = eigenvalues only, 'S' = Schur form, 'G' = full\n" \
    "  jobu (str): 'N' = no U, 'U' = compute U\n" \
    "  jobv (str): 'N' = no V, 'V' = compute V\n" \
    "  n (int): Order of matrix A\n" \
    "  a (ndarray): Matrix A, shape (n,n)\n" \
    "  qg (ndarray): Q (lower) and G (upper), shape (n,n+1)\n" \
    "\n" \
    "Returns:\n" \
    "  (S, T, QG, U1, U2, V1, V2, wr, wi, ilo, scale, info)"

#define DOC_MB03XP "Periodic Schur decomposition of A*B product.\n" \
    "\n" \
    "Computes Q'*A*Z = S (Schur form) and Z'*B*Q = T (upper triangular).\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'E' = eigenvalues only, 'S' = Schur form\n" \
    "  compq (str): 'N' = no Q, 'I' = init Q to I, 'V' = update Q\n" \
    "  compz (str): 'N' = no Z, 'I' = init Z to I, 'V' = update Z\n" \
    "  n (int): Order of matrices\n" \
    "  ilo (int): Lower index (1-based)\n" \
    "  ihi (int): Upper index (1-based)\n" \
    "  a (ndarray): Upper Hessenberg matrix A\n" \
    "  b (ndarray): Upper triangular matrix B\n" \
    "\n" \
    "Returns:\n" \
    "  (S, T, Q, Z, alphar, alphai, beta, info)"

#define DOC_MB03XS "Eigenvalues and real skew-Hamiltonian Schur form of a skew-Hamiltonian matrix.\n" \
    "\n" \
    "Computes eigenvalues and real skew-Hamiltonian Schur form of W = [[A, G], [Q, A^T]]\n" \
    "where G, Q are N-by-N skew-symmetric. Computes orthogonal symplectic U such that\n" \
    "U^T W U = [[Aout, Gout], [0, Aout^T]] with Aout in Schur canonical form.\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): N-by-N matrix A (F-order)\n" \
    "  qg (ndarray): N-by-(N+1) packed Q (cols 0:N lower tri) and G (cols 1:N+1 upper tri)\n" \
    "  jobu (str, optional): 'N'=don't compute U (default), 'U'=compute U1 and U2\n" \
    "\n" \
    "Returns (jobu='N'):\n" \
    "  (a_out, qg_out, wr, wi, info):\n" \
    "    a_out: Aout in Schur form\n" \
    "    qg_out: Gout in cols 1:N+1, Q part zeroed\n" \
    "    wr, wi: Real and imaginary parts of eigenvalues\n" \
    "    info: 0=success, <0=param error, >0=DHSEQR failed\n" \
    "\n" \
    "Returns (jobu='U'):\n" \
    "  (a_out, qg_out, u1, u2, wr, wi, info):\n" \
    "    u1, u2: N-by-N matrices where U = [[U1, U2], [-U2, U1]]"

#define DOC_MB03XU "Panel reduction for blocked Hamiltonian matrix.\n" \
    "\n" \
    "Reduces 2*NB columns and rows of a (K+2N)-by-(K+2N) Hamiltonian\n" \
    "matrix H = [[op(A), G], [Q, op(B)]] using orthogonal symplectic\n" \
    "transformations. Auxiliary routine for MB04TB.\n" \
    "\n" \
    "Parameters:\n" \
    "  ltra (bool): If True, op(A) = A', else op(A) = A\n" \
    "  ltrb (bool): If True, op(B) = B', else op(B) = B\n" \
    "  n (int): Order of Q (n >= 0)\n" \
    "  k (int): Offset of reduction (k >= 0)\n" \
    "  nb (int): Number of columns/rows to reduce (0 <= nb < n)\n" \
    "  a (ndarray): Matrix A (F-order)\n" \
    "  b (ndarray): Matrix B (F-order)\n" \
    "  g (ndarray): Matrix G (F-order)\n" \
    "  q (ndarray): Matrix Q (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (A, B, G, Q, XA, XB, XG, XQ, YA, YB, YG, YQ, CSL, CSR, TAUL, TAUR, info):\n" \
    "    Modified matrices, update matrices, Givens rotations, reflectors, exit code"

#define DOC_MB03XZ "Eigenvalues of complex Hamiltonian matrix.\n" \
    "\n" \
    "Computes eigenvalues of H = [[A, G], [Q, -A^H]] where G, Q are Hermitian.\n" \
    "Due to structure, eigenvalues appear in pairs (lambda, -conj(lambda)).\n" \
    "Uses embedding to real skew-Hamiltonian and structured Schur form.\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): N-by-N complex matrix A (F-order)\n" \
    "  qg (ndarray): N-by-(N+1) packed Q (cols 0:N lower tri) and G (cols 1:N+1 upper tri)\n" \
    "  balanc (str): 'N'=no balancing, 'P'=permute, 'S'=scale, 'B'=both\n" \
    "  job (str): 'E'=eigenvalues only, 'S'=compute Schur form Sc, 'G'=compute Sc and Gc\n" \
    "  jobu (str): 'N'=don't compute U (default), 'U'=compute unitary symplectic U\n" \
    "\n" \
    "Returns:\n" \
    "  (wr, wi, sc, gc, u1, u2, ilo, scale, info):\n" \
    "    wr, wi: Real and imaginary parts of 2N eigenvalues\n" \
    "    sc: 2N-by-2N upper triangular Schur form (if job='S' or 'G')\n" \
    "    gc: 2N-by-2N Hermitian Gc matrix (if job='G')\n" \
    "    u1, u2: 2N-by-2N blocks of unitary symplectic U (if job!='E' and jobu='U')\n" \
    "    ilo: Balancing index (1 if balanc='N')\n" \
    "    scale: Balancing scale factors\n" \
    "    info: 0=success, <0=param error, >0=QR failed, 2N+1=2x2 block failed"

#define DOC_MB03YA "Annihilate subdiagonal entries of Hessenberg matrix.\n" \
    "\n" \
    "Auxiliary routine for MB03XP and MB03YD.\n" \
    "\n" \
    "Parameters:\n" \
    "  wantt (bool): Compute full Schur form\n" \
    "  wantq (bool): Accumulate Q matrix\n" \
    "  wantz (bool): Accumulate Z matrix\n" \
    "  ilo, ihi (int): Active block bounds (1-based)\n" \
    "  iloq, ihiq (int): Q/Z row bounds (1-based)\n" \
    "  pos (int): Position of zero diagonal in B (1-based)\n" \
    "  a, b, q, z (ndarray): Matrices\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, q, z, info): Transformed matrices and exit code"

#define DOC_MB03YD "Periodic QR iteration for eigenvalues of periodic Hessenberg matrix.\n" \
    "\n" \
    "Computes eigenvalues of A*inv(B) where A is upper Hessenberg and B\n" \
    "is upper triangular, using periodic QR iteration.\n" \
    "\n" \
    "Parameters:\n" \
    "  wantt (bool): Compute full Schur form (True) or eigenvalues only (False)\n" \
    "  wantq (bool): Update matrix Q\n" \
    "  wantz (bool): Update matrix Z\n" \
    "  n (int): Order of matrices\n" \
    "  ilo, ihi (int): Active block bounds (1-based)\n" \
    "  iloq, ihiq (int): Q/Z row bounds (1-based)\n" \
    "  a, b, q, z (ndarray): Input matrices (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (alphar, alphai, beta, info): Eigenvalue numerators/denominators and exit code"

#define DOC_MB03YT "Periodic Schur factorization of 2x2 matrix pair.\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): 2x2 matrix A\n" \
    "  b (ndarray): 2x2 upper triangular matrix B\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, alphar, alphai, beta, csl, snl, csr, snr):\n" \
    "    Transformed matrices, eigenvalue info, rotation parameters"

#define DOC_MB03ZA "Reorder eigenvalues in periodic Schur form and compute stable subspace.\n" \
    "\n" \
    "Computes orthogonal matrices Ur and Vr so that:\n" \
    "  Vr' * A * Ur = [A11  A12]    Ur' * B * Vr = [B11  B12]\n" \
    "                 [0    A22]                   [0    B22]\n" \
    "is in periodic Schur form with eigenvalues of A11*B11 forming selected cluster.\n" \
    "\n" \
    "Also computes orthogonal W transforming [0, -A11; B11, 0] to block triangular\n" \
    "form with eigenvalues of R11 having positive real part.\n" \
    "\n" \
    "Parameters:\n" \
    "  compc (str): 'U' to update C, 'N' to skip\n" \
    "  compu (str): 'U' to update U1/U2, 'N' to skip\n" \
    "  compv (str): 'V' to update V1/V2, 'N' to skip\n" \
    "  compw (str): 'N' W not needed, 'I' init to identity, 'V' accumulate\n" \
    "  which (str): 'A' all eigenvalues, 'S' selected by select\n" \
    "  select (ndarray): Boolean array specifying selected eigenvalues\n" \
    "  a (ndarray): Upper quasi-triangular matrix A (F-order)\n" \
    "  b (ndarray): Upper triangular matrix B (F-order)\n" \
    "  c (ndarray): General matrix C (F-order)\n" \
    "  u1, u2 (ndarray): Blocks of orthogonal symplectic U\n" \
    "  v1, v2 (ndarray): Blocks of orthogonal symplectic V\n" \
    "  w (ndarray): Orthogonal transformation matrix\n" \
    "\n" \
    "Returns:\n" \
    "  (r22, b_out, c_out, u1_out, u2_out, v1_out, v2_out, w_out, wr, wi, m, info):\n" \
    "    r22: M-by-M R22 matrix stored in A\n" \
    "    wr, wi: Real/imaginary parts of eigenvalues of R11\n" \
    "    m: Number of selected eigenvalues\n" \
    "    info: 0=success, 1-4=various algorithm failures"

#define DOC_MB03ZD "Compute stable/unstable invariant subspaces for Hamiltonian matrix.\n" \
    "\n" \
    "Computes orthonormal bases US and/or UU for the stable and/or unstable\n" \
    "invariant subspaces of a Hamiltonian matrix using the output of MB03XD.\n" \
    "The Hamiltonian is represented in product periodic Schur form.\n" \
    "\n" \
    "Parameters:\n" \
    "  which (str): 'A' all eigenvalues, 'S' selected eigenvalues\n" \
    "  meth (str): 'Q' QR-based, 'L' 2*n vectors, 'R' extended (for extractor)\n" \
    "  stab (str): 'S' stable only, 'U' unstable only, 'B' both\n" \
    "  balanc (str): 'N' no balancing, 'P'/'S'/'B' permute/scale/both\n" \
    "  ortbal (str): 'B' balance before orthonormalization, 'A' after\n" \
    "  n (int): Order of Hamiltonian matrix (n >= 0)\n" \
    "  mm (int): Number of vectors (must be >= 2*n)\n" \
    "  ilo (int): Lower index from balancing (1 if balanc='N')\n" \
    "  scale (ndarray): Scale factors from balancing (n,)\n" \
    "  s (ndarray): Real Schur form S (n x n, F-order)\n" \
    "  t (ndarray): Upper triangular T (n x n, F-order)\n" \
    "  g (ndarray): Matrix G (n x n, F-order)\n" \
    "  u1, u2 (ndarray): Orthogonal symplectic U blocks (n x n, F-order)\n" \
    "  v1, v2 (ndarray): Orthogonal symplectic V blocks (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (m, wr, wi, us, uu, info):\n" \
    "    m: Number of selected eigenvalues\n" \
    "    wr, wi: Real/imaginary parts of eigenvalues\n" \
    "    us: Stable invariant subspace basis (mm x m, F-order)\n" \
    "    uu: Unstable invariant subspace basis (mm x m, F-order)\n" \
    "    info: 0=success, 1-5=various algorithm failures"

#define DOC_MB04AD "Compute eigenvalues of skew-Hamiltonian/Hamiltonian pencil.\n" \
    "\n" \
    "Computes eigenvalues of a real N-by-N skew-Hamiltonian/Hamiltonian pencil\n" \
    "aS - bH with S = T Z = J Z' J' Z using generalized symplectic URV\n" \
    "decomposition. Optionally computes orthogonal transformation matrices.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'E'=eigenvalues only, 'T'=also compute triangular T\n" \
    "  compq1 (str): 'N'=no Q1, 'I'=init identity, 'U'=update Q1\n" \
    "  compq2 (str): 'N'=no Q2, 'I'=init identity, 'U'=update Q2\n" \
    "  compu1 (str): 'N'=no U1, 'I'=init identity, 'U'=update U1\n" \
    "  compu2 (str): 'N'=no U2, 'I'=init identity, 'U'=update U2\n" \
    "  z (ndarray): Z matrix from skew-Hamiltonian S = T Z (n x n, F-order)\n" \
    "  h (ndarray): Hamiltonian matrix H (n x n, F-order)\n" \
    "  q1 (ndarray, optional): Orthogonal Q1 matrix (n x n, F-order)\n" \
    "  q2 (ndarray, optional): Orthogonal Q2 matrix (n x n, F-order)\n" \
    "  u11 (ndarray, optional): Symplectic U1 block (n/2 x n/2, F-order)\n" \
    "  u12 (ndarray, optional): Symplectic U1 block (n/2 x n/2, F-order)\n" \
    "  u21 (ndarray, optional): Symplectic U2 block (n/2 x n/2, F-order)\n" \
    "  u22 (ndarray, optional): Symplectic U2 block (n/2 x n/2, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (t, z, h, q1, q2, u11, u12, u21, u22, alphar, alphai, beta, info):\n" \
    "    t: Upper triangular factor T (n x n)\n" \
    "    z: Transformed Z matrix (n x n)\n" \
    "    h: Transformed H matrix (n x n)\n" \
    "    q1, q2: Orthogonal matrices (n x n)\n" \
    "    u11, u12, u21, u22: Symplectic transformation blocks (n/2 x n/2)\n" \
    "    alphar, alphai, beta: Generalized eigenvalues (n/2,)\n" \
    "    info: 0=success, 3=QZ did not converge"

#define DOC_MB04AZ "Eigenvalues of complex skew-Hamiltonian/Hamiltonian pencil.\n" \
    "\n" \
    "Computes eigenvalues of a complex N-by-N skew-Hamiltonian/Hamiltonian\n" \
    "pencil aS - bH, with S = J*Z^H*J^T*Z and H = [[B,F],[G,-B^H]].\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'E'=eigenvalues only, 'T'=Schur form and eigenvalues\n" \
    "  compq (str): 'N'=no Q, 'C'=compute Q\n" \
    "  compu (str): 'N'=no U, 'C'=compute U\n" \
    "  z (ndarray): Complex N-by-N matrix Z (F-order)\n" \
    "  b (ndarray): Complex M-by-M matrix B, M=N/2 (F-order)\n" \
    "  fg (ndarray): Complex M-by-(M+1) with F diagonal and G upper (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  For JOB='E': (alphar, alphai, beta, info)\n" \
    "  For JOB='T': (z, b, fg, d, c, q, u, alphar, alphai, beta, info)\n" \
    "    z: Transformed Z matrix (N x N)\n" \
    "    b: Transformed B matrix (N x N)\n" \
    "    fg: Transformed FG matrix (N x N)\n" \
    "    d: D matrix (N x N)\n" \
    "    c: C matrix (N x N)\n" \
    "    q: Unitary Q matrix (2N x 2N) if COMPQ='C'\n" \
    "    u: Unitary U matrix (N x 2N) if COMPU='C'\n" \
    "    alphar, alphai, beta: Generalized eigenvalues (N,)\n" \
    "    info: 0=success, positive=failure"

#define DOC_MB04BD "Eigenvalues of skew-Hamiltonian/Hamiltonian pencil.\n" \
    "\n" \
    "Computes eigenvalues of a real N-by-N skew-Hamiltonian/Hamiltonian pencil\n" \
    "aS - bH with\n" \
    "      (  A  D  )         (  C  V  )\n" \
    "  S = (        ) and H = (        )\n" \
    "      (  E  A' )         (  W -C' )\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'E'=eigenvalues only, 'T'=triangular form and eigenvalues\n" \
    "  compq1 (str): 'N'=no Q1, 'I'=init identity, 'U'=update Q1\n" \
    "  compq2 (str): 'N'=no Q2, 'I'=init identity, 'U'=update Q2\n" \
    "  a (ndarray): Matrix A (n/2 x n/2, F-order)\n" \
    "  de (ndarray): Matrix containing E (lower) and D (upper) (n/2 x n/2+1, F-order)\n" \
    "  c1 (ndarray): Matrix C1=C (n/2 x n/2, F-order)\n" \
    "  vw (ndarray): Matrix containing W (lower) and V (upper) (n/2 x n/2+1, F-order)\n" \
    "  q1 (ndarray, optional): Orthogonal Q1 matrix (n x n, F-order, for COMPQ1='U')\n" \
    "\n" \
    "Returns:\n" \
    "  (a, de, c1, vw, q1, q2, b, f, c2, alphar, alphai, beta, info):\n" \
    "    a, de, c1, vw: Transformed input matrices\n" \
    "    q1: Orthogonal transformation Q1 (n x n)\n" \
    "    q2: Orthogonal transformation Q2 (n x n)\n" \
    "    b: Output matrix B (n/2 x n/2)\n" \
    "    f: Skew-symmetric matrix F (n/2 x n/2)\n" \
    "    c2: Output matrix C2 (n/2 x n/2)\n" \
    "    alphar, alphai, beta: Generalized eigenvalues (n/2,)\n" \
    "    info: 0=success, 1=eigenvalue problem, 2=QZ failed, 3=warning"

#define DOC_MB04BP "Eigenvalues of skew-Hamiltonian/Hamiltonian pencil (block algorithm).\n" \
    "\n" \
    "Block algorithm variant of MB04BD for better performance on large matrices.\n" \
    "Computes eigenvalues of a real N-by-N skew-Hamiltonian/Hamiltonian pencil\n" \
    "aS - bH. For small N (<=250), delegates to MB04BD.\n" \
    "\n" \
    "Same interface and output as MB04BD.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'E'=eigenvalues only, 'T'=triangular form and eigenvalues\n" \
    "  compq1 (str): 'N'=no Q1, 'I'=init identity, 'U'=update Q1\n" \
    "  compq2 (str): 'N'=no Q2, 'I'=init identity, 'U'=update Q2\n" \
    "  a (ndarray): Matrix A (n/2 x n/2, F-order)\n" \
    "  de (ndarray): Matrix containing E (lower) and D (upper) (n/2 x n/2+1, F-order)\n" \
    "  c1 (ndarray): Matrix C1=C (n/2 x n/2, F-order)\n" \
    "  vw (ndarray): Matrix containing W (lower) and V (upper) (n/2 x n/2+1, F-order)\n" \
    "  q1 (ndarray, optional): Orthogonal Q1 matrix (n x n, F-order, for COMPQ1='U')\n" \
    "\n" \
    "Returns:\n" \
    "  (a, de, c1, vw, q1, q2, b, f, c2, alphar, alphai, beta, info):\n" \
    "    a, de, c1, vw: Transformed input matrices\n" \
    "    q1: Orthogonal transformation Q1 (n x n)\n" \
    "    q2: Orthogonal transformation Q2 (n x n)\n" \
    "    b: Output matrix B (n/2 x n/2)\n" \
    "    f: Skew-symmetric matrix F (n/2 x n/2)\n" \
    "    c2: Output matrix C2 (n/2 x n/2)\n" \
    "    alphar, alphai, beta: Generalized eigenvalues (n/2,)\n" \
    "    info: 0=success, 1=eigenvalue problem, 2=QZ failed, 3=warning"

#define DOC_MB04BZ "Eigenvalues of complex skew-Hamiltonian/Hamiltonian pencil.\n" \
    "\n" \
    "Computes eigenvalues of a complex N-by-N skew-Hamiltonian/Hamiltonian\n" \
    "pencil aS - bH, with S = [[A,D],[E,A^H]] and H = [[B,F],[G,-B^H]],\n" \
    "using an embedding to a real skew-Hamiltonian/skew-Hamiltonian pencil.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'E'=eigenvalues only, 'T'=Schur form and eigenvalues\n" \
    "  compq (str): 'N'=no Q, 'C'=compute unitary Q\n" \
    "  a (ndarray): Complex N/2-by-N/2 matrix A (F-order)\n" \
    "  de (ndarray): Complex N/2-by-(N/2+1) with E (lower) and D (upper) (F-order)\n" \
    "  b (ndarray): Complex N/2-by-N/2 matrix B (F-order)\n" \
    "  fg (ndarray): Complex N/2-by-(N/2+1) with G (lower) and F (upper) (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  For JOB='E' COMPQ='N': (alphar, alphai, beta, info)\n" \
    "  For JOB='T' COMPQ='N': (a, de, b, fg, alphar, alphai, beta, info)\n" \
    "  For JOB='T' COMPQ='C': (a, de, b, fg, q, alphar, alphai, beta, info)\n" \
    "    a: Upper triangular BA matrix (N x N)\n" \
    "    de: Skew-Hermitian BD matrix (N x N)\n" \
    "    b: Upper triangular BB matrix (N x N)\n" \
    "    fg: Hermitian BF matrix (N x N)\n" \
    "    q: Unitary Q matrix (2N x 2N)\n" \
    "    alphar, alphai, beta: Generalized eigenvalues (N,)\n" \
    "    info: 0=success, 1=MB04FD QZ failed, 2=ZHGEQZ failed, 3=pencil singular"

#define DOC_MB04CD "Reduce skew-Hamiltonian/Hamiltonian pencil to generalized Schur form.\n" \
    "\n" \
    "Computes transformed matrices A, B and D using orthogonal matrices Q1, Q2, Q3\n" \
    "for a real N-by-N regular pencil aA*B - bD where:\n" \
    "- A11, A22, B11, B22, D12 are upper triangular\n" \
    "- D21 is upper quasi-triangular\n" \
    "- The transformed pencil is in generalized Schur form\n" \
    "\n" \
    "Parameters:\n" \
    "  compq1 (str): Q1 computation: 'N'=none, 'I'=init, 'U'=update\n" \
    "  compq2 (str): Q2 computation: 'N'=none, 'I'=init, 'U'=update\n" \
    "  compq3 (str): Q3 computation: 'N'=none, 'I'=init, 'U'=update\n" \
    "  a (ndarray): N-by-N matrix A (F-order)\n" \
    "  b (ndarray): N-by-N matrix B (F-order)\n" \
    "  d (ndarray): N-by-N matrix D (F-order)\n" \
    "  q1 (ndarray, optional): N-by-N matrix Q1 for update mode (F-order)\n" \
    "  q2 (ndarray, optional): N-by-N matrix Q2 for update mode (F-order)\n" \
    "  q3 (ndarray, optional): N-by-N matrix Q3 for update mode (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, d, q1, q2, q3, info): Transformed matrices, orthogonal Q matrices, exit code\n" \
    "  info=0: success, info=1-4: algorithm warnings"

#define DOC_MB04DB "Apply inverse balancing transformation to [V1; sgn*V2].\n" \
    "\n" \
    "Applies from the left the inverse of a balancing transformation\n" \
    "computed by MB04DP to the matrix [V1; sgn*V2].\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'N'=none, 'P'=permute only, 'S'=scale only, 'B'=both\n" \
    "  sgn (str): 'P'=sgn=+1, 'N'=sgn=-1\n" \
    "  ilo (int): ILO from MB04DP (1 <= ilo <= n+1)\n" \
    "  lscale (ndarray): Left scaling factors from MB04DP (n,)\n" \
    "  rscale (ndarray): Right scaling factors from MB04DP (n,)\n" \
    "  v1 (ndarray): N-by-M matrix V1 (F-order)\n" \
    "  v2 (ndarray): N-by-M matrix V2 (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (v1, v2, info): Transformed matrices and exit code"

#define DOC_MB04DD "Balance a real Hamiltonian matrix.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'N' = none, 'P' = permute, 'S' = scale, 'B' = both\n" \
    "  a (ndarray): N-by-N matrix A (F-order)\n" \
    "  qg (ndarray): N-by-(N+1) matrix with lower tri Q, upper tri G\n" \
    "\n" \
    "Returns:\n" \
    "  (a, qg, ilo, scale, info): Balanced matrices and scaling info"

#define DOC_MB04DI "Apply inverse balancing transformation to [V1; sgn*V2].\n" \
    "\n" \
    "Applies the inverse of a balancing transformation computed by MB04DD\n" \
    "or MB04DS to the matrix [V1; sgn*V2] where sgn is +1 or -1.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'N'=none, 'P'=permute only, 'S'=scale only, 'B'=both\n" \
    "  sgn (str): 'P'=sgn=+1, 'N'=sgn=-1\n" \
    "  ilo (int): ILO from MB04DD or MB04DS (1 <= ilo <= n+1)\n" \
    "  scale (ndarray): Scaling factors from MB04DD or MB04DS (n,)\n" \
    "  v1 (ndarray): N-by-M matrix V1 (F-order)\n" \
    "  v2 (ndarray): N-by-M matrix V2 (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (v1, v2, info): Transformed matrices and exit code"

#define DOC_MB04DL "Balance a real matrix pencil (A, B).\n" \
    "\n" \
    "Balances a pair of N-by-N real matrices (A,B) by permuting to isolate\n" \
    "eigenvalues and applying diagonal equivalence transformations.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'N'=none, 'P'=permute only, 'S'=scale only, 'B'=both\n" \
    "  n (int): Order of matrices A and B\n" \
    "  thresh (float): Threshold for scaling (>=0 fixed, <0 automatic)\n" \
    "  a (ndarray): N-by-N matrix A (F-order, modified in place)\n" \
    "  b (ndarray): N-by-N matrix B (F-order, modified in place)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, ilo, ihi, lscale, rscale, dwork, iwarn, info):\n" \
    "    Balanced matrices, indices, scaling factors, workspace, and exit codes"

#define DOC_MB04DP "Balance a real skew-Hamiltonian/Hamiltonian pencil.\n" \
    "\n" \
    "Balances the pencil aS - bH where S=[[A,D],[E,A']] (skew-Hamiltonian)\n" \
    "and H=[[C,V],[W,-C']] (Hamiltonian), with D,E skew-symmetric and V,W symmetric.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'N'=none, 'P'=permute only, 'S'=scale only, 'B'=both\n" \
    "  n (int): Order of matrices A, D, E, C, V, W\n" \
    "  thresh (float): Threshold for scaling (>=0 fixed, <0 automatic)\n" \
    "  a (ndarray): N-by-N matrix A (F-order, modified in place)\n" \
    "  de (ndarray): N-by-(N+1) array with E (lower) and D (upper) (F-order)\n" \
    "  c (ndarray): N-by-N matrix C (F-order, modified in place)\n" \
    "  vw (ndarray): N-by-(N+1) array with W (lower) and V (upper) (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, de, c, vw, ilo, lscale, rscale, dwork, iwarn, info):\n" \
    "    Balanced matrices, ILO, scaling factors, workspace, and exit codes"

#define DOC_MB04DS "Balance a real skew-Hamiltonian matrix.\n" \
    "\n" \
    "Balances a 2N-by-2N skew-Hamiltonian matrix S = [[A,G],[Q,A']] where\n" \
    "A is N-by-N and G, Q are N-by-N skew-symmetric matrices.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'N'=none, 'P'=permute only, 'S'=scale only, 'B'=both\n" \
    "  n (int): Order of matrix A (n >= 0)\n" \
    "  a (ndarray): N-by-N matrix A (F-order, modified in place)\n" \
    "  qg (ndarray): N-by-(N+1) array with Q (strictly lower, cols 1:N)\n" \
    "                and G (strictly upper, cols 2:N+1) (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, qg, ilo, scale, info):\n" \
    "    Balanced matrices, ILO (number of deflated eigenvalues + 1),\n" \
    "    scaling factors, and exit code"

#define DOC_MB04DY "Symplectic scaling of a Hamiltonian matrix.\n" \
    "\n" \
    "Performs symplectic scaling on a Hamiltonian matrix H = [[A,G],[Q,-A']]\n" \
    "where A is N-by-N and G, Q are symmetric N-by-N matrices.\n" \
    "\n" \
    "Scaling strategies:\n" \
    "  'S': Symplectic scaling using DGEBAL + equilibration\n" \
    "  '1'/'O': 1-norm scaling by power of machine base\n" \
    "  'N': No scaling\n" \
    "\n" \
    "Parameters:\n" \
    "  jobscl (str): 'S'=symplectic, '1'/'O'=1-norm, 'N'=none\n" \
    "  a (ndarray): N-by-N matrix A (F-order, modified in place)\n" \
    "  qg (ndarray): N-by-(N+1) array with Q (lower tri, cols 0:N-1)\n" \
    "                and G (upper tri, cols 1:N) (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, qg, d, info):\n" \
    "    Scaled matrices, scaling factors (N for 'S', 1 for '1'/'O'),\n" \
    "    and exit code"

#define DOC_MB04DZ "Balance a complex Hamiltonian matrix.\n" \
    "\n" \
    "Balances a complex Hamiltonian matrix H = [[A,G],[Q,-A^H]] where A is NxN\n" \
    "and G, Q are NxN Hermitian matrices. Involves permuting to isolate\n" \
    "eigenvalues and diagonal similarity transformations.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'N'=none, 'P'=permute only, 'S'=scale only, 'B'=both\n" \
    "  a (ndarray): N-by-N complex matrix A (F-order, modified in place)\n" \
    "  qg (ndarray): N-by-(N+1) complex array with lower tri of Q and\n" \
    "                upper tri of G (F-order, modified in place)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, qg, ilo, scale, info):\n" \
    "    Balanced matrices, ILO (number of deflated eigenvalues + 1),\n" \
    "    scaling factors (real array), and exit code"

#define DOC_MB04ED "Eigenvalues of skew-Hamiltonian/skew-Hamiltonian pencil.\n" \
    "\n" \
    "Computes eigenvalues of aS - bT where S = J*Z'*J'*Z and T = [[B,F],[G,B']]\n" \
    "with J = [[0,I],[-I,0]]. Optionally computes structured Schur form.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'E'=eigenvalues only, 'T'=also structured Schur form\n" \
    "  compq (str): 'N'=don't compute Q, 'I'=initialize and compute Q\n" \
    "  compu (str): 'N'=don't compute U, 'I'=init U, 'U'=update U0\n" \
    "  z (ndarray): N-by-N matrix Z (F-order)\n" \
    "  b (ndarray): N/2-by-N/2 matrix B (F-order)\n" \
    "  fg (ndarray): N/2-by-(N/2+1) array with G (strictly lower) and F (strictly upper)\n" \
    "  u1 (ndarray, optional): N/2-by-N/2 for COMPU='U'\n" \
    "  u2 (ndarray, optional): N/2-by-N/2 for COMPU='U'\n" \
    "\n" \
    "Returns:\n" \
    "  (z, b, fg, q, u1, u2, alphar, alphai, beta, info):\n" \
    "    Transformed matrices, orthogonal Q, symplectic U blocks,\n" \
    "    eigenvalue components (lambda = (alphar+alphai*j)/beta), info"

#define DOC_MB04FD "Eigenvalues of skew-Hamiltonian/skew-Hamiltonian pencil (real).\n" \
    "\n" \
    "Computes eigenvalues of aS - bT where S = [[A,D],[E,A']] and T = [[B,F],[G,B']]\n" \
    "with D, E, F, G skew-symmetric. Optionally computes structured Schur form.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'E'=eigenvalues only, 'T'=also structured Schur form\n" \
    "  compq (str): 'N'=don't compute Q, 'I'=initialize and compute Q, 'U'=update Q\n" \
    "  a (ndarray): N/2-by-N/2 matrix A (F-order)\n" \
    "  de (ndarray): N/2-by-(N/2+1) array with E (strictly lower) and D (strictly upper)\n" \
    "  b (ndarray): N/2-by-N/2 matrix B (F-order)\n" \
    "  fg (ndarray): N/2-by-(N/2+1) array with G (strictly lower) and F (strictly upper)\n" \
    "  q (ndarray, optional): N-by-N orthogonal matrix Q for COMPQ='U'\n" \
    "  n (int, optional): Order of pencil (must be even, default from matrix sizes)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, de, b, fg, q, alphar, alphai, beta, info):\n" \
    "    Transformed matrices, orthogonal Q,\n" \
    "    eigenvalue components (lambda = (alphar+alphai*j)/beta), info"

#define DOC_MB04FP "Eigenvalues of skew-Hamiltonian/skew-Hamiltonian pencil (panel-based).\n" \
    "\n" \
    "Panel-based version of MB04FD for better performance on large matrices.\n" \
    "Computes eigenvalues of aS - bT where S = [[A,D],[E,A']] and T = [[B,F],[G,B']]\n" \
    "with D, E, F, G skew-symmetric. For small matrices (N/2 <= 32), delegates to MB04FD.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'E'=eigenvalues only, 'T'=also structured Schur form\n" \
    "  compq (str): 'N'=don't compute Q, 'I'=initialize and compute Q, 'U'=update Q\n" \
    "  a (ndarray): N/2-by-N/2 matrix A (F-order)\n" \
    "  de (ndarray): N/2-by-(N/2+1) array with E (strictly lower) and D (strictly upper)\n" \
    "  b (ndarray): N/2-by-N/2 matrix B (F-order)\n" \
    "  fg (ndarray): N/2-by-(N/2+1) array with G (strictly lower) and F (strictly upper)\n" \
    "  q (ndarray, optional): N-by-N orthogonal matrix Q for COMPQ='U'\n" \
    "  n (int, optional): Order of pencil (must be even, default from matrix sizes)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, de, b, fg, q, alphar, alphai, beta, iwork, info):\n" \
    "    Transformed matrices, orthogonal Q,\n" \
    "    eigenvalue components (lambda = (alphar+alphai*j)/beta),\n" \
    "    iwork (inaccurate eigenvalue info), info"

#define DOC_MB04GD "RQ factorization with row pivoting.\n" \
    "\n" \
    "Computes P*A = R*Q where P is a permutation matrix, R is upper triangular\n" \
    "(or trapezoidal if m >= n), and Q is orthogonal.\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): M-by-N matrix A (F-order), modified in place\n" \
    "  jpvt (ndarray): Pivot indices (M,), dtype=int32\n" \
    "                  On entry: if jpvt[i] != 0, row i is constrained to bottom\n" \
    "                  On exit: jpvt[i] = k means row i of P*A was row k of A\n" \
    "\n" \
    "Returns:\n" \
    "  (a, jpvt, tau, info): Modified matrix, pivot indices, reflector factors, exit code"

#define DOC_MB04HD "Reduce skew-Hamiltonian/Hamiltonian pencil to generalized Schur form.\n" \
    "\n" \
    "Reduces a special block (anti-)diagonal skew-Hamiltonian/Hamiltonian pencil\n" \
    "aAB - bBB with A = diag(A11,A22) and B = anti-diag(B21,B12) to generalized\n" \
    "Schur form using orthogonal transformations Q1 and Q2.\n" \
    "\n" \
    "Parameters:\n" \
    "  compq1 (str): 'N'=no Q1, 'I'=initialize Q1, 'U'=update Q1\n" \
    "  compq2 (str): 'N'=no Q2, 'I'=initialize Q2, 'U'=update Q2\n" \
    "  a (ndarray): N-by-N matrix A with block diagonal structure (F-order)\n" \
    "  b (ndarray): N-by-N matrix B with block anti-diagonal structure (F-order)\n" \
    "  q1 (ndarray, optional): Orthogonal transformation Q1 (N-by-N, F-order)\n" \
    "  q2 (ndarray, optional): Orthogonal transformation Q2 (N-by-N, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, q1, q2, info): Transformed matrices and exit code\n" \
    "  info=0: success\n" \
    "  info=1: QZ algorithm failed\n" \
    "  info=2: warning - eigenvalue reordering failed\n" \
    "  info=3: reordering in periodic QZ failed\n" \
    "  info=4: illegal structure (deflation window too small)"

#define DOC_MB04ID "QR factorization of matrix with lower-left zero triangle.\n" \
    "\n" \
    "Computes A = Q*R where A has p-by-min(p,m) zero triangle in lower-left.\n" \
    "Optionally applies Q' to matrix B.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Number of rows of A\n" \
    "  m (int): Number of columns of A\n" \
    "  p (int): Order of zero triangle\n" \
    "  a (ndarray): Matrix A (n x m, F-order), modified in place\n" \
    "  b (ndarray, optional): Matrix B (n x l, F-order), modified in place\n" \
    "  l (int, optional): Number of columns of B (default 0)\n" \
    "  ldwork (int, optional): Workspace size (-1 for query, default auto)\n" \
    "\n" \
    "Returns:\n" \
    "  If l>0: (b, tau, info) - transformed B, Householder factors, exit code\n" \
    "  If l=0: (tau, info) - Householder factors, exit code\n" \
    "  If ldwork=-1: adds optimal workspace size to tuple"

#define DOC_MB04IY "Apply orthogonal transformations from MB04ID to matrix C.\n" \
    "\n" \
    "Applies Q or Q' to matrix C, where Q is product of elementary reflectors\n" \
    "as returned by MB04ID (special structure for lower-left zero triangle).\n" \
    "\n" \
    "Parameters:\n" \
    "  side (str): 'L' for left (Q*C or Q'*C), 'R' for right (C*Q or C*Q')\n" \
    "  trans (str): 'N' for Q, 'T' for Q'\n" \
    "  a (ndarray): Reflector storage (lda x k, F-order), modified but restored\n" \
    "  tau (ndarray): Reflector scalar factors (k,)\n" \
    "  c (ndarray): Matrix C (n x m, F-order), modified in place\n" \
    "  p (int, optional): Order of zero triangle (default 0)\n" \
    "\n" \
    "Returns:\n" \
    "  (c, info): Transformed matrix and exit code"

#define DOC_MB04IZ "QR factorization of complex matrix with lower-left zero triangle.\n" \
    "\n" \
    "Computes A = Q*R where A is n-by-m complex matrix with p-by-min(p,m)\n" \
    "zero triangle in the lower left corner. Optionally applies Q^H to B.\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): Complex matrix A (n x m, F-order), modified in place\n" \
    "  p (int): Order of zero triangle (>= 0)\n" \
    "  b (ndarray, optional): Complex matrix B (n x l, F-order), modified in place\n" \
    "  lzwork (int, optional): Workspace size (-1 for query)\n" \
    "\n" \
    "Returns:\n" \
    "  dict with keys:\n" \
    "    a: Factored matrix (R in upper triangle, reflectors below)\n" \
    "    tau: Complex scalar factors of reflectors (min(n,m),)\n" \
    "    info: Exit code (0 = success)\n" \
    "    zwork_opt: Optimal workspace size\n" \
    "    b: Transformed B (only if b provided)"

#define DOC_MB04JD "LQ factorization of matrix with upper-right zero triangle.\n" \
    "\n" \
    "Computes A = L*Q where A is n-by-m matrix with min(n,p)-by-p\n" \
    "zero triangle in the upper right corner. Optionally applies Q to matrix B\n" \
    "from the right. Exploits the zero structure.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Number of rows of A (>= 0)\n" \
    "  m (int): Number of columns of A (>= 0)\n" \
    "  p (int): Order of zero triangle (>= 0)\n" \
    "  a (ndarray): Matrix A (n x m, F-order), modified in place\n" \
    "  b (ndarray, optional): Matrix B (l x m, F-order), modified in place\n" \
    "  l (int, optional): Number of rows of B (default 0)\n" \
    "  ldwork (int, optional): Workspace size (-1 for query, default auto)\n" \
    "\n" \
    "Returns:\n" \
    "  If l>0: (a, b, tau, info) - factored A, transformed B, reflectors, exit code\n" \
    "  If l=0: (a, tau, info) - factored A, Householder factors, exit code\n" \
    "  If ldwork=-1: adds optimal workspace size to tuple"

#define DOC_MB04KD "QR factorization of special structured block matrix.\n" \
    "\n" \
    "Computes Q' * [[R],[A B]] = [[R_bar C],[0 D]]\n" \
    "\n" \
    "Parameters:\n" \
    "  uplo (str): 'U' if A is upper trapezoidal, 'F' if A is full\n" \
    "  n (int): Order of R and R_bar\n" \
    "  m (int): Number of columns of B, C, D\n" \
    "  p (int): Number of rows of A, B, D\n" \
    "  r (ndarray): Upper triangular matrix R (n x n, F-order)\n" \
    "  a (ndarray): Matrix A (p x n, F-order)\n" \
    "  b (ndarray): Matrix B (p x m, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (r_bar, a_out, d, c, tau): Transformed matrices and Householder factors"

#define DOC_MB04LD "LQ factorization of special structured block matrix.\n" \
    "\n" \
    "Computes [[L A],[0 B]] * Q = [[L_bar 0],[C D]]\n" \
    "\n" \
    "Useful for Kalman filter square-root covariance updates.\n" \
    "\n" \
    "Parameters:\n" \
    "  uplo (str): 'L' if A is lower trapezoidal, 'F' if A is full\n" \
    "  n (int): Order of L and L_bar\n" \
    "  m (int): Number of columns of A, B, D\n" \
    "  p (int): Number of rows of B, C, D\n" \
    "  l (ndarray): Lower triangular matrix L (n x n, F-order)\n" \
    "  a (ndarray): Matrix A (n x m, F-order)\n" \
    "  b (ndarray): Matrix B (p x m, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (l_bar, a_out, d, c, tau): Transformed matrices and Householder factors"

#define DOC_MB04MD "Balance a general real matrix to reduce its 1-norm.\n" \
    "\n" \
    "Applies diagonal similarity transformation inv(D) * A * D iteratively\n" \
    "to make rows and columns as close in norm as possible.\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): N-by-N matrix A (F-order), balanced on exit\n" \
    "  maxred (float, optional): Max allowed reduction if zero rows/cols.\n" \
    "    If > 0, must be > 1. If <= 0, uses default 10.0.\n" \
    "\n" \
    "Returns:\n" \
    "  (a_balanced, scale, maxred_out, info):\n" \
    "    a_balanced: Balanced matrix\n" \
    "    scale: Scaling factors D(j) for j=1,...,N\n" \
    "    maxred_out: Ratio of original/balanced 1-norm (if A non-zero)\n" \
    "    info: 0=success, <0=param -info invalid"

#define DOC_MB04ND "RQ factorization of special structured block matrix.\n" \
    "\n" \
    "Calculates RQ factorization of [A R; C B] to produce [0 R_new; C_new B_new].\n" \
    "\n" \
    "Parameters:\n" \
    "  uplo (str): 'U' for upper trapezoidal A, 'F' for full A\n" \
    "  n (int): Order of R and R_new\n" \
    "  m (int): Number of rows of B and C\n" \
    "  p (int): Number of columns of A and C\n" \
    "  r (ndarray): n-by-n upper triangular R (modified in place)\n" \
    "  a (ndarray): n-by-p matrix A (modified, stores Householder vectors)\n" \
    "  b (ndarray): m-by-n matrix B (modified in place)\n" \
    "  c (ndarray): m-by-p matrix C (modified in place)\n" \
    "\n" \
    "Returns:\n" \
    "  tau (ndarray): Scalar factors of elementary reflectors"

#define DOC_MB04NY "Apply Householder reflector to matrix [A B] from the right.\n" \
    "\n" \
    "Applies H = I - tau*[1;v]*[1;v]' to m-by-(n+1) matrix [A B],\n" \
    "where A has one column.\n" \
    "\n" \
    "Parameters:\n" \
    "  m (int): Number of rows of A and B\n" \
    "  n (int): Number of columns of B\n" \
    "  v (ndarray): Householder vector (1+(n-1)*abs(incv), F-order)\n" \
    "  incv (int): Increment between elements of v\n" \
    "  tau (float): Householder scalar\n" \
    "  a (ndarray): Matrix A (m x 1, F-order, modified in place)\n" \
    "  b (ndarray): Matrix B (m x n, F-order, modified in place)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b): Updated matrices"

#define DOC_MB04OD "QR factorization of structured block matrix.\n" \
    "\n" \
    "Computes QR factorization of first block column and applies\n" \
    "transformations to second block column:\n" \
    "  Q' * [R  B] = [R_  B_]\n" \
    "       [A  C]   [0   C_]\n" \
    "\n" \
    "Parameters:\n" \
    "  uplo (str): 'U' for A upper trapezoidal, 'F' for A full\n" \
    "  n (int): Order of R\n" \
    "  m (int): Columns in B, C\n" \
    "  p (int): Rows in A, C\n" \
    "  r (ndarray): n-by-n upper triangular matrix R (F-order), modified\n" \
    "  a (ndarray): p-by-n matrix A (F-order), overwritten with Householder vectors\n" \
    "  b (ndarray): n-by-m matrix B (F-order), modified\n" \
    "  c (ndarray): p-by-m matrix C (F-order), modified\n" \
    "\n" \
    "Returns:\n" \
    "  (r, a, b, c, tau): Updated matrices and Householder scalars"

#define DOC_MB04OW "Performs a QR factorization update.\n" \
    "\n" \
    "Parameters:\n" \
    "  m (int): Rows of U1\n" \
    "  n (int): Rows of T\n" \
    "  p (int): Columns of B, C\n" \
    "  a (ndarray): Matrix U (lda x m+n), modified in place\n" \
    "  t (ndarray): Matrix T (ldt x n), modified in place\n" \
    "  x (ndarray): Vector x (m+n), modified in place\n" \
    "  b (ndarray): Matrix B (ldb x p), modified in place\n" \
    "  c (ndarray): Matrix C (ldc x p), modified in place\n" \
    "  d (ndarray): Vector d (p), modified in place\n" \
    "  incx (int, optional): Increment for x\n" \
    "  incd (int, optional): Increment for d\n" \
    "\n" \
    "Returns:\n" \
    "  (a, t, x, b, c, d): Modified arrays"

#define DOC_MB04OX "QR factorization update for rank-one modification.\n" \
    "\n" \
    "Performs QR factorization: (U; x') = Q*(R; 0) where U,R are upper triangular.\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): Upper triangular matrix U (n x n, F-order), overwritten by R\n" \
    "  x (ndarray): Vector x (n,), overwritten\n" \
    "\n" \
    "Returns:\n" \
    "  (r, x_out): Updated upper triangular matrix R, modified x"

#define DOC_MB04OY "Apply Householder reflector to matrix [A; B].\n" \
    "\n" \
    "Applies H = I - tau*[1;v]*[1;v]' to (m+1)-by-n matrix [A; B],\n" \
    "where A has one row.\n" \
    "\n" \
    "Parameters:\n" \
    "  m (int): Number of rows of B\n" \
    "  n (int): Number of columns\n" \
    "  v (ndarray): Householder vector (m, F-order)\n" \
    "  tau (float): Householder scalar\n" \
    "  a (ndarray): Matrix A (1 x n, F-order, modified in place)\n" \
    "  b (ndarray): Matrix B (m x n, F-order, modified in place)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b): Updated matrices"

#define DOC_MB04PA "Reduce (skew-)Hamiltonian like matrix with orthogonal symplectic transformation.\n" \
    "\n" \
    "Reduces a Hamiltonian H = [A G; Q -A'] or skew-Hamiltonian W = [A G; Q A']\n" \
    "so that elements below (k+1)-th subdiagonal in first nb columns of A,\n" \
    "and offdiagonal elements in first nb columns/rows of Q are zero.\n" \
    "This is an auxiliary routine called by MB04PB.\n" \
    "\n" \
    "Parameters:\n" \
    "  lham (bool): True for Hamiltonian, False for skew-Hamiltonian\n" \
    "  n (int): Number of columns of A, n >= 0\n" \
    "  k (int): Offset of reduction, k >= 0\n" \
    "  nb (int): Number of columns/rows to reduce, n > nb >= 0\n" \
    "  a (ndarray): (k+n)-by-n matrix A (F-order), modified on exit\n" \
    "  qg (ndarray): (n+k)-by-(n+1) matrix containing Q and G (F-order)\n" \
    "  xa (ndarray): n-by-(2*nb) output matrix XA (F-order)\n" \
    "  xg (ndarray): (k+n)-by-(2*nb) output matrix XG (F-order)\n" \
    "  xq (ndarray): n-by-(2*nb) output matrix XQ (F-order)\n" \
    "  ya (ndarray): (k+n)-by-(2*nb) output matrix YA (F-order)\n" \
    "  cs (ndarray): Output cosines/sines array (2*nb,)\n" \
    "  tau (ndarray): Output scalar factors array (nb,)\n" \
    "  dwork (ndarray): Workspace (3*nb,)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, qg, xa, xg, xq, ya, cs, tau): Modified arrays"

#define DOC_MB04PB "Reduce Hamiltonian matrix to PVL form (blocked version).\n" \
    "\n" \
    "Blocked algorithm version of mb04pu. Reduces a Hamiltonian matrix\n" \
    "H = [A G; Q -A^T] where A is N-by-N and G, Q are N-by-N symmetric,\n" \
    "to PVL form where Aout is upper Hessenberg and Qout is diagonal.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Order of matrix A (n >= 0)\n" \
    "  ilo (int): Starting index (1 <= ilo <= max(1,n) for n>0, ilo=1 for n=0)\n" \
    "  a (ndarray): N-by-N matrix A (F-order), modified to Aout\n" \
    "  qg (ndarray): N-by-(N+1) matrix with Q (lower) and G (upper) (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, qg, cs, tau, info):\n" \
    "    a: Upper Hessenberg Aout with reflector info\n" \
    "    qg: Diagonal Qout, upper tri Gout, reflector info\n" \
    "    cs: Cosines and sines of Givens rotations (2*n-2,)\n" \
    "    tau: Scalar factors of elementary reflectors (n-1,)\n" \
    "    info: 0=success, <0=param -info invalid"

#define DOC_MB04PU "Reduce Hamiltonian matrix to Paige/Van Loan (PVL) form.\n" \
    "\n" \
    "Reduces a Hamiltonian matrix H = [A G; Q -A^T] where A is N-by-N\n" \
    "and G, Q are N-by-N symmetric, to PVL form where Aout is upper\n" \
    "Hessenberg and Qout is diagonal. Uses orthogonal symplectic U.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Order of matrix A (n >= 0)\n" \
    "  ilo (int): Starting index (1 <= ilo <= max(1,n) for n>0, ilo=1 for n=0)\n" \
    "  a (ndarray): N-by-N matrix A (F-order), modified to Aout\n" \
    "  qg (ndarray): N-by-(N+1) matrix with Q (lower) and G (upper) (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, qg, cs, tau, info):\n" \
    "    a: Upper Hessenberg Aout with reflector info\n" \
    "    qg: Diagonal Qout, upper tri Gout, reflector info\n" \
    "    cs: Cosines and sines of Givens rotations (2*n-2,)\n" \
    "    tau: Scalar factors of elementary reflectors (n-1,)\n" \
    "    info: 0=success, <0=param -info invalid"

#define DOC_MB04PY "Apply elementary reflector to matrix from left or right.\n" \
    "\n" \
    "Applies H = I - tau*[1;v]*[1;v]' to m-by-n matrix C.\n" \
    "H*C if side='L', C*H if side='R'.\n" \
    "\n" \
    "Parameters:\n" \
    "  side (str): 'L' for left, 'R' for right\n" \
    "  m (int): Number of rows of C\n" \
    "  n (int): Number of columns of C\n" \
    "  v (ndarray): Householder vector (m-1 if 'L', n-1 if 'R')\n" \
    "  tau (float): Householder scalar\n" \
    "  c (ndarray): Matrix C (m x n, F-order, modified in place)\n" \
    "\n" \
    "Returns:\n" \
    "  c: Updated matrix"

#define DOC_MB04QB "Apply symplectic reflectors and Givens rotations (blocked).\n" \
    "\n" \
    "Overwrites C and D with Q*[op(C);op(D)] or Q^T*[op(C);op(D)] where\n" \
    "Q is a product of symplectic reflectors and Givens rotations.\n" \
    "Blocked version of mb04qu.\n" \
    "\n" \
    "Parameters:\n" \
    "  tranc (str): 'N' for op(C)=C, 'T'/'C' for op(C)=C^T\n" \
    "  trand (str): 'N' for op(D)=D, 'T'/'C' for op(D)=D^T\n" \
    "  tranq (str): 'N' for Q, 'T' for Q^T\n" \
    "  storev (str): 'C' for columnwise V, 'R' for rowwise\n" \
    "  storew (str): 'C' for columnwise W, 'R' for rowwise\n" \
    "  m (int): Number of rows of op(C) and op(D)\n" \
    "  n (int): Number of columns of op(C) and op(D)\n" \
    "  k (int): Number of elementary reflectors\n" \
    "  v (ndarray): Reflector vectors F(i)\n" \
    "  w (ndarray): Reflector vectors H(i)\n" \
    "  c (ndarray): Matrix C (F-order, modified)\n" \
    "  d (ndarray): Matrix D (F-order, modified)\n" \
    "  cs (ndarray): Cosines/sines of Givens rotations (2*k)\n" \
    "  tau (ndarray): Scalar factors of reflectors F(i) (k)\n" \
    "  ldwork (int, optional): Workspace size, -1 for query\n" \
    "\n" \
    "Returns:\n" \
    "  (c, d, info): Modified matrices and exit code"

#define DOC_MB04QC "Apply symplectic block reflector to matrices.\n" \
    "\n" \
    "Applies Q or Q^T to [op(A); op(B)] from the left using block factors from MB04QF.\n" \
    "\n" \
    "Parameters:\n" \
    "  strab (str): 'Z' for zero structure, 'N' for no structure\n" \
    "  trana (str): 'N' for A, 'T'/'C' for A^T\n" \
    "  tranb (str): 'N' for B, 'T'/'C' for B^T\n" \
    "  tranq (str): 'N' for Q, 'T' for Q^T\n" \
    "  direct (str): Reserved (use 'F')\n" \
    "  storev (str): 'C' for columnwise V, 'R' for rowwise\n" \
    "  storew (str): 'C' for columnwise W, 'R' for rowwise\n" \
    "  m, n, k (int): Dimensions (m rows, n cols, k reflectors)\n" \
    "  v, w (ndarray): Reflector vector matrices\n" \
    "  rs, t (ndarray): Block factors from MB04QF\n" \
    "  a, b (ndarray): Input matrices (modified)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b): Modified matrices"

#define DOC_MB04QF "Form triangular block factors of symplectic block reflector.\n" \
    "\n" \
    "Forms R, S, T factors for SH = diag(F,F)*prod(H_i)*prod(G_i) representation.\n" \
    "\n" \
    "Parameters:\n" \
    "  direct (str): 'F' for forward, 'B' for backward\n" \
    "  storev (str): 'C' for columnwise V, 'R' for rowwise\n" \
    "  storew (str): 'C' for columnwise W, 'R' for rowwise\n" \
    "  n (int): Order of reflectors\n" \
    "  k (int): Number of reflectors\n" \
    "  v (ndarray): V matrix storing F reflector vectors\n" \
    "  w (ndarray): W matrix storing H reflector vectors\n" \
    "  cs (ndarray): Givens rotation cosines/sines\n" \
    "  tau (ndarray): F reflector scalars\n" \
    "\n" \
    "Returns:\n" \
    "  (rs, t, info): Block factor matrices RS (k,6k), T (k,9k), exit code"

#define DOC_MB04QS "Multiply by product of symplectic reflectors and Givens rotations.\n" \
    "\n" \
    "Overwrites C and D with U*[op(C);op(D)] or U^T*[op(C);op(D)] where\n" \
    "U is defined as a product of symplectic reflectors and Givens\n" \
    "rotations as returned by MB04PU or MB04RU.\n" \
    "\n" \
    "Parameters:\n" \
    "  tranc (str): 'N' for op(C)=C, 'T'/'C' for op(C)=C^T\n" \
    "  trand (str): 'N' for op(D)=D, 'T'/'C' for op(D)=D^T\n" \
    "  tranu (str): 'N' for U, 'T' for U^T\n" \
    "  m (int): Number of rows of op(C) and op(D)\n" \
    "  n (int): Number of columns of op(C) and op(D)\n" \
    "  ilo (int): Index from previous MB04PU/MB04RU call (1 <= ilo <= m+1)\n" \
    "  v (ndarray): M-by-M matrix with reflector vectors H(i)\n" \
    "  w (ndarray): M-by-M matrix with reflector vectors F(i)\n" \
    "  c (ndarray): Matrix C (F-order, modified)\n" \
    "  d (ndarray): Matrix D (F-order, modified)\n" \
    "  cs (ndarray): Cosines/sines of Givens rotations (2*(m-1))\n" \
    "  tau (ndarray): Scalar factors of reflectors F(i) (m-1)\n" \
    "  ldwork (int, optional): Workspace size, -1 for query\n" \
    "\n" \
    "Returns:\n" \
    "  (c, d, info): Modified matrices and exit code"

#define DOC_MB04QU "Apply symplectic reflectors and Givens rotations (unblocked).\n" \
    "\n" \
    "Overwrites C and D with Q*[op(C);op(D)] or Q^T*[op(C);op(D)] where\n" \
    "Q is a product of symplectic reflectors and Givens rotations.\n" \
    "\n" \
    "Parameters:\n" \
    "  tranc (str): 'N' for op(C)=C, 'T'/'C' for op(C)=C^T\n" \
    "  trand (str): 'N' for op(D)=D, 'T'/'C' for op(D)=D^T\n" \
    "  tranq (str): 'N' for Q, 'T' for Q^T\n" \
    "  storev (str): 'C' for columnwise V, 'R' for rowwise\n" \
    "  storew (str): 'C' for columnwise W, 'R' for rowwise\n" \
    "  c (ndarray): Matrix C (F-order, modified)\n" \
    "  d (ndarray): Matrix D (F-order, modified)\n" \
    "  v (ndarray): Reflector vectors F(i)\n" \
    "  w (ndarray): Reflector vectors H(i)\n" \
    "  cs (ndarray): Cosines/sines of Givens rotations (2*k)\n" \
    "  tau (ndarray): Scalar factors of reflectors F(i) (k)\n" \
    "  k (int, optional): Number of reflectors (default: infer from V)\n" \
    "\n" \
    "Returns:\n" \
    "  (c, d, info): Modified matrices and exit code"

#define DOC_MB04RB "Reduce skew-Hamiltonian matrix to PVL form (blocked).\n" \
    "\n" \
    "Reduces a skew-Hamiltonian matrix W = [[A, G], [Q, A^T]] to\n" \
    "Paige/Van Loan (PVL) form using orthogonal symplectic U:\n" \
    "\n" \
    "  U^T W U = [[Aout, Gout], [0, Aout^T]]\n" \
    "\n" \
    "where Aout is upper Hessenberg, G and Q are skew-symmetric.\n" \
    "This is the blocked version of MB04RU.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Order of matrix A. n >= 0.\n" \
    "  ilo (int): Lower index of non-triangular block. 1 <= ilo <= n+1.\n" \
    "  a (ndarray): N-by-N matrix A (F-order, modified in place).\n" \
    "  qg (ndarray): N-by-(N+1) array. Cols 1:N contain strictly lower\n" \
    "               triangular Q, cols 2:N+1 contain strictly upper\n" \
    "               triangular G. Modified in place.\n" \
    "\n" \
    "Returns:\n" \
    "  (a, qg, cs, tau, info):\n" \
    "    a: Upper Hessenberg Aout with reflector info below subdiagonal\n" \
    "    qg: Reflector info in cols 1:N-1, strictly upper Gout in cols 2:N+1\n" \
    "    cs: Cosines/sines of Givens rotations G(i), length 2*(N-1)\n" \
    "    tau: Scalar factors of reflectors F(i), length N-1\n" \
    "    info: Exit code (0=success, <0=invalid parameter -info)"

#define DOC_MB04RD "Block-diagonalize generalized real Schur form using well-conditioned transformations.\n" \
    "\n" \
    "Reduces a matrix pair (A,B) in generalized real Schur form to block-diagonal\n" \
    "form using non-orthogonal equivalence transformations bounded by PMAX.\n" \
    "Optionally reorders blocks to cluster eigenvalues.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobx (str): 'N' = don't accumulate left transforms, 'U' = accumulate in X\n" \
    "  joby (str): 'N' = don't accumulate right transforms, 'U' = accumulate in Y\n" \
    "  sort (str): 'N' = no reordering, 'S' = cluster eigenvalues,\n" \
    "              'C' = closest-neighbour, 'B' = both strategies\n" \
    "  n (int): Order of matrices A, B, X, Y (n >= 0)\n" \
    "  pmax (float): Bound for transformation element magnitudes (pmax >= 1)\n" \
    "  a (ndarray): N-by-N upper quasi-triangular matrix A (F-order, modified)\n" \
    "  b (ndarray): N-by-N upper triangular matrix B (F-order, modified)\n" \
    "  x (ndarray): N-by-N left transformation matrix (F-order, modified if jobx='U')\n" \
    "  y (ndarray): N-by-N right transformation matrix (F-order, modified if joby='U')\n" \
    "  tol (float): Clustering tolerance (if sort='S' or 'B'); 0 for default\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, x, y, nblcks, blsize, alphar, alphai, beta, info):\n" \
    "    Block-diagonalized matrices, number of blocks, block sizes,\n" \
    "    eigenvalue components (alphar+i*alphai)/beta, exit code"

#define DOC_MB04RS "Solve generalized real Sylvester equation with Schur form matrices.\n" \
    "\n" \
    "Solves the generalized real Sylvester equation:\n" \
    "  A * R - L * B = scale * C,\n" \
    "  D * R - L * E = scale * F,\n" \
    "\n" \
    "where R and L are unknown M-by-N matrices, and (A,D), (B,E) and\n" \
    "(C,F) are given matrix pairs. (A,D) and (B,E) must be in generalized\n" \
    "Schur canonical form (A,B upper quasi-triangular; D,E upper triangular).\n" \
    "\n" \
    "The solution (R,L) overwrites (C,F). 0 <= scale <= 1 is an output\n" \
    "scaling factor chosen to avoid overflow.\n" \
    "\n" \
    "Parameters:\n" \
    "  m (int): Order of A and D, row dimension of C, F, R, L. m >= 0.\n" \
    "  n (int): Order of B and E, column dimension of C, F, R, L. n >= 0.\n" \
    "  pmax (float): Upper bound for absolute value of solution elements. pmax >= 1.\n" \
    "  a (ndarray): M-by-M upper quasi-triangular matrix (Schur form, F-order)\n" \
    "  b (ndarray): N-by-N upper quasi-triangular matrix (Schur form, F-order)\n" \
    "  c (ndarray): M-by-N matrix, RHS of first equation (F-order, modified)\n" \
    "  d (ndarray): M-by-M upper triangular matrix (Schur form, F-order)\n" \
    "  e (ndarray): N-by-N upper triangular matrix (Schur form, F-order)\n" \
    "  f (ndarray): M-by-N matrix, RHS of second equation (F-order, modified)\n" \
    "\n" \
    "Returns:\n" \
    "  (r, l, scale, info): Solution R (in c), solution L (in f),\n" \
    "    scaling factor, exit code (0=success, 1=element > pmax, 2=singular)"

#define DOC_MB04RT "Blocked solver for generalized real Sylvester equation (Level 3 BLAS).\n" \
    "\n" \
    "Solves the generalized real Sylvester equation:\n" \
    "  A * R - L * B = scale * C,\n" \
    "  D * R - L * E = scale * F,\n" \
    "\n" \
    "using Level 3 BLAS for improved performance on larger matrices.\n" \
    "This is the blocked version of mb04rs.\n" \
    "\n" \
    "Parameters:\n" \
    "  m (int): Order of A and D, row dimension of C, F, R, L. m >= 0.\n" \
    "  n (int): Order of B and E, column dimension of C, F, R, L. n >= 0.\n" \
    "  pmax (float): Upper bound for absolute value of solution elements. pmax >= 1.\n" \
    "  a (ndarray): M-by-M upper quasi-triangular matrix (Schur form, F-order)\n" \
    "  b (ndarray): N-by-N upper quasi-triangular matrix (Schur form, F-order)\n" \
    "  c (ndarray): M-by-N matrix, RHS of first equation (F-order, modified)\n" \
    "  d (ndarray): M-by-M upper triangular matrix (Schur form, F-order)\n" \
    "  e (ndarray): N-by-N upper triangular matrix (Schur form, F-order)\n" \
    "  f (ndarray): M-by-N matrix, RHS of second equation (F-order, modified)\n" \
    "\n" \
    "Returns:\n" \
    "  (r, l, scale, info): Solution R (in c), solution L (in f),\n" \
    "    scaling factor, exit code (0=success, 1=element > pmax, 2=singular)"

#define DOC_MB04RU "Reduce skew-Hamiltonian matrix to PVL form (unblocked).\n" \
    "\n" \
    "Reduces a skew-Hamiltonian matrix W = [[A, G], [Q, A^T]] to\n" \
    "Paige/Van Loan (PVL) form using orthogonal symplectic U:\n" \
    "\n" \
    "  U^T W U = [[Aout, Gout], [0, Aout^T]]\n" \
    "\n" \
    "where Aout is upper Hessenberg, G and Q are skew-symmetric.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Order of matrix A. n >= 0.\n" \
    "  ilo (int): Lower index of non-triangular block. 1 <= ilo <= n+1.\n" \
    "  a (ndarray): N-by-N matrix A (F-order, modified in place).\n" \
    "  qg (ndarray): N-by-(N+1) array. Cols 1:N contain strictly lower\n" \
    "               triangular Q, cols 2:N+1 contain strictly upper\n" \
    "               triangular G. Modified in place.\n" \
    "\n" \
    "Returns:\n" \
    "  (a, qg, cs, tau, info):\n" \
    "    a: Upper Hessenberg Aout with reflector info below subdiagonal\n" \
    "    qg: Reflector info in cols 1:N-1, strictly upper Gout in cols 2:N+1\n" \
    "    cs: Cosines/sines of Givens rotations G(i), length 2*(N-1)\n" \
    "    tau: Scalar factors of reflectors F(i), length N-1\n" \
    "    info: Exit code (0=success, <0=invalid parameter -info)"

#define DOC_MB04RV "Solve the generalized complex Sylvester equation.\n" \
    "\n" \
    "Solves the generalized complex Sylvester equation:\n" \
    "  A * R - L * B = scale * C,\n" \
    "  D * R - L * E = scale * F,\n" \
    "\n" \
    "where R and L are unknown M-by-N complex matrices, and (A,D), (B,E)\n" \
    "and (C,F) are given complex matrix pairs. A, B, D, and E are upper\n" \
    "triangular (i.e., (A,D) and (B,E) are in generalized complex Schur form).\n" \
    "\n" \
    "The solution (R,L) overwrites (C,F). 0 <= scale <= 1 is an output\n" \
    "scaling factor chosen to avoid overflow.\n" \
    "\n" \
    "For efficiency, the 'absolute value' of a complex number x is computed\n" \
    "as |real(x)| + |imag(x)| (taxi-cab norm).\n" \
    "\n" \
    "Parameters:\n" \
    "  m (int): Order of A and D, row dimension of C, F, R, L. m >= 0.\n" \
    "  n (int): Order of B and E, column dimension of C, F, R, L. n >= 0.\n" \
    "  pmax (float): Upper bound for absolute value of solution elements. pmax >= 1.\n" \
    "  a (ndarray): M-by-M upper triangular complex matrix (Schur form, F-order)\n" \
    "  b (ndarray): N-by-N upper triangular complex matrix (Schur form, F-order)\n" \
    "  c (ndarray): M-by-N complex matrix, RHS of first equation (F-order, modified)\n" \
    "  d (ndarray): M-by-M upper triangular complex matrix (Schur form, F-order)\n" \
    "  e (ndarray): N-by-N upper triangular complex matrix (Schur form, F-order)\n" \
    "  f (ndarray): M-by-N complex matrix, RHS of second equation (F-order, modified)\n" \
    "\n" \
    "Returns:\n" \
    "  (r, l, scale, info): Solution R (in c), solution L (in f),\n" \
    "    scaling factor, exit code (0=success, 1=element > pmax, 2=singular/close eigenvalues)"

#define DOC_MB04RW "Blocked solver for generalized complex Sylvester equation (Level 3 BLAS).\n" \
    "\n" \
    "Solves the generalized complex Sylvester equation:\n" \
    "  A * R - L * B = scale * C,\n" \
    "  D * R - L * E = scale * F,\n" \
    "\n" \
    "using Level 3 BLAS for improved performance on larger matrices.\n" \
    "This is the blocked version of mb04rv.\n" \
    "\n" \
    "Where R and L are unknown M-by-N complex matrices, and (A,D), (B,E)\n" \
    "and (C,F) are given complex matrix pairs. A, B, D, and E are upper\n" \
    "triangular (i.e., (A,D) and (B,E) are in generalized complex Schur form).\n" \
    "\n" \
    "The solution (R,L) overwrites (C,F). 0 <= scale <= 1 is an output\n" \
    "scaling factor chosen to avoid overflow.\n" \
    "\n" \
    "For efficiency, the 'absolute value' of a complex number x is computed\n" \
    "as |real(x)| + |imag(x)| (taxi-cab norm).\n" \
    "\n" \
    "Parameters:\n" \
    "  m (int): Order of A and D, row dimension of C, F, R, L. m >= 0.\n" \
    "  n (int): Order of B and E, column dimension of C, F, R, L. n >= 0.\n" \
    "  pmax (float): Upper bound for absolute value of solution elements. pmax >= 1.\n" \
    "  a (ndarray): M-by-M upper triangular complex matrix (Schur form, F-order)\n" \
    "  b (ndarray): N-by-N upper triangular complex matrix (Schur form, F-order)\n" \
    "  c (ndarray): M-by-N complex matrix, RHS of first equation (F-order, modified)\n" \
    "  d (ndarray): M-by-M upper triangular complex matrix (Schur form, F-order)\n" \
    "  e (ndarray): N-by-N upper triangular complex matrix (Schur form, F-order)\n" \
    "  f (ndarray): M-by-N complex matrix, RHS of second equation (F-order, modified)\n" \
    "\n" \
    "Returns:\n" \
    "  (r, l, scale, info): Solution R (in c), solution L (in f),\n" \
    "    scaling factor, exit code (0=success, 1=element > pmax, 2=singular/close eigenvalues)"

#define DOC_MB04RZ "Block-diagonalize generalized complex Schur form using well-conditioned transformations.\n" \
    "\n" \
    "Reduces a complex matrix pair (A,B) in generalized complex Schur form to\n" \
    "block-diagonal form using non-unitary equivalence transformations bounded by PMAX.\n" \
    "Optionally reorders blocks to cluster eigenvalues.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobx (str): 'N' = don't accumulate left transforms, 'U' = accumulate in X\n" \
    "  joby (str): 'N' = don't accumulate right transforms, 'U' = accumulate in Y\n" \
    "  sort (str): 'N' = no reordering, 'S' = cluster eigenvalues,\n" \
    "              'C' = closest-neighbour, 'B' = both strategies\n" \
    "  n (int): Order of matrices A, B, X, Y (n >= 0)\n" \
    "  pmax (float): Bound for transformation element magnitudes (pmax >= 1)\n" \
    "  a (ndarray): N-by-N complex upper triangular matrix A (F-order, modified)\n" \
    "  b (ndarray): N-by-N complex upper triangular matrix B (F-order, modified)\n" \
    "  x (ndarray): N-by-N complex left transformation matrix (F-order, modified if jobx='U')\n" \
    "  y (ndarray): N-by-N complex right transformation matrix (F-order, modified if joby='U')\n" \
    "  tol (float): Clustering tolerance (if sort='S' or 'B'); 0 for default\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, x, y, nblcks, blsize, alpha, beta, info):\n" \
    "    Block-diagonalized matrices, number of blocks, block sizes,\n" \
    "    complex eigenvalue components alpha/beta, exit code"

#define DOC_MB04SU "Compute symplectic QR decomposition of a real 2M-by-N matrix [A; B].\n" \
    "\n" \
    "Computes [A; B] = Q * R where Q is symplectic orthogonal, R11 is upper\n" \
    "triangular, and R21 is strictly upper triangular. Unblocked version.\n" \
    "\n" \
    "Q = diag(H(1),H(1)) G(1) diag(F(1),F(1)) ... diag(H(k),H(k)) G(k) diag(F(k),F(k))\n" \
    "where k = min(m,n), H(i) and F(i) are Householder reflectors, and G(i) are\n" \
    "Givens rotations.\n" \
    "\n" \
    "Parameters:\n" \
    "  m (int): Number of rows in A and B (m >= 0)\n" \
    "  n (int): Number of columns in A and B (n >= 0)\n" \
    "  a (ndarray): M-by-N matrix A (F-order, modified in place)\n" \
    "  b (ndarray): M-by-N matrix B (F-order, modified in place)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, cs, tau, info):\n" \
    "    a: [R11 R12] with reflector info in zero parts\n" \
    "    b: [R21 R22] with reflector info in zero parts\n" \
    "    cs: Cosines/sines of Givens rotations G(i), length 2*min(m,n)\n" \
    "    tau: Scalar factors of reflectors F(i), length min(m,n)\n" \
    "    info: Exit code (0=success, <0=invalid parameter -info)"

#define DOC_MB04TB "Symplectic URV decomposition (blocked).\n" \
    "\n" \
    "Computes H = U * R * V^T where H = [[op(A), G], [Q, op(B)]] is\n" \
    "a 2N-by-2N Hamiltonian matrix, U, V are orthogonal symplectic,\n" \
    "and R = [[R11, R12], [0, R22]] with R11 upper triangular,\n" \
    "R22 lower Hessenberg. Blocked version of mb04ts.\n" \
    "\n" \
    "Parameters:\n" \
    "  trana (str): 'N' for op(A)=A, 'T'/'C' for op(A)=A^T\n" \
    "  tranb (str): 'N' for op(B)=B, 'T'/'C' for op(B)=B^T\n" \
    "  n (int): Order of matrices (n >= 0)\n" \
    "  ilo (int): Starting index for reduction (1 <= ilo <= n+1)\n" \
    "  a (ndarray): N-by-N matrix A (F-order, modified)\n" \
    "  b (ndarray): N-by-N matrix B (F-order, modified)\n" \
    "  g (ndarray): N-by-N matrix G (F-order, modified)\n" \
    "  q (ndarray): N-by-N matrix Q (F-order, modified)\n" \
    "  ldwork (int, optional): Workspace size, -1 for query\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, g, q, csl, csr, taul, taur, info):\n" \
    "    Modified matrices, Givens rotations, reflector factors, exit code"

#define DOC_MB04TS "Symplectic URV decomposition (unblocked).\n" \
    "\n" \
    "Computes H = U * R * V^T where H = [[op(A), G], [Q, op(B)]] is\n" \
    "a 2N-by-2N Hamiltonian matrix, U, V are orthogonal symplectic,\n" \
    "and R = [[R11, R12], [0, R22]] with R11 upper triangular,\n" \
    "R22 lower Hessenberg.\n" \
    "\n" \
    "Parameters:\n" \
    "  trana (str): 'N' for op(A)=A, 'T'/'C' for op(A)=A^T\n" \
    "  tranb (str): 'N' for op(B)=B, 'T'/'C' for op(B)=B^T\n" \
    "  n (int): Order of matrices (n >= 0)\n" \
    "  ilo (int): Starting index for reduction (1 <= ilo <= n)\n" \
    "  a (ndarray): N-by-N matrix A (F-order, modified)\n" \
    "  b (ndarray): N-by-N matrix B (F-order, modified)\n" \
    "  g (ndarray): N-by-N matrix G (F-order, modified)\n" \
    "  q (ndarray): N-by-N matrix Q (F-order, modified)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, g, q, csl, csr, taul, taur, info):\n" \
    "    Modified matrices, Givens rotations, reflector factors, exit code"

#define DOC_MB04TT "Row compression with column echelon form preservation.\n" \
    "\n" \
    "Transforms submatrices (AA, EE) of A and E such that Aj is row compressed\n" \
    "while keeping Ej in column echelon form. This is step j of Algorithm 3.2.1\n" \
    "from Beelen's thesis for computing Kronecker structure of matrix pencils.\n" \
    "\n" \
    "Let AA = A(IFIRA:M, IFICA:N), EE = E(IFIRA:M, IFICA:N)\n" \
    "Let Aj = A(IFIRA:M, IFICA:IFICA+NCA-1), Ej = E(IFIRA:M, IFICA+NCA:N)\n" \
    "\n" \
    "Parameters:\n" \
    "  updatq (bool): True to accumulate row transforms in Q\n" \
    "  updatz (bool): True to accumulate column transforms in Z\n" \
    "  m (int): Number of rows in A, E, Q (m >= 0)\n" \
    "  n (int): Number of columns in A, E, Z (n >= 0)\n" \
    "  ifira (int): First row index (1-based)\n" \
    "  ifica (int): First column index (1-based)\n" \
    "  nca (int): Number of columns in Aj\n" \
    "  a (ndarray): M-by-N matrix A (F-order, modified)\n" \
    "  e (ndarray): M-by-N matrix E in column echelon form (F-order, modified)\n" \
    "  q (ndarray): M-by-M orthogonal matrix (F-order, modified if updatq)\n" \
    "  z (ndarray): N-by-N orthogonal matrix (F-order, modified if updatz)\n" \
    "  istair (ndarray): Column echelon encoding (+j=corner, -j=boundary)\n" \
    "  tol (float): Tolerance for zero elements\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, q, z, istair, rank, info):\n" \
    "    Transformed matrices, updated staircase array, rank of Aj, exit code"

#define DOC_MB04TU "Givens transformation with interchange (modified DROT).\n" \
    "\n" \
    "Applies row-permuted Givens transformation:\n" \
    "  X_new = C*Y - S*X\n" \
    "  Y_new = C*X + S*Y\n" \
    "\n" \
    "NOTE: This is NOT standard DROT (which computes X' = CX + SY, Y' = -SX + CY).\n" \
    "\n" \
    "Parameters:\n" \
    "  x (ndarray): Input array X (modified in place)\n" \
    "  y (ndarray): Input array Y (modified in place)\n" \
    "  c (float): Cosine of Givens rotation\n" \
    "  s (float): Sine of Givens rotation\n" \
    "  n (int, optional): Number of elements to transform (default: min(len(x), len(y)))\n" \
    "  incx (int, optional): Increment for X (default: 1)\n" \
    "  incy (int, optional): Increment for Y (default: 1)\n" \
    "\n" \
    "Returns:\n" \
    "  (x, y): Transformed arrays"

#define DOC_MB04TV "Reduce submatrix A(k) to upper triangular form using column Givens rotations.\n" \
    "\n" \
    "Reduces A(k) = A(IFIRA:ma, IFICA:na) where ma = IFIRA-1+NRA, na = IFICA-1+NCA\n" \
    "to upper triangular form. Matrix A(k) must have full row rank.\n" \
    "\n" \
    "The same column transformations are applied to E(k) = E(1:IFIRA-1, IFICA:na).\n" \
    "Note: E uses the same column indices but different row indices than A.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Number of columns of A and E, n >= 0\n" \
    "  nra (int): Number of rows in A to transform, 0 <= nra <= lda\n" \
    "  nca (int): Number of columns in A to transform, 0 <= nca <= n\n" \
    "  ifira (int): First row index in A (1-based)\n" \
    "  ifica (int): First column index in A (1-based)\n" \
    "  a (ndarray): Matrix A (F-order, modified in place)\n" \
    "  e (ndarray): Matrix E (F-order, modified in place)\n" \
    "  z (ndarray): Transform matrix Z (F-order, modified if updatz=True)\n" \
    "  updatz (bool, optional): Whether to accumulate transforms in Z (default: False)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, z, info): Transformed matrices and exit code (0=success)"

#define DOC_MB04TW "Reduce submatrix E(k) to upper triangular form using row Givens rotations.\n" \
    "\n" \
    "Reduces E(k) = E(IFIRE:me, IFICE:ne) where me = IFIRE-1+NRE, ne = IFICE-1+NCE\n" \
    "to upper triangular form. Matrix E(k) must have full column rank.\n" \
    "\n" \
    "The same row transformations are applied to A(k) = A(IFIRE:me, IFICA:N).\n" \
    "Note: A uses the same row indices but different column indices than E.\n" \
    "\n" \
    "Parameters:\n" \
    "  m (int): Number of rows of A and E, m >= 0\n" \
    "  n (int): Number of columns of A and E, n >= 0\n" \
    "  nre (int): Number of rows in E to transform, 0 <= nre <= m\n" \
    "  nce (int): Number of columns in E to transform, 0 <= nce <= n\n" \
    "  ifire (int): First row index in E (1-based)\n" \
    "  ifice (int): First column index in E (1-based)\n" \
    "  ifica (int): First column index in A (1-based)\n" \
    "  a (ndarray): Matrix A (F-order, modified in place)\n" \
    "  e (ndarray): Matrix E (F-order, modified in place)\n" \
    "  q (ndarray): Transform matrix Q (F-order, modified if updatq=True)\n" \
    "  updatq (bool, optional): Whether to accumulate transforms in Q (default: False)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, q, info): Transformed matrices and exit code (0=success)"

#define DOC_MB04TX "Separate pencils s*E(eps)-A(eps) and s*E(inf)-A(inf).\n" \
    "\n" \
    "Separates the epsilon and infinite parts of the pencil s*E(eps,inf)-A(eps,inf)\n" \
    "in staircase form using Algorithm 3.3.3 from Beelen's thesis.\n" \
    "\n" \
    "On entry, matrices A and E are assumed to be in staircase form. On exit,\n" \
    "the pencil is separated into:\n" \
    "  - s*E(eps)-A(eps): contains all Kronecker column indices\n" \
    "  - s*E(inf)-A(inf): contains all infinite elementary divisors\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): M-by-N matrix A (F-order, modified)\n" \
    "  e (ndarray): M-by-N matrix E (F-order, modified)\n" \
    "  inuk (ndarray): Row dimensions nu(k) of full-rank submatrices (modified)\n" \
    "  imuk (ndarray): Column dimensions mu(k) of full-rank submatrices (modified)\n" \
    "  q (ndarray, optional): M-by-M matrix for row transforms (if updatq=True)\n" \
    "  z (ndarray, optional): N-by-N matrix for column transforms (if updatz=True)\n" \
    "  m (int, optional): Number of rows (default: from array shape)\n" \
    "  n (int, optional): Number of columns (default: from array shape)\n" \
    "  updatq (bool, optional): Accumulate row transforms in Q (default: True)\n" \
    "  updatz (bool, optional): Accumulate column transforms in Z (default: True)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, q, z, nblcks, inuk, imuk, mnei, info):\n" \
    "    Transformed matrices, updated block count, dimensions, pencil sizes, exit code.\n" \
    "    mnei = [meps, neps, minf, ninf] dimensions of epsilon and infinite pencils."

#define DOC_MB04TY "Triangularize full rank submatrices in staircase pencil.\n" \
    "\n" \
    "Performs triangularization of submatrices with full row and column rank\n" \
    "in the pencil s*E(eps,inf)-A(eps,inf) using Algorithm 3.3.1 from Beelen's thesis.\n" \
    "\n" \
    "On entry, A and E are assumed to be in generalized Schur form and\n" \
    "s*E(eps,inf)-A(eps,inf) is in staircase form.\n" \
    "\n" \
    "Parameters:\n" \
    "  m (int): Number of rows of A and E, m >= 0\n" \
    "  n (int): Number of columns of A and E, n >= 0\n" \
    "  nblcks (int): Number of submatrices with full row rank in A(eps,inf)\n" \
    "  inuk (ndarray): Row dimensions nu(k) of full-rank submatrices\n" \
    "  imuk (ndarray): Column dimensions mu(k) of full-rank submatrices\n" \
    "  a (ndarray): M-by-N matrix A (F-order, modified)\n" \
    "  e (ndarray): M-by-N matrix E (F-order, modified)\n" \
    "  q (ndarray): M-by-M transform matrix Q (F-order, modified if updatq=True)\n" \
    "  z (ndarray): N-by-N transform matrix Z (F-order, modified if updatz=True)\n" \
    "  updatq (bool, optional): Accumulate row transforms in Q (default: False)\n" \
    "  updatz (bool, optional): Accumulate column transforms in Z (default: False)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, q, z, info): Transformed matrices and exit code.\n" \
    "    info=0: success\n" \
    "    info=1: incorrect dimensions of full column rank submatrix (mu(k+1) > nu(k))\n" \
    "    info=2: incorrect dimensions of full row rank submatrix (nu(k) > mu(k))"

#define DOC_MB04UD "Reduce pencil sE-A to column echelon form with orthogonal transformations.\n" \
    "\n" \
    "Computes orthogonal Q and Z such that Q'(sE-A)Z has E in column echelon form.\n" \
    "Column echelon form: first (N-r) columns are zero, remaining columns have\n" \
    "last nonzero element with strictly increasing row indices.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobq (str): 'N' - don't form Q, 'I' - init to I, 'U' - update input Q\n" \
    "  jobz (str): 'N' - don't form Z, 'I' - init to I, 'U' - update input Z\n" \
    "  m (int): Number of rows in A and E (m >= 0)\n" \
    "  n (int): Number of columns in A and E (n >= 0)\n" \
    "  a (ndarray): M-by-N matrix A (F-order, modified in place)\n" \
    "  e (ndarray): M-by-N matrix E (F-order, modified in place)\n" \
    "  q (ndarray, optional): M-by-M matrix Q (for jobq='U')\n" \
    "  z (ndarray, optional): N-by-N matrix Z (for jobz='U')\n" \
    "  tol (float, optional): Zero tolerance (default: eps*||E||_max)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, q, z, ranke, istair, info):\n" \
    "    a: Transformed Q'*A*Z\n" \
    "    e: Transformed Q'*E*Z in column echelon form\n" \
    "    q: Orthogonal row transformation matrix (if jobq='I' or 'U')\n" \
    "    z: Orthogonal column transformation matrix (if jobz='I' or 'U')\n" \
    "    ranke: Computed rank of E\n" \
    "    istair: Staircase structure array (+j=corner, -j=boundary)\n" \
    "    info: Exit code (0=success, <0=invalid parameter -info)"

#define DOC_MB04VD "Upper block triangular form for rectangular pencil sE-A.\n" \
    "\n" \
    "Computes orthogonal transformations Q and Z such that Q'(sE-A)Z is in upper\n" \
    "block triangular form, where E is in column echelon form.\n" \
    "\n" \
    "MODE options:\n" \
    "  'B': Basic reduction to staircase form\n" \
    "  'T': Triangularization of full rank submatrices\n" \
    "  'S': Separation of epsilon and infinite parts\n" \
    "\n" \
    "Parameters:\n" \
    "  mode (str): 'B', 'T', or 'S' - specifies desired structure\n" \
    "  jobq (str): 'N', 'I', or 'U' - how to handle Q matrix\n" \
    "  jobz (str): 'N', 'I', or 'U' - how to handle Z matrix\n" \
    "  m (int): Number of rows in A, E (order of Q)\n" \
    "  n (int): Number of columns in A, E (order of Z)\n" \
    "  ranke (int): Rank of E in column echelon form\n" \
    "  a (ndarray): M-by-N matrix A (F-order)\n" \
    "  e (ndarray): M-by-N matrix E in column echelon form (F-order)\n" \
    "  q (ndarray): Transform matrix Q (F-order)\n" \
    "  z (ndarray): Transform matrix Z (F-order)\n" \
    "  istair (ndarray): Column echelon info from MB04UD\n" \
    "  tol (float, optional): Tolerance for zero elements (default: 0)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, q, z, nblcks, nblcki, imuk, inuk, imuk0, mnei, info):\n" \
    "    Transformed matrices, block counts, dimension arrays, exit code."

#define DOC_MB04VX "Separate pencils s*E(eps)-A(eps) and s*E(inf)-A(inf) (variant).\n" \
    "\n" \
    "Similar to MB04TX but with NBLCKS as input only and MNEI has 3 elements.\n" \
    "Separates the epsilon and infinite parts of the staircase pencil.\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): M-by-N matrix A (F-order, modified)\n" \
    "  e (ndarray): M-by-N matrix E (F-order, modified)\n" \
    "  q (ndarray): Transform matrix Q (F-order, modified if updatq=True)\n" \
    "  z (ndarray): Transform matrix Z (F-order, modified if updatz=True)\n" \
    "  nblcks (int): Number of full-row-rank submatrices\n" \
    "  inuk (ndarray): Row dimensions nu(k) (modified)\n" \
    "  imuk (ndarray): Column dimensions mu(k) (modified)\n" \
    "  updatq (bool, optional): Accumulate row transforms (default: False)\n" \
    "  updatz (bool, optional): Accumulate column transforms (default: False)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, q, z, inuk, imuk, mnei): Transformed matrices and pencil dimensions.\n" \
    "    mnei = [meps, neps, minf] dimensions of epsilon and infinite pencils."

#define DOC_MB04WD "Generate matrix Q with orthogonal columns from symplectic reflectors (blocked version).\n" \
    "\n" \
    "Blocked version of MB04WU. Generates a matrix Q with orthogonal columns\n" \
    "(spanning an isotropic subspace), which is defined as the first n columns\n" \
    "of a product of symplectic reflectors and Givens rotations.\n" \
    "\n" \
    "Q is returned as [op(Q1), op(Q2)] where the full symplectic form is:\n" \
    "  Q = [[op(Q1), op(Q2)], [-op(Q2), op(Q1)]]\n" \
    "\n" \
    "Parameters:\n" \
    "  tranq1 (bool): True for Q1 transposed, False for Q1 non-transposed\n" \
    "  tranq2 (bool): True for Q2 transposed, False for Q2 non-transposed\n" \
    "  m (int): Number of rows of Q1 and Q2 (m >= 0)\n" \
    "  n (int): Number of columns of Q1 and Q2 (0 <= n <= m)\n" \
    "  k (int): Number of symplectic Givens rotations (0 <= k <= n)\n" \
    "  q1 (ndarray): Input reflectors F(i), output matrix Q1 or Q1' (F-order)\n" \
    "  q2 (ndarray): Input reflectors H(i) with factors, output Q2 or Q2' (F-order)\n" \
    "  cs (ndarray): Cosines/sines of Givens rotations G(i), length 2*k\n" \
    "  tau (ndarray): Scalar factors of reflectors F(i), length k\n" \
    "\n" \
    "Returns:\n" \
    "  (q1, q2, info): Generated orthogonal matrices, exit code (0=success)"

#define DOC_MB04WP "Generate orthogonal symplectic matrix U from MB04PU output.\n" \
    "\n" \
    "Generates an orthogonal symplectic matrix U defined as a product of\n" \
    "symplectic reflectors and Givens rotations as returned by MB04PU.\n" \
    "The matrix U is returned in terms of its first N rows:\n" \
    "  U = [[U1, U2], [-U2, U1]]\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Order of matrices U1 and U2 (n >= 0)\n" \
    "  ilo (int): Same value as in previous MB04PU call. U is identity except\n" \
    "             in submatrix U([ilo+1:n n+ilo+1:2n], [ilo+1:n n+ilo+1:2n]).\n" \
    "             1 <= ilo <= max(1,n)\n" \
    "  u1 (ndarray): N-by-N matrix. Input: reflector vectors F(i) in columns.\n" \
    "                Output: U1 (F-order)\n" \
    "  u2 (ndarray): N-by-N matrix. Input: reflector vectors H(i) and scalars\n" \
    "                on subdiagonal. Output: U2 (F-order)\n" \
    "  cs (ndarray): Cosines/sines of Givens rotations G(i), length 2*n-2\n" \
    "  tau (ndarray): Scalar factors of reflectors F(i), length n-1\n" \
    "\n" \
    "Returns:\n" \
    "  (u1, u2, info): Generated orthogonal symplectic matrix blocks, exit code (0=success)"

#define DOC_MB04WR "Generate orthogonal symplectic matrices U or V from symplectic reflectors.\n" \
    "\n" \
    "Generates orthogonal symplectic matrices U or V, defined as products of\n" \
    "symplectic reflectors and Givens rotations, as returned by MB04TS or MB04TB.\n" \
    "\n" \
    "The matrices U and V are returned in terms of their first N/2 rows:\n" \
    "  U = [[U1, U2], [-U2, U1]]\n" \
    "  V = [[V1, V2], [-V2, V1]]\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'U' to generate U, 'V' to generate V\n" \
    "  trans (str): Must match TRANA (job='U') or TRANB (job='V') from MB04TS/MB04TB.\n" \
    "               'N' for no transpose, 'T'/'C' for transpose\n" \
    "  n (int): Order of matrices Q1 and Q2 (n >= 0)\n" \
    "  ilo (int): Same value as in previous MB04TS/MB04TB call.\n" \
    "             1 <= ilo <= max(1,n)\n" \
    "  q1 (ndarray): N-by-N matrix. Input: reflector vectors FU/FV(i).\n" \
    "                Output: U1/V1 or their transposes depending on job/trans (F-order)\n" \
    "  q2 (ndarray): N-by-N matrix. Input: reflector vectors HU/HV(i).\n" \
    "                Output: U2 or V2^T depending on job (F-order)\n" \
    "  cs (ndarray): Cosines/sines of Givens rotations, length 2n (job='U') or 2n-2 (job='V')\n" \
    "  tau (ndarray): Scalar factors of reflectors, length n (job='U') or n-1 (job='V')\n" \
    "\n" \
    "Returns:\n" \
    "  (q1, q2, info): Generated orthogonal symplectic matrix blocks, exit code (0=success)"

#define DOC_MB04WU "Generate matrix Q with orthogonal columns from symplectic reflectors and Givens rotations.\n" \
    "\n" \
    "Generates a matrix Q with orthogonal columns (spanning an isotropic subspace),\n" \
    "which is defined as the first n columns of a product of symplectic reflectors\n" \
    "and Givens rotations. This is the inverse operation of MB04SU.\n" \
    "\n" \
    "Q is returned as [op(Q1), op(Q2)] where the full symplectic form is:\n" \
    "  Q = [[op(Q1), op(Q2)], [-op(Q2), op(Q1)]]\n" \
    "\n" \
    "Parameters:\n" \
    "  tranq1 (bool): True for Q1 transposed, False for Q1 non-transposed\n" \
    "  tranq2 (bool): True for Q2 transposed, False for Q2 non-transposed\n" \
    "  m (int): Number of rows of Q1 and Q2 (m >= 0)\n" \
    "  n (int): Number of columns of Q1 and Q2 (0 <= n <= m)\n" \
    "  k (int): Number of symplectic Givens rotations (0 <= k <= n)\n" \
    "  q1 (ndarray): Input reflectors F(i), output matrix Q1 or Q1' (F-order)\n" \
    "  q2 (ndarray): Input reflectors H(i) with factors, output Q2 or Q2' (F-order)\n" \
    "  cs (ndarray): Cosines/sines of Givens rotations G(i), length 2*k\n" \
    "  tau (ndarray): Scalar factors of reflectors F(i), length k\n" \
    "\n" \
    "Returns:\n" \
    "  (q1, q2, info): Generated orthogonal matrices, exit code (0=success)"

#define DOC_MB04XD "Compute basis for left/right singular subspace for smallest singular values.\n" \
    "\n" \
    "Computes a basis for the left and/or right singular subspace of an M-by-N\n" \
    "matrix A corresponding to its smallest singular values using Partial Singular\n" \
    "Value Decomposition (PSVD).\n" \
    "\n" \
    "Parameters:\n" \
    "  jobu (str): 'N' = don't compute left subspace, 'A' = return (M-RANK) base\n" \
    "              vectors in U, 'S' = return first (min(M,N)-RANK) vectors\n" \
    "  jobv (str): 'N' = don't compute right subspace, 'A' = return (N-RANK) base\n" \
    "              vectors in V, 'S' = return first (min(M,N)-RANK) vectors\n" \
    "  a (ndarray): M-by-N matrix A (F-order, destroyed on exit)\n" \
    "  rank (int): If < 0, compute rank as number of singular values > theta.\n" \
    "              Otherwise specifies desired rank (rank <= min(M,N)).\n" \
    "  theta (float): If rank < 0, upper bound on smallest singular values (>= 0).\n" \
    "                 Otherwise initial estimate for computing upper bound.\n" \
    "  tol (float): Tolerance for negligible elements and singular value multiplicity.\n" \
    "               If <= 0, default tolerance is used.\n" \
    "  reltol (float): Relative tolerance for bisection interval width.\n" \
    "                  If < base*eps, base*eps is used.\n" \
    "\n" \
    "Returns:\n" \
    "  (rank, theta, u, v, q, inul, iwarn, info):\n" \
    "    Computed/adjusted rank; computed upper bound (if rank >= 0 on entry);\n" \
    "    U matrix (or None if jobu='N'); V matrix (or None if jobv='N');\n" \
    "    partially diagonalized bidiagonal q[0:p] diagonal, q[p:2p-1] superdiagonal;\n" \
    "    boolean array where True indicates subspace basis columns;\n" \
    "    warning (1 = rank lowered due to multiplicity);\n" \
    "    exit code (0 = success, 1 = max QR/QL iterations exceeded)"

#define DOC_MB04XY "Apply Householder transformations from bidiagonalization to matrices.\n" \
    "\n" \
    "Applies the Householder transformations Pj stored in factored form into\n" \
    "the columns of array X, to the desired columns of matrix U by premultiplication,\n" \
    "and/or the Householder transformations Qj stored in factored form into the\n" \
    "rows of array X, to the desired columns of matrix V by premultiplication.\n" \
    "The Householder transformations Pj and Qj are stored as produced by DGEBRD.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobu (str): 'N' - don't transform U, 'A' - transform U (M columns),\n" \
    "              'S' - transform U (min(M,N) columns)\n" \
    "  jobv (str): 'N' - don't transform V, 'A' - transform V (N columns),\n" \
    "              'S' - transform V (min(M,N) columns)\n" \
    "  m (int): Number of rows of matrix X (m >= 0)\n" \
    "  n (int): Number of columns of matrix X (n >= 0)\n" \
    "  x (ndarray): M-by-N matrix with Householder vectors (F-order, modified but restored)\n" \
    "  taup (ndarray): Scalar factors of Pj transformations, length min(m,n)\n" \
    "  tauq (ndarray): Scalar factors of Qj transformations, length min(m,n)\n" \
    "  u (ndarray, optional): Matrix U to transform (F-order, required unless jobu='N')\n" \
    "  v (ndarray, optional): Matrix V to transform (F-order, required unless jobv='N')\n" \
    "  inul (ndarray): Boolean array of length max(m,n). inul[i]=True to transform column i.\n" \
    "\n" \
    "Returns:\n" \
    "  (u, v, info): Transformed matrices, exit code (0=success, <0=-i param error)"

#define DOC_MB04YD "Partial diagonalization of a bidiagonal matrix.\n" \
    "\n" \
    "Partially diagonalizes a bidiagonal matrix J using QR or QL iterations\n" \
    "such that J is split into unreduced bidiagonal submatrices whose singular\n" \
    "values are either all larger than a given bound or all smaller than (or\n" \
    "equal to) this bound.\n" \
    "\n" \
    "The left- and right-hand Givens rotations performed on J may be optionally\n" \
    "accumulated in matrices U and V.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobu (str): 'N' do not form U, 'I' initialize U to identity and accumulate,\n" \
    "              'U' update given matrix U\n" \
    "  jobv (str): 'N' do not form V, 'I' initialize V to identity and accumulate,\n" \
    "              'U' update given matrix V\n" \
    "  m (int): Number of rows of matrix U (m >= 0)\n" \
    "  n (int): Number of rows of matrix V (n >= 0)\n" \
    "  rank (int): If < 0, compute rank as number of singular values > theta.\n" \
    "              Otherwise specifies desired rank (rank <= min(m,n)).\n" \
    "  theta (float): If rank < 0, upper bound on smallest singular values (>= 0).\n" \
    "                 Otherwise initial estimate for computing upper bound.\n" \
    "  q (ndarray): Diagonal elements of J, length min(m,n) (F-order)\n" \
    "  e (ndarray): Superdiagonal elements of J, length min(m,n)-1 (F-order)\n" \
    "  tol (float): Tolerance for negligible elements and singular value multiplicity.\n" \
    "               If <= 0, taken as eps * max(|q|, |e|).\n" \
    "  reltol (float): Relative tolerance for bisection interval width.\n" \
    "                  If < base*eps, taken as base*eps.\n" \
    "  u (ndarray, optional): Matrix U (required if jobu='U')\n" \
    "  v (ndarray, optional): Matrix V (required if jobv='U')\n" \
    "\n" \
    "Returns:\n" \
    "  Depends on jobu/jobv:\n" \
    "  - jobu='N', jobv='N': (q, e, theta, rank, inul, iwarn, info)\n" \
    "  - jobu='I'/'U': (q, e, u, theta, rank, inul, iwarn, info)\n" \
    "  - jobv='I'/'U': (q, e, v, theta, rank, inul, iwarn, info)\n" \
    "  - both: (q, e, u, v, theta, rank, inul, iwarn, info)\n" \
    "  where inul (bool array) indicates elements with singular values <= theta."

#define DOC_MB04YW "Perform one QR or QL iteration step on bidiagonal submatrix.\n" \
    "\n" \
    "Performs either one QR or QL iteration step onto the unreduced bidiagonal\n" \
    "submatrix Jk (from index l to k) of a bidiagonal matrix J. The submatrix Jk\n" \
    "is transformed to S' Jk T where S and T are products of Givens rotations.\n" \
    "Optionally accumulates these rotations into U and V matrices.\n" \
    "\n" \
    "Parameters:\n" \
    "  qrit (bool): True for QR iteration (top to bottom), False for QL (bottom to top)\n" \
    "  updatu (bool): True to accumulate left rotations S into U\n" \
    "  updatv (bool): True to accumulate right rotations T into V\n" \
    "  m (int): Number of rows of matrix U (m >= 0)\n" \
    "  n (int): Number of rows of matrix V (n >= 0)\n" \
    "  l (int): Index of first diagonal entry of submatrix (1-based, l >= 1)\n" \
    "  k (int): Index of last diagonal entry of submatrix (1-based, k <= min(m,n))\n" \
    "  shift (float): Shift value for QR/QL iteration step\n" \
    "  d (ndarray): Diagonal entries of bidiagonal J, length min(m,n) (F-order, modified)\n" \
    "  e (ndarray): Superdiagonal entries of J, length min(m,n)-1 (F-order, modified)\n" \
    "  u (ndarray, optional): M-by-p left transformation matrix (F-order, required if updatu)\n" \
    "  v (ndarray, optional): N-by-p right transformation matrix (F-order, required if updatv)\n" \
    "\n" \
    "Returns:\n" \
    "  (d, e, u, v, info): Transformed diagonal/superdiagonal, updated U/V, exit code (0=success)"

#define DOC_MB04ZD "Hamiltonian matrix square-reduction.\n" \
    "\n" \
    "Transforms Hamiltonian H=[[A,G],[Q,-A^T]] to square-reduced form\n" \
    "H'=[[A',G'],[Q',-A'^T]] by orthogonal symplectic similarity.\n" \
    "Square-reduced: Q'A' - A'^T Q' = 0.\n" \
    "\n" \
    "Parameters:\n" \
    "  compu (str): 'N' - no transform, 'I'/'F' - compute U, 'V'/'A' - accumulate\n" \
    "  n (int): Order of matrices A, G, Q (n >= 0)\n" \
    "  a (ndarray): N-by-N matrix A (F-order, modified)\n" \
    "  qg (ndarray): N-by-(N+1) packed Q,G (F-order, modified)\n" \
    "               Q in lower triangular, G in upper triangular cols 1:n\n" \
    "  u (ndarray, optional): N-by-2N transform matrix (for 'V'/'A' modes)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, qg, u, info): Square-reduced matrices and exit code"

#define DOC_MB05MD "Matrix exponential for a real non-defective matrix.\n" \
    "\n" \
    "Computes exp(A*delta) where A is a real N-by-N non-defective matrix\n" \
    "using eigenvalue/eigenvector decomposition (Moler-Van Loan Method 15).\n" \
    "\n" \
    "Parameters:\n" \
    "  balanc (str): 'N' - no scaling, 'S' - diagonal scaling\n" \
    "  n (int): Order of matrix A (n >= 0)\n" \
    "  delta (float): Scalar time parameter\n" \
    "  a (ndarray): N-by-N matrix A (F-order, modified in place)\n" \
    "\n" \
    "Returns:\n" \
    "  (exp_a_delta, v, y, valr, vali, info):\n" \
    "    exp_a_delta: exp(A*delta)\n" \
    "    v: Eigenvector matrix\n" \
    "    y: Intermediate matrix (exp(A*delta) = V*Y)\n" \
    "    valr: Real parts of eigenvalues\n" \
    "    vali: Imaginary parts of eigenvalues\n" \
    "    info: Exit code"

#define DOC_MB05MY "Schur form, eigenvalues, and right eigenvectors.\n" \
    "\n" \
    "Computes for an N-by-N real nonsymmetric matrix A:\n" \
    "  - Orthogonal matrix Q reducing A to real Schur form T\n" \
    "  - Eigenvalues (WR + i*WI)\n" \
    "  - Right eigenvectors R of T (upper triangular)\n" \
    "\n" \
    "Parameters:\n" \
    "  balanc (str): 'N' - no scaling, 'S' - diagonal scaling\n" \
    "  a (ndarray): N-by-N matrix A (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (wr, wi, r, q, t, info): Eigenvalues, eigenvectors, Schur form"

#define DOC_MB05ND "Matrix exponential and integral.\n" \
    "\n" \
    "Computes F(delta) = exp(A*delta) and\n" \
    "H(delta) = integral from 0 to delta of exp(A*s) ds.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Order of matrix A (n >= 0)\n" \
    "  delta (float): Scalar time parameter\n" \
    "  a (ndarray): n-by-n matrix A (F-order)\n" \
    "  tol (float): Tolerance for Pade approximation order\n" \
    "\n" \
    "Returns:\n" \
    "  (ex, exint, info): exp(A*delta), integral H(delta), exit code"

#define DOC_MB05OD "Matrix exponential with accuracy estimate.\n" \
    "\n" \
    "Computes exp(A*delta) where A is real N-by-N matrix using diagonal\n" \
    "Pade approximation with scaling and squaring.\n" \
    "\n" \
    "Parameters:\n" \
    "  balanc (str): 'N' - no balancing, 'S' - use scaling (via MB04MD)\n" \
    "  n (int): Order of matrix A (n >= 0)\n" \
    "  ndiag (int): Order of diagonal Pade approximant (1-15, default 9)\n" \
    "  delta (float): Scalar time parameter\n" \
    "  a (ndarray): N-by-N matrix A (F-order, modified in place)\n" \
    "\n" \
    "Returns:\n" \
    "  (exp_a, mdig, idig, iwarn, info):\n" \
    "    exp_a: exp(A*delta)\n" \
    "    mdig: Minimal accurate digits in 1-norm of result\n" \
    "    idig: Accurate digits at 95% confidence level\n" \
    "    iwarn: Warning (0=ok, 1=possible, 2=severe inaccuracy, 3=balancing unused)\n" \
    "    info: Exit code (0=ok, 1=norm too large, 2=singular, 3=overflow)"

#define DOC_MB05OY "Restore matrix after balancing transformations.\n" \
    "\n" \
    "Computes A <- P * D * A * D^{-1} * P' where P is permutation,\n" \
    "D is diagonal scaling from DGEBAL.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'N'=nothing, 'P'=permute, 'S'=scale, 'B'=both\n" \
    "  n (int): Order of matrix A (n >= 0)\n" \
    "  low (int): Low index from DGEBAL (1-based)\n" \
    "  igh (int): High index from DGEBAL (1-based)\n" \
    "  a (ndarray): N-by-N matrix (F-order, modified)\n" \
    "  scale (ndarray): Permutation/scaling from DGEBAL\n" \
    "\n" \
    "Returns:\n" \
    "  (a, info): Back-transformed matrix and exit code"

#define DOC_MB3JZP "Reorder eigenvalues of complex skew-Hamiltonian/Hamiltonian pencil (panel variant).\n" \
    "\n" \
    "Panel-based blocked variant of MB03JZ for better performance on large matrices.\n" \
    "Moves eigenvalues with strictly negative real parts of an N-by-N complex\n" \
    "skew-Hamiltonian/Hamiltonian pencil aS - bH in structured Schur form to the\n" \
    "leading principal subpencil, while keeping the triangular form.\n" \
    "\n" \
    "On entry: S = [[A, D], [0, A']], H = [[B, F], [0, -B']]\n" \
    "where A, B are upper triangular, D is skew-Hermitian, F is Hermitian.\n" \
    "\n" \
    "Parameters:\n" \
    "  compq (str): 'N'=no Q, 'I'=init Q to identity, 'U'=update Q\n" \
    "  n (int): Order of the pencil (n >= 0, even)\n" \
    "  a (ndarray): Upper triangular matrix A (n/2 x n/2, F-order, complex)\n" \
    "  d (ndarray): Skew-Hermitian matrix D upper part (n/2 x n/2, F-order, complex)\n" \
    "  b (ndarray): Upper triangular matrix B (n/2 x n/2, F-order, complex)\n" \
    "  f (ndarray): Hermitian matrix F upper part (n/2 x n/2, F-order, complex)\n" \
    "  q (ndarray, optional): Unitary matrix Q (n x n, F-order, complex)\n" \
    "  tol (float, optional): Tolerance for eigenvalue sign (default: MIN(N,10)*EPS)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, d, b, f, q, neig, info):\n" \
    "    a, d, b, f: Transformed matrices (complex)\n" \
    "    q: Unitary matrix Q (complex)\n" \
    "    neig (int): Number of eigenvalues with negative real part\n" \
    "    info (int): 0 = success, <0 = -i means i-th argument invalid"

#define DOC_MB3LZP "Eigenvalues and deflating subspace of complex skew-Hamiltonian/Hamiltonian pencil.\n" \
    "\n" \
    "Computes eigenvalues of a complex N-by-N skew-Hamiltonian/Hamiltonian pencil\n" \
    "aS - bH where:\n" \
    "  S = [[A, D], [E, A']], with D skew-Hermitian, E skew-Hermitian\n" \
    "  H = [[B, F], [G, -B']], with F Hermitian, G Hermitian\n" \
    "\n" \
    "The routine embeds the complex pencil into a real skew-Hamiltonian/\n" \
    "skew-Hamiltonian pencil, applies MB04FP for the structured Schur form,\n" \
    "MB3JZP to reorder eigenvalues, and optionally computes the right\n" \
    "deflating subspace via QR with pivoting or SVD.\n" \
    "\n" \
    "Parameters:\n" \
    "  compq (str): 'N'=eigenvalues only, 'C'=also compute deflating subspace\n" \
    "  orth (str): Orthogonalization method (if compq='C'): 'P'=QR with pivoting, 'S'=SVD\n" \
    "  n (int): Order of pencil (n >= 0, must be even)\n" \
    "  a (ndarray): N/2-by-N/2 complex matrix A (F-order)\n" \
    "  de (ndarray): N-by-N complex packed D and E matrices (F-order)\n" \
    "  b (ndarray): N/2-by-N/2 complex matrix B (F-order)\n" \
    "  fg (ndarray): N-by-N complex packed F and G matrices (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  If compq='N':\n" \
    "    (alphar, alphai, beta, neig, info)\n" \
    "  If compq='C':\n" \
    "    (a, de, b, fg, q, alphar, alphai, beta, neig, info)\n" \
    "  where:\n" \
    "    a, de, b, fg: Transformed matrices (complex)\n" \
    "    q: 2N-by-2N complex unitary matrix, leading N-by-NEIG columns contain\n" \
    "       orthonormal basis of right deflating subspace\n" \
    "    alphar, alphai, beta: Eigenvalue components (lambda = (alphar+i*alphai)/beta)\n" \
    "    neig: Number of eigenvalues with negative real part\n" \
    "    info: 0=success, 1=MB04FP QZ failed, 2=ZHGEQZ failed, 3=ZGESVD failed, 4=singular"

#define DOC_MB3OYZ "Complex rank-revealing QR factorization with column pivoting.\n" \
    "\n" \
    "Computes truncated QR factorization A*P = Q*R with column pivoting,\n" \
    "estimating effective rank using incremental condition estimation.\n" \
    "\n" \
    "Parameters:\n" \
    "  m (int): Number of rows (m >= 0)\n" \
    "  n (int): Number of columns (n >= 0)\n" \
    "  a (ndarray): Complex matrix (m x n, F-order)\n" \
    "  rcond (float): Rank threshold (0 <= rcond <= 1)\n" \
    "  svlmax (float): Estimate of largest singular value, or 0\n" \
    "\n" \
    "Returns:\n" \
    "  (a, rank, info, sval, jpvt, tau): QR factors, rank, exit code,\n" \
    "    singular value estimates, column permutation, reflector factors"

#define DOC_MB3PYZ "Complex rank-revealing RQ factorization with row pivoting.\n" \
    "\n" \
    "Computes truncated RQ factorization P*A = R*Q with row pivoting,\n" \
    "estimating effective rank using incremental condition estimation.\n" \
    "\n" \
    "Parameters:\n" \
    "  m (int): Number of rows (m >= 0)\n" \
    "  n (int): Number of columns (n >= 0)\n" \
    "  a (ndarray): Complex matrix (m x n, F-order)\n" \
    "  rcond (float): Rank threshold (0 <= rcond <= 1)\n" \
    "  svlmax (float): Estimate of largest singular value, or 0\n" \
    "\n" \
    "Returns:\n" \
    "  (a, rank, info, sval, jpvt, tau): RQ factors, rank, exit code,\n" \
    "    singular value estimates, row permutation, reflector factors"

#define DOC_MB4DBZ "Inverse balancing transformation for complex skew-Hamiltonian/Hamiltonian eigenvectors.\n" \
    "\n" \
    "Applies from the left the inverse of a balancing transformation computed by MB4DPZ\n" \
    "to the complex matrix [[V1], [sgn*V2]] where sgn is +1 or -1.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'N' - do nothing, 'P' - permutation only, 'S' - scaling only, 'B' - both\n" \
    "  sgn (str): 'P' - sgn=+1, 'N' - sgn=-1\n" \
    "  n (int): Number of rows of V1 and V2 (n >= 0)\n" \
    "  ilo (int): Index from MB4DPZ (1 <= ilo <= n+1)\n" \
    "  lscale (ndarray): Permutation and left scaling factors (dimension n)\n" \
    "  rscale (ndarray): Permutation and right scaling factors (dimension n)\n" \
    "  v1 (ndarray): Complex N-by-M matrix V1 (F-order, modified in place)\n" \
    "  v2 (ndarray): Complex N-by-M matrix V2 (F-order, modified in place)\n" \
    "\n" \
    "Returns:\n" \
    "  (v1, v2, info): Modified matrices and exit code (info=0: success, <0: -info arg illegal)"

#define DOC_MB4DLZ "Balance a complex matrix pencil (A,B).\n" \
    "\n" \
    "Balances a pair of N-by-N complex matrices (A,B) by equivalence transformations.\n" \
    "This involves permuting to isolate eigenvalues and/or diagonal scaling to make\n" \
    "rows and columns as close in 1-norm as possible. Improves eigenvalue accuracy.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'N' - none, 'P' - permute only, 'S' - scale only, 'B' - both\n" \
    "  n (int): Order of matrices A and B (n >= 0)\n" \
    "  thresh (float): Threshold for scaling (see SLICOT docs for negative values)\n" \
    "  a (ndarray): Complex N-by-N matrix A (F-order, modified in place)\n" \
    "  b (ndarray): Complex N-by-N matrix B (F-order, modified in place)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, ilo, ihi, lscale, rscale, dwork, iwarn, info):\n" \
    "    a, b: Balanced matrices\n" \
    "    ilo, ihi: Indices of isolated eigenvalues (1-based)\n" \
    "    lscale: Left permutation/scaling factors\n" \
    "    rscale: Right permutation/scaling factors\n" \
    "    dwork: Contains initial/final 1-norms and threshold used\n" \
    "    iwarn: Warning indicator (0=ok, 1=scaling reset to 1)\n" \
    "    info: Exit code (0=success, <0: -info arg illegal)"

#define DOC_MB4DPZ "Balance a complex skew-Hamiltonian/Hamiltonian pencil.\n" \
    "\n" \
    "Balances the 2N-by-2N complex pencil aS - bH where:\n" \
    "  S = [[A, D], [E, A']] (skew-Hamiltonian, D,E skew-Hermitian)\n" \
    "  H = [[C, V], [W, -C']] (Hamiltonian, V,W Hermitian)\n" \
    "and ' denotes conjugate transpose.\n" \
    "\n" \
    "First permutes to isolate eigenvalues, then applies diagonal scaling\n" \
    "to make row/column pairs close in 1-norm.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'N' - none, 'P' - permute only, 'S' - scale only, 'B' - both\n" \
    "  n (int): Order of matrices A, C, D, E, V, W (n >= 0)\n" \
    "  thresh (float): Threshold for scaling (-1,-2,-3,-4 or <=-10 for auto search)\n" \
    "  a (ndarray): Complex N-by-N matrix A (F-order, modified in place)\n" \
    "  de (ndarray): Complex N-by-(N+1) matrix with lower tri of E in cols 1:N,\n" \
    "                upper tri of D in cols 2:N+1 (F-order, modified in place)\n" \
    "  c (ndarray): Complex N-by-N matrix C (F-order, modified in place)\n" \
    "  vw (ndarray): Complex N-by-(N+1) matrix with lower tri of W in cols 1:N,\n" \
    "                upper tri of V in cols 2:N+1 (F-order, modified in place)\n" \
    "\n" \
    "Returns:\n" \
    "  (ilo, lscale, rscale, dwork, iwarn, info):\n" \
    "    ilo: ILO-1 = number of deflated eigenvalues (1-based)\n" \
    "    lscale: Left permutation/scaling factors\n" \
    "    rscale: Right permutation/scaling factors\n" \
    "    dwork: Contains initial/final 1-norms (S,H) and threshold used\n" \
    "    iwarn: Warning indicator (0=ok, 1=scaling reset to 1)\n" \
    "    info: Exit code (0=success, <0: -info arg illegal)"

#define DOC_MC01MD "Compute shifted polynomial coefficients using Horner's algorithm.\n" \
    "\n" \
    "Given P(x) = p[0] + p[1]*x + ... + p[dp]*x^dp and scalar alpha,\n" \
    "computes the first k coefficients of the shifted polynomial:\n" \
    "P(x) = q[0] + q[1]*(x-alpha) + ... + q[k-1]*(x-alpha)^(k-1) + ...\n" \
    "\n" \
    "Parameters:\n" \
    "  alpha (float): Shift value\n" \
    "  k (int): Number of shifted coefficients to compute (1 <= k <= dp+1)\n" \
    "  p (ndarray): Polynomial coefficients in increasing powers of x\n" \
    "\n" \
    "Returns:\n" \
    "  (q, info): Shifted coefficients (leading k meaningful), exit code"

#define DOC_MC01ND "Evaluate real polynomial at complex point using Horner's algorithm.\n" \
    "\n" \
    "Computes P(x0) where P(x) = p[0] + p[1]*x + ... + p[dp]*x^dp is a\n" \
    "real polynomial and x0 = xr + xi*j is a complex point.\n" \
    "\n" \
    "Parameters:\n" \
    "  xr (float): Real part of evaluation point\n" \
    "  xi (float): Imaginary part of evaluation point\n" \
    "  p (ndarray): Polynomial coefficients in increasing powers of x\n" \
    "\n" \
    "Returns:\n" \
    "  (vr, vi, info): Real and imaginary parts of P(x0), exit code"

#define DOC_MC01OD "Compute complex polynomial coefficients from zeros.\n" \
    "\n" \
    "Computes the coefficients of a complex polynomial P(x) from its zeros:\n" \
    "  P(x) = (x - r(1)) * (x - r(2)) * ... * (x - r(K))\n" \
    "where r(i) = REZ(i) + j*IMZ(i).\n" \
    "\n" \
    "Parameters:\n" \
    "  rez (ndarray): Real parts of zeros\n" \
    "  imz (ndarray): Imaginary parts of zeros\n" \
    "\n" \
    "Returns:\n" \
    "  (rep, imp, info): Real and imaginary parts of coefficients\n" \
    "    (increasing powers), exit code"

#define DOC_MC01PD "Compute polynomial coefficients from zeros.\n" \
    "\n" \
    "Parameters:\n" \
    "  rez (ndarray): Real parts of zeros\n" \
    "  imz (ndarray): Imaginary parts of zeros\n" \
    "\n" \
    "Returns:\n" \
    "  (p, info): Polynomial coefficients (increasing powers), exit code"

#define DOC_MC01PY "Compute polynomial coefficients from zeros (decreasing order).\n" \
    "\n" \
    "Computes the coefficients of a real polynomial P(x) from its zeros:\n" \
    "  P(x) = (x - r(1)) * (x - r(2)) * ... * (x - r(K))\n" \
    "where r(i) = REZ(i) + j*IMZ(i). Complex zeros must appear as\n" \
    "consecutive conjugate pairs.\n" \
    "\n" \
    "Unlike mc01pd, coefficients are stored in DECREASING powers of x.\n" \
    "\n" \
    "Parameters:\n" \
    "  rez (ndarray): Real parts of zeros\n" \
    "  imz (ndarray): Imaginary parts of zeros\n" \
    "\n" \
    "Returns:\n" \
    "  (p, info): Polynomial coefficients (decreasing powers), exit code"

#define DOC_MC01QD "Polynomial division: quotient and remainder.\n" \
    "\n" \
    "Computes Q(x) and R(x) such that A(x) = B(x) * Q(x) + R(x)\n" \
    "where deg(R) < deg(B).\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): Dividend polynomial coefficients (increasing powers)\n" \
    "  b (ndarray): Divisor polynomial coefficients (increasing powers)\n" \
    "\n" \
    "Returns:\n" \
    "  (rq, db, iwarn, info): Combined [R | Q] array, divisor degree,\n" \
    "    warning count (leading zeros removed), exit code"

#define DOC_MC01RD "Compute polynomial P(x) = P1(x)*P2(x) + alpha*P3(x).\n" \
    "\n" \
    "Computes the coefficients of the polynomial:\n" \
    "  P(x) = P1(x) * P2(x) + alpha * P3(x)\n" \
    "where P1(x), P2(x), P3(x) are real polynomials and alpha is a scalar.\n" \
    "\n" \
    "Parameters:\n" \
    "  p1 (ndarray): Coefficients of P1(x) in increasing powers of x\n" \
    "  p2 (ndarray): Coefficients of P2(x) in increasing powers of x\n" \
    "  p3 (ndarray): Coefficients of P3(x) in increasing powers of x (modified on exit)\n" \
    "  alpha (float): Scalar multiplier for P3(x)\n" \
    "  dp1 (int, optional): Degree of P1(x), -1 for zero polynomial\n" \
    "  dp2 (int, optional): Degree of P2(x), -1 for zero polynomial\n" \
    "  dp3 (int, optional): Degree of P3(x), -1 for zero polynomial\n" \
    "\n" \
    "Returns:\n" \
    "  (p, dp, info): Result polynomial coefficients, degree, exit code"

#define DOC_MC01SD "Scale polynomial coefficients for minimal variation.\n" \
    "\n" \
    "Scales the coefficients of real polynomial P(x) such that the\n" \
    "coefficients of Q(x) = s*P(t*x) have minimal variation, where\n" \
    "s = BASE^S and t = BASE^T (BASE is the machine floating-point base).\n" \
    "\n" \
    "Parameters:\n" \
    "  p (ndarray): Polynomial coefficients in increasing powers of x\n" \
    "\n" \
    "Returns:\n" \
    "  (q, s, t, mant, e, info): Scaled coefficients Q(x), scaling exponents S and T,\n" \
    "    mantissas and exponents of Q(x) coefficients, exit code (0=success, 1=zero polynomial)"

#define DOC_MC01SW "Extract mantissa and exponent of a real number.\n" \
    "\n" \
    "Finds M and E such that A = M * B^E where 1 <= |M| < B.\n" \
    "If A = 0, then M = E = 0.\n" \
    "\n" \
    "Parameters:\n" \
    "  a (float): The real number to decompose\n" \
    "  b (int): The base (>= 2)\n" \
    "\n" \
    "Returns:\n" \
    "  (m, e): Mantissa and exponent"

#define DOC_MC01SX "Compute variation of exponents in floating-point series.\n" \
    "\n" \
    "Parameters:\n" \
    "  e (ndarray): Array of integer exponents\n" \
    "  mant (ndarray): Array of mantissas (f64)\n" \
    "\n" \
    "Returns:\n" \
    "  int: Variation V = max(E(j)) - min(E(j)) for non-zero mantissas"

#define DOC_MC01SY "Reconstruct a real number from mantissa and exponent.\n" \
    "\n" \
    "Computes A = M * B^E given mantissa M and exponent E.\n" \
    "\n" \
    "Parameters:\n" \
    "  m (float): The mantissa\n" \
    "  e (int): The exponent\n" \
    "  b (int): The base (>= 2)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, ovflow): The reconstructed value and overflow flag"

#define DOC_MC01TD "Determine polynomial stability (Routh or Schur-Cohn).\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C'=continuous-time, 'D'=discrete-time\n" \
    "  p (ndarray): Polynomial coefficients in increasing powers\n" \
    "\n" \
    "Returns:\n" \
    "  (stable, nz, dp_out, iwarn, info): Stability flag, unstable zeros,\n" \
    "    actual degree, warning, exit code"

#define DOC_MC01VD "Compute roots of a quadratic equation with real coefficients.\n" \
    "\n" \
    "Computes the roots of: a*x^2 + b*x + c = 0\n" \
    "\n" \
    "The algorithm uses a numerically stable formula to avoid loss of significance.\n" \
    "\n" \
    "Parameters:\n" \
    "  a (float): Coefficient of x^2\n" \
    "  b (float): Coefficient of x\n" \
    "  c (float): Constant term\n" \
    "\n" \
    "Returns:\n" \
    "  (z1re, z1im, z2re, z2im, info): Real and imaginary parts of roots.\n" \
    "    z1 is the largest root in magnitude, z2 is the smallest.\n" \
    "    info = 0: success\n" \
    "    info = 1: a=b=0 or a=0 and -c/b overflows\n" \
    "    info = 2: a=0 (linear equation), z1re=BIG\n" \
    "    info = 3: c=0 and -b/a overflows, or largest root overflows\n" \
    "    info = 4: roots cannot be computed without overflow"

#define DOC_MC01WD "Compute polynomial quotient and remainder for quadratic divisor.\n" \
    "\n" \
    "Divides P(x) by (x^2 - u1*x + u2) returning quotient coefficients.\n" \
    "\n" \
    "Parameters:\n" \
    "  p (ndarray): Polynomial coefficients in increasing powers\n" \
    "  u1 (float): Linear coefficient of divisor\n" \
    "  u2 (float): Constant coefficient of divisor\n" \
    "\n" \
    "Returns:\n" \
    "  (q, info): Quotient polynomial coefficients, exit code"

#define DOC_MC01XD "Compute roots of cubic polynomial.\n" \
    "\n" \
    "Computes roots of P(t) = alpha + beta*t + gamma*t^2 + delta*t^3.\n" \
    "Roots are returned as quotients (EVR + i*EVI) / EVQ.\n" \
    "\n" \
    "Parameters:\n" \
    "  alpha (float): Constant coefficient\n" \
    "  beta (float): Linear coefficient\n" \
    "  gamma (float): Quadratic coefficient\n" \
    "  delta (float): Cubic coefficient\n" \
    "\n" \
    "Returns:\n" \
    "  (evr, evi, evq, info): Real parts, imaginary parts, quotients, exit code"

#define DOC_MC03MD "Compute polynomial matrix P(x) = P1(x)*P2(x) + alpha*P3(x).\n" \
    "\n" \
    "Computes the coefficients of the polynomial matrix:\n" \
    "  P(x) = P1(x) * P2(x) + alpha * P3(x)\n" \
    "where P1(x), P2(x), P3(x) are real polynomial matrices and alpha is a scalar.\n" \
    "\n" \
    "Parameters:\n" \
    "  p1 (ndarray): 3D array (rp1, cp1, dp1+1) of P1 coefficients\n" \
    "  p2 (ndarray): 3D array (cp1, cp2, dp2+1) of P2 coefficients\n" \
    "  p3 (ndarray): 3D array (rp1, cp2, dp3+1) of P3 coefficients\n" \
    "  alpha (float): Scalar multiplier for P3(x)\n" \
    "  dp1 (int, optional): Degree of P1(x), -1 for zero polynomial\n" \
    "  dp2 (int, optional): Degree of P2(x), -1 for zero polynomial\n" \
    "  dp3 (int, optional): Degree of P3(x), -1 for zero polynomial\n" \
    "\n" \
    "Returns:\n" \
    "  (p, dp, info): Result polynomial matrix coefficients (rp1, cp2, max_deg+1), degree, exit code"

#define DOC_MC03ND "Compute minimal polynomial basis for right nullspace of polynomial matrix.\n" \
    "\n" \
    "Computes the coefficients of a minimal polynomial basis K(s) for the right\n" \
    "nullspace of an MP-by-NP polynomial matrix P(s) of degree DP, solving:\n" \
    "  P(s) * K(s) = 0\n" \
    "\n" \
    "K(s) = K(0) + K(1)*s + ... + K(DK)*s^DK\n" \
    "\n" \
    "Parameters:\n" \
    "  mp (int): Number of rows of P(s). mp >= 0.\n" \
    "  np (int): Number of columns of P(s). np >= 0.\n" \
    "  dp (int): Degree of P(s). dp >= 1.\n" \
    "  p (ndarray): 3D array (mp, np, dp+1) of polynomial coefficients.\n" \
    "               p[:,:,k] = coefficient matrix P(k) for s^k.\n" \
    "  tol (float): Tolerance for rank determination.\n" \
    "\n" \
    "Returns:\n" \
    "  (dk, gam, nullsp, ker, info):\n" \
    "    dk (int): Degree of K(s). -1 if no right nullspace.\n" \
    "    gam (ndarray): Information about nullspace vector ordering.\n" \
    "    nullsp (ndarray): Right nullspace vectors in condensed form.\n" \
    "    ker (ndarray): 3D array (np, nk, dk+1) of kernel coefficients.\n" \
    "    info (int): Exit code. 0 = success."

#define DOC_MC03NX "Construct companion pencil from polynomial matrix.\n" \
    "\n" \
    "Constructs the pencil s*E - A related to a polynomial matrix P(s) of degree dp:\n" \
    "  P(s) = P(0) + P(1)*s + ... + P(dp)*s^dp\n" \
    "\n" \
    "The pencil matrices have dimensions (dp*mp) x ((dp-1)*mp + np).\n" \
    "\n" \
    "Parameters:\n" \
    "  p (ndarray): 3D array (mp, np, dp+1) of polynomial coefficients.\n" \
    "              p[:,:,k] = coefficient matrix P(k) for s^k.\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e): Companion pencil matrices A and E"

#define DOC_MC03NY "Minimal polynomial basis for right nullspace of staircase pencil.\n" \
    "\n" \
    "Determines a minimal basis of the right nullspace of the subpencil\n" \
    "s*E(eps)-A(eps) using the method given in Beelen's thesis. This pencil\n" \
    "must be in staircase form as supplied by MB04VD.\n" \
    "\n" \
    "NOTE: This routine is intended to be called only from MC03ND.\n" \
    "\n" \
    "Parameters:\n" \
    "  nblcks (int): Number of full row rank blocks (>= 0)\n" \
    "  nra (int): Number of rows = sum(nu(i)) (>= 0)\n" \
    "  nca (int): Number of columns = sum(mu(i)) (>= 0)\n" \
    "  a (ndarray): Matrix A (nra x nca, F-order), modified on exit\n" \
    "  e (ndarray): Matrix E (nra x nca, F-order), modified on exit\n" \
    "  imuk (ndarray): Column dimensions mu(k), int32 array\n" \
    "  inuk (ndarray): Row dimensions nu(k), int32 array\n" \
    "\n" \
    "Returns:\n" \
    "  (veps, imuk, info): Minimal polynomial basis (nca x ncv),\n" \
    "    restored mu(k) array, exit code (>0 means block not full row rank)"

#define DOC_MD03BA "QR factorization with column pivoting for Levenberg-Marquardt.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Number of columns of J\n" \
    "  ipar (ndarray): Integer parameters (M=ipar[0])\n" \
    "  fnorm (float): Norm of error vector\n" \
    "  j (ndarray): Jacobian matrix (M x N, F-order)\n" \
    "  e (ndarray): Error vector (M)\n" \
    "\n" \
    "Returns:\n" \
    "  (r, e, jnorms, gnorm, ipvt, info): R factor, Q'*e, norms, gradient norm, permutation, exit code"

#define DOC_MD03BB "Compute Levenberg-Marquardt parameter for compressed Jacobian (Wrapper for MD03BY).\n" \
    "\n" \
    "Parameters:\n" \
    "  cond (str): 'E' = estimate condition, 'N' = check zeros, 'U' = use rank\n" \
    "  n (int): Order of matrix R\n" \
    "  r (ndarray): Upper triangular matrix R (n x n, F-order)\n" \
    "  ipvt (ndarray): Permutation vector (1-based indices)\n" \
    "  diag (ndarray): Diagonal scaling D (all nonzero)\n" \
    "  qtb (ndarray): First n elements of Q'*b\n" \
    "  delta (float): Trust region radius (> 0)\n" \
    "  par (float): Initial LM parameter estimate (>= 0)\n" \
    "  rank (int): Input rank (COND='U') or 0 otherwise\n" \
    "  tol (float): Tolerance for rank determination (COND='E')\n" \
    "\n" \
    "Returns:\n" \
    "  (r, par, rank, x, rx, info): Modified R, LM parameter, rank, solution, residual, exit code"

#define DOC_MD03BD "Levenberg-Marquardt nonlinear least squares optimizer.\n" \
    "\n" \
    "Parameters:\n" \
    "  m (int): Number of functions\n" \
    "  n (int): Number of variables\n" \
    "  x (ndarray): Initial guess (n,) or random if not provided\n" \
    "  fcn (callable): Function f(x) returning residual vector (m,)\n" \
    "  jac (callable): Function f(x) returning Jacobian matrix (m, n)\n" \
    "  itmax (int): Maximum iterations (default 100)\n" \
    "  ftol (float): Function tolerance (default sqrt(eps))\n" \
    "  xtol (float): Solution tolerance (default sqrt(eps))\n" \
    "  gtol (float): Gradient tolerance (default eps)\n" \
    "\n" \
    "Returns:\n" \
    "  (x, nfev, njev, fnorm, iwarn, info): Solution, evaluations, norm, status"

#define DOC_MD03BF "Kowalik-Osborne test function for nonlinear least squares.\n" \
    "\n" \
    "Example FCN callback implementing the Kowalik-Osborne test problem\n" \
    "from MINPACK with m=15 observations and n=3 parameters.\n" \
    "\n" \
    "Parameters:\n" \
    "  iflag (int): Operation mode (1=error, 2=Jacobian, 3=workspace)\n" \
    "  x (ndarray): Parameter vector (3,)\n" \
    "\n" \
    "Returns:\n" \
    "  iflag=1: (e, info) - Error vector (15,), exit code\n" \
    "  iflag=2: (j, e, nfevl, info) - Jacobian (15x3), error, evals, code\n" \
    "  iflag=3: (ipar, info) - Workspace requirements tuple, exit code"

#define DOC_MD03BY "Compute Levenberg-Marquardt parameter for trust region subproblem.\n" \
    "\n" \
    "Parameters:\n" \
    "  cond (str): 'E' = estimate condition, 'N' = check zeros, 'U' = use rank\n" \
    "  n (int): Order of matrix R\n" \
    "  r (ndarray): Upper triangular matrix R (n x n, F-order)\n" \
    "  ipvt (ndarray): Permutation vector (1-based indices)\n" \
    "  diag (ndarray): Diagonal scaling D (all nonzero)\n" \
    "  qtb (ndarray): First n elements of Q'*b\n" \
    "  delta (float): Trust region radius (> 0)\n" \
    "  par (float): Initial LM parameter estimate (>= 0)\n" \
    "  rank (int): Input rank (COND='U') or 0 otherwise\n" \
    "  tol (float): Tolerance for rank determination (COND='E')\n" \
    "\n" \
    "Returns:\n" \
    "  (r, par, rank, x, rx, info): Modified R, LM parameter, rank, solution, residual, exit code"

#define DOC_NF01AY "Calculate the output of a set of neural networks.\n" \
    "\n" \
    "Parameters:\n" \
    "  nsmp (int): Number of training samples\n" \
    "  nz (int): Length of each input sample\n" \
    "  l (int): Length of each output sample\n" \
    "  ipar (ndarray): Integer parameters (nn, ...)\n" \
    "  wb (ndarray): Weights and biases\n" \
    "  z (ndarray): Input samples (nsmp x nz, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (y, info): Output samples, exit code"

#define DOC_NF01BR "Solve system of linear equations R*x = b or R'*x = b in least squares sense.\n" \
    "\n" \
    "Parameters:\n" \
    "  cond (str): 'E' = estimate condition, 'N' = check zeros, 'U' = use rank\n" \
    "  uplo (str): 'U' or 'L' - storage scheme\n" \
    "  trans (str): 'N'/'T'/'C' - transpose option\n" \
    "  n (int): Order of matrix R\n" \
    "  ipar (ndarray): Integer parameters (st, bn, bsm, bsn)\n" \
    "  r (ndarray): Matrix R (ldr x nc, F-order)\n" \
    "  sdiag (ndarray): Diagonal elements (if uplo='L')\n" \
    "  s (ndarray): Transpose of last block column (if uplo='L')\n" \
    "  b (ndarray): RHS vector b\n" \
    "  ranks (ndarray): Ranks\n" \
    "  tol (float): Tolerance for rank determination\n" \
    "\n" \
    "Returns:\n" \
    "  (b, ranks, info): Solution x, ranks, exit code"

#define DOC_NF01BS "QR factorization of Jacobian in compressed form.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Number of columns of J\n" \
    "  ipar (ndarray): Integer parameters (st, bn, bsm, bsn)\n" \
    "  fnorm (float): Norm of error vector\n" \
    "  j (ndarray): Jacobian matrix (ldj x nc, F-order)\n" \
    "  e (ndarray): Error vector\n" \
    "\n" \
    "Returns:\n" \
    "  (r, e, jnorms, gnorm, ipvt, info): R factor, Q'*e, norms, gradient norm, permutation, exit code"

#define DOC_NF01BY "Compute the Jacobian of the error function for a neural network.\n" \
    "\n" \
    "Parameters:\n" \
    "  cjte (str): 'C' to compute J'*e, 'N' to skip\n" \
    "  nsmp (int): Number of training samples\n" \
    "  nz (int): Length of each input sample\n" \
    "  l (int): Length of each output sample (must be 1)\n" \
    "  ipar (ndarray): Integer parameters (nn, ...)\n" \
    "  wb (ndarray): Weights and biases\n" \
    "  z (ndarray): Input samples (nsmp x nz, F-order)\n" \
    "  e (ndarray): Error vector (nsmp)\n" \
    "\n" \
    "Returns:\n" \
    "  (j, jte, info): Jacobian, J'*e, exit code"

#define DOC_SB01BD "Multi-input pole assignment using state feedback.\n" \
    "\n" \
    "Computes state feedback F such that A + B*F has specified eigenvalues.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous, 'D' for discrete\n" \
    "  n (int): State dimension\n" \
    "  m (int): Input dimension\n" \
    "  np (int): Number of eigenvalues to assign\n" \
    "  alpha (float): Threshold for fixed eigenvalues\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  wr (ndarray): Real parts of desired eigenvalues\n" \
    "  wi (ndarray): Imaginary parts of desired eigenvalues\n" \
    "  tol (float): Tolerance for controllability tests\n" \
    "\n" \
    "Returns:\n" \
    "  (a, wr, wi, nfp, nap, nup, f, z, iwarn, info):\n" \
    "    Schur form, eigenvalues, counts, feedback, transformation"

#define DOC_SB01BX "Select eigenvalue(s) closest to a given value.\n" \
    "\n" \
    "Chooses real eigenvalue or complex conjugate pair at minimal distance.\n" \
    "\n" \
    "Parameters:\n" \
    "  reig (bool): True for real, False for complex pair\n" \
    "  n (int): Number of eigenvalues\n" \
    "  xr (float): Real part of target\n" \
    "  xi (float): Imaginary part of target (for complex)\n" \
    "  wr (ndarray): Real parts of eigenvalues\n" \
    "  wi (ndarray): Imaginary parts of eigenvalues\n" \
    "\n" \
    "Returns:\n" \
    "  (wr, wi, s, p): Reordered arrays, sum, product"

#define DOC_SB01BY "Pole placement for N=1 or N=2 systems.\n" \
    "\n" \
    "Constructs feedback F such that A + B*F has prescribed eigenvalues.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Order of A (1 or 2)\n" \
    "  m (int): Number of inputs\n" \
    "  s (float): Sum of eigenvalues\n" \
    "  p (float): Product of eigenvalues (for N=2)\n" \
    "  a (ndarray): State matrix (n x n, F-order)\n" \
    "  b (ndarray): Input matrix (n x m, F-order)\n" \
    "  tol (float): Tolerance for controllability\n" \
    "\n" \
    "Returns:\n" \
    "  (f, info): Feedback matrix (m x n), exit code"

#define DOC_SB01DD "Eigenstructure assignment for multi-input system in orthogonal canonical form.\n" \
    "\n" \
    "Computes feedback matrix G such that A - B*G has the desired eigenstructure.\n" \
    "The pair (A, B) must be in orthogonal canonical form from ab01nd.\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): N-by-N matrix A in canonical form (F-order)\n" \
    "  b (ndarray): N-by-M matrix B in canonical form (F-order)\n" \
    "  indcon (int): Controllability index (0 <= indcon <= n)\n" \
    "  nblk (ndarray): Block sizes (indcon elements, int32)\n" \
    "  wr (ndarray): Real parts of desired poles\n" \
    "  wi (ndarray): Imaginary parts of desired poles\n" \
    "  z (ndarray): Orthogonal matrix from ab01nd (n x n, F-order)\n" \
    "  y (ndarray): Free parameters for eigenvector design (m*n elements)\n" \
    "  tol (float, optional): Tolerance for rank determination\n" \
    "\n" \
    "Returns:\n" \
    "  (a_schur, b_out, z_out, g, count, info):\n" \
    "    a_schur: Real Schur form of A-B*G\n" \
    "    b_out: Transformed B\n" \
    "    z_out: Orthogonal transformation\n" \
    "    g: M-by-N feedback matrix\n" \
    "    count: Number of Y elements used\n" \
    "    info: 0=success, <0=invalid arg, 1=not controllable"

#define DOC_SB01FY "Inner denominator of right-coprime factorization (order 1 or 2).\n" \
    "\n" \
    "Computes F and V such that (A+B*F, B*V, F, V) is inner.\n" \
    "\n" \
    "Parameters:\n" \
    "  discr (bool): True for discrete-time\n" \
    "  a (ndarray): State matrix (n x n, n=1 or 2)\n" \
    "  b (ndarray): Input matrix (n x m)\n" \
    "\n" \
    "Returns:\n" \
    "  (f, v, info): Feedback matrix (m x n), V matrix (m x m), exit code"

#define DOC_SB01MD "Single-input pole assignment via state feedback.\n" \
    "\n" \
    "Determines the one-dimensional state feedback matrix G such that\n" \
    "the closed-loop system dX/dt = (A - B*G)*X has desired poles.\n" \
    "The system must be reduced to orthogonal canonical form via AB01MD first.\n" \
    "\n" \
    "Parameters:\n" \
    "  ncont (int): Controllable order from AB01MD (ncont >= 0)\n" \
    "  n (int): Order of matrix Z (n >= ncont)\n" \
    "  a (ndarray): NCONT-by-NCONT canonical form of A (F-order)\n" \
    "  b (ndarray): NCONT-element canonical form of B\n" \
    "  wr (ndarray): Real parts of desired poles (NCONT elements)\n" \
    "  wi (ndarray): Imaginary parts (complex conjugates consecutive)\n" \
    "  z (ndarray): N-by-N orthogonal transformation from AB01MD (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a_out, b_out, z_out, g, info):\n" \
    "    a_out: Schur form S of closed-loop (A-B*G)\n" \
    "    b_out: Transformed B (Z*B)\n" \
    "    z_out: Orthogonal Z reducing (A-B*G) to Schur form\n" \
    "    g: State feedback matrix (NCONT elements)\n" \
    "    info: 0=success, <0=parameter -info invalid"

#define DOC_SB02CX "Select purely imaginary eigenvalues for H-infinity norm.\n" \
    "\n" \
    "Callback function for DGEES eigenvalue selection. Returns True for\n" \
    "eigenvalues with |real part| < 100*eps, where eps is machine epsilon.\n" \
    "\n" \
    "Used internally by AB13CD for H-infinity norm computation.\n" \
    "\n" \
    "Parameters:\n" \
    "  reig (float): Real part of eigenvalue\n" \
    "  ieig (float): Imaginary part of eigenvalue (unused)\n" \
    "\n" \
    "Returns:\n" \
    "  bool: True if purely imaginary, False otherwise"

#define DOC_SB02MD "Solve continuous/discrete-time algebraic Riccati equation.\n" \
    "\n" \
    "Continuous-time (DICO='C'):\n" \
    "  Q + A'*X + X*A - X*G*X = 0\n" \
    "\n" \
    "Discrete-time (DICO='D'):\n" \
    "  X = A'*X*A - A'*X*B*(R + B'*X*B)^(-1)*B'*X*A + Q\n" \
    "\n" \
    "where G = B*R^(-1)*B' must be provided.\n" \
    "\n" \
    "Uses Laub's Schur vector approach.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' continuous-time, 'D' discrete-time\n" \
    "  hinv (str): 'D' symplectic, 'I' inverse symplectic (discrete-time)\n" \
    "  uplo (str): 'U' upper triangle, 'L' lower triangle\n" \
    "  scal (str): 'G' general scaling, 'N' no scaling\n" \
    "  sort (str): 'S' stable eigenvalues first, 'U' unstable first\n" \
    "  n (int): Order of A, G, Q (n >= 0)\n" \
    "  a (ndarray): Matrix A (n x n, F-order)\n" \
    "  g (ndarray): Symmetric matrix G = B*R^(-1)*B' (n x n, F-order)\n" \
    "  q (ndarray): Symmetric matrix Q (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  - X: Solution matrix (n x n)\n" \
    "  - rcond: Reciprocal condition number of U(1,1) block\n" \
    "  - wr: Real parts of eigenvalues (2n)\n" \
    "  - wi: Imaginary parts of eigenvalues (2n)\n" \
    "  - S: Schur form (2n x 2n)\n" \
    "  - U: Orthogonal transformation (2n x 2n)\n" \
    "  - info: Exit code"

#define DOC_SB02MR "Select unstable eigenvalues for continuous-time Riccati.\n" \
    "\n" \
    "Callback function for DGEES eigenvalue selection. Returns True for\n" \
    "eigenvalues with real part >= 0 (unstable in continuous-time).\n" \
    "\n" \
    "Used internally for continuous-time algebraic Riccati equation solvers.\n" \
    "\n" \
    "Parameters:\n" \
    "  reig (float): Real part of eigenvalue\n" \
    "  ieig (float): Imaginary part of eigenvalue (unused)\n" \
    "\n" \
    "Returns:\n" \
    "  bool: True if unstable (reig >= 0), False otherwise"

#define DOC_SB02MS "Select unstable eigenvalues for discrete-time Riccati.\n" \
    "\n" \
    "Callback function for DGEES eigenvalue selection. Returns True for\n" \
    "eigenvalues with modulus >= 1 (unstable in discrete-time).\n" \
    "\n" \
    "Used internally for discrete-time algebraic Riccati equation solvers.\n" \
    "\n" \
    "Parameters:\n" \
    "  reig (float): Real part of eigenvalue\n" \
    "  ieig (float): Imaginary part of eigenvalue\n" \
    "\n" \
    "Returns:\n" \
    "  bool: True if unstable (|lambda| >= 1), False otherwise"

#define DOC_SB02MT "Riccati preprocessing - convert coupling weight problems to standard form.\n" \
    "\n" \
    "Computes:\n" \
    "  G = B*R^(-1)*B'\n" \
    "  A_bar = A - B*R^(-1)*L'\n" \
    "  Q_bar = Q - L*R^(-1)*L'\n" \
    "\n" \
    "Parameters:\n" \
    "  jobg (str): 'G' to compute G, 'N' to skip\n" \
    "  jobl (str): 'Z' if L is zero, 'N' if L is nonzero\n" \
    "  fact (str): 'N' R unfactored, 'C' Cholesky, 'U' UdU'/LdL'\n" \
    "  uplo (str): 'U' upper triangle, 'L' lower triangle\n" \
    "  n (int): Order of A, Q, G (n >= 0)\n" \
    "  m (int): Order of R (m >= 0)\n" \
    "  a (ndarray or None): Matrix A (n x n, F-order) if jobl='N'\n" \
    "  b (ndarray): Matrix B (n x m, F-order)\n" \
    "  q (ndarray or None): Matrix Q (n x n, F-order) if jobl='N'\n" \
    "  r (ndarray): Matrix R (m x m, F-order)\n" \
    "  l (ndarray or None): Matrix L (n x m, F-order) if jobl='N'\n" \
    "  g (ndarray or None): Output G (n x n, F-order) if jobg='G'\n" \
    "\n" \
    "Returns:\n" \
    "  Variable outputs based on jobg and jobl:\n" \
    "  - If jobg='G', jobl='Z': (g, oufact, info)\n" \
    "  - If jobg='N', jobl='N': (a, b, q, l, oufact, info)\n" \
    "  - If jobg='G', jobl='N': (a, b, q, l, g, oufact, info)"

#define DOC_SB02MU "Construct Hamiltonian or symplectic matrix for Riccati equations.\n" \
    "\n" \
    "For continuous-time (DICO='C'), constructs Hamiltonian:\n" \
    "  S = [  A   -G ]\n" \
    "      [ -Q   -A']\n" \
    "\n" \
    "For discrete-time (DICO='D', HINV='D'), constructs symplectic:\n" \
    "  S = [  A^(-1)        A^(-1)*G     ]\n" \
    "      [ Q*A^(-1)   A' + Q*A^(-1)*G  ]\n" \
    "\n" \
    "For discrete-time (DICO='D', HINV='I'), constructs inverse symplectic:\n" \
    "  S = [ A + G*A^(-T)*Q   -G*A^(-T) ]\n" \
    "      [    -A^(-T)*Q       A^(-T)  ]\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' continuous-time, 'D' discrete-time\n" \
    "  hinv (str): 'D' or 'I' (only for discrete-time)\n" \
    "  uplo (str): 'U' upper triangle, 'L' lower triangle\n" \
    "  n (int): Order of A, G, Q (n >= 0)\n" \
    "  a (ndarray): Matrix A (n x n, F-order)\n" \
    "  g (ndarray): Symmetric matrix G (n x n, F-order)\n" \
    "  q (ndarray): Symmetric matrix Q (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  - S: Hamiltonian/symplectic matrix (2n x 2n)\n" \
    "  - rcond: Reciprocal condition number of A (discrete) or 1.0 (continuous)\n" \
    "  - info: Exit code (0=success)"

#define DOC_SB02MV "Select stable eigenvalues for continuous-time Riccati.\n" \
    "\n" \
    "Callback function for DGEES eigenvalue selection. Returns True for\n" \
    "eigenvalues with real part < 0 (stable in continuous-time).\n" \
    "\n" \
    "Used internally for continuous-time algebraic Riccati equation solvers.\n" \
    "\n" \
    "Parameters:\n" \
    "  reig (float): Real part of eigenvalue\n" \
    "  ieig (float): Imaginary part of eigenvalue (unused)\n" \
    "\n" \
    "Returns:\n" \
    "  bool: True if stable (reig < 0), False otherwise"

#define DOC_SB02MW "Select stable eigenvalues for discrete-time Riccati.\n" \
    "\n" \
    "Callback function for DGEES eigenvalue selection. Returns True for\n" \
    "eigenvalues with modulus < 1 (stable in discrete-time).\n" \
    "\n" \
    "Used internally for discrete-time algebraic Riccati equation solvers.\n" \
    "\n" \
    "Parameters:\n" \
    "  reig (float): Real part of eigenvalue\n" \
    "  ieig (float): Imaginary part of eigenvalue\n" \
    "\n" \
    "Returns:\n" \
    "  bool: True if stable (|lambda| < 1), False otherwise"

#define DOC_SB02MX "Extended Riccati preprocessing - convert coupling weight problems.\n" \
    "\n" \
    "Computes:\n" \
    "  G = B*R^(-1)*B'\n" \
    "  A_bar = A +/- op(B*R^(-1)*L')\n" \
    "  Q_bar = Q +/- L*R^(-1)*L'\n" \
    "\n" \
    "Extended version of SB02MT with TRANS, FLAG, and DEF parameters.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobg (str): 'G' to compute G, 'N' to skip\n" \
    "  jobl (str): 'Z' if L is zero, 'N' if L is nonzero\n" \
    "  fact (str): 'N' R unfactored, 'C' Cholesky, 'U' UdU'/LdL'\n" \
    "  uplo (str): 'U' upper triangle, 'L' lower triangle\n" \
    "  trans (str): 'N' op(W)=W, 'T'/'C' op(W)=W'\n" \
    "  flag (str): 'P' plus sign, 'M' minus sign\n" \
    "  def (str): 'D' R positive definite, 'I' R indefinite (for fact='N')\n" \
    "  n (int): Order of A, Q, G (n >= 0)\n" \
    "  m (int): Order of R (m >= 0)\n" \
    "  a (ndarray or None): Matrix A (n x n, F-order) if jobl='N'\n" \
    "  b (ndarray): Matrix B (n x m, F-order)\n" \
    "  q (ndarray or None): Matrix Q (n x n, F-order) if jobl='N'\n" \
    "  r (ndarray): Matrix R (m x m, F-order)\n" \
    "  l (ndarray or None): Matrix L (n x m, F-order) if jobl='N'\n" \
    "  g (ndarray or None): Output G (n x n, F-order) if jobg='G'\n" \
    "\n" \
    "Returns:\n" \
    "  Variable outputs based on jobg and jobl:\n" \
    "  - If jobg='G', jobl='Z': (g, oufact, info)\n" \
    "  - If jobg='N', jobl='N': (a, b, q, l, oufact, info)\n" \
    "  - If jobg='G', jobl='N': (a, b, q, l, g, oufact, info)"

#define DOC_SB02ND "Optimal state feedback matrix for optimal control problem.\n" \
    "\n" \
    "Computes:\n" \
    "  F = (R + B'XB)^(-1) (B'XA + L')  [discrete-time]\n" \
    "  F = R^(-1) (B'X + L')            [continuous-time]\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'D' for discrete-time, 'C' for continuous-time\n" \
    "  fact (str): 'N' R unfactored, 'D' R=D'D, 'C' Cholesky, 'U' UdU'/LdL' (continuous only)\n" \
    "  uplo (str): 'U' upper triangle, 'L' lower triangle\n" \
    "  jobl (str): 'Z' if L is zero, 'N' if L is nonzero\n" \
    "  n (int): Order of A, X (n >= 0)\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  p (int): Rows of D if fact='D' (p >= m for continuous)\n" \
    "  rnorm (float): 1-norm of R (required for fact='U' only)\n" \
    "  a (ndarray): State matrix A (n x n, F-order), discrete only\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  r (ndarray): Weighting matrix R (m x m or p x m, F-order)\n" \
    "  ipiv (ndarray): Pivot indices (m,), input for fact='U'\n" \
    "  l (ndarray): Cross weighting L (n x m, F-order) if jobl='N'\n" \
    "  x (ndarray): Riccati solution X (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (f, r_out, x_out, oufact, rcond, info):\n" \
    "  - f: Optimal feedback matrix F (m x n)\n" \
    "  - r_out: Factored R or R+B'XB\n" \
    "  - x_out: Possibly modified X (for discrete with factored R)\n" \
    "  - oufact: Array [fact_type, x_fact_type]\n" \
    "  - rcond: Reciprocal condition number\n" \
    "  - info: Exit code (0=success, m+1=singular)"

#define DOC_SB02OD "Solve continuous- or discrete-time algebraic Riccati equations.\n" \
    "\n" \
    "Solves for X either:\n" \
    "  Continuous: Q + A'X + XA - (L+XB)R^(-1)(L+XB)' = 0\n" \
    "  Discrete:   X = A'XA - (L+A'XB)(R + B'XB)^(-1)(L+A'XB)' + Q\n" \
    "\n" \
    "Uses deflating subspaces with eigenvalue reordering.\n" \
    "Suited for ill-conditioned R (uses internal scaling).\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' continuous-time, 'D' discrete-time\n" \
    "  jobb (str): 'B' matrices B,R given, 'G' matrix G given\n" \
    "  fact (str): 'N' Q,R given, 'C' Q=C'C, 'D' R=D'D, 'B' both factored\n" \
    "  uplo (str): 'U' upper triangle, 'L' lower triangle\n" \
    "  jobl (str): 'Z' L is zero, 'N' L is nonzero\n" \
    "  sort (str): 'S' stable eigenvalues first, 'U' unstable first\n" \
    "  n (int): State dimension (>= 0)\n" \
    "  m (int): Number of inputs (>= 0, for JOBB='B')\n" \
    "  p (int): Number of outputs (for FACT='C','D','B')\n" \
    "  a (ndarray): N-by-N state matrix A (F-order)\n" \
    "  b (ndarray): N-by-M input matrix B or N-by-N matrix G (F-order)\n" \
    "  q (ndarray): N-by-N symmetric Q or P-by-N output matrix C (F-order)\n" \
    "  r (ndarray or None): M-by-M symmetric R or P-by-M matrix D (F-order)\n" \
    "  l (ndarray or None): N-by-M cross-weighting L (F-order)\n" \
    "  tol (float): Tolerance for singularity test (<=0 uses eps)\n" \
    "\n" \
    "Returns:\n" \
    "  (x, rcond, alfar, alfai, beta, s, t, u, info):\n" \
    "    x: N-by-N solution matrix\n" \
    "    rcond: Reciprocal condition number\n" \
    "    alfar, alfai, beta: Generalized eigenvalues (lambda = (alfar+j*alfai)/beta)\n" \
    "    s, t: 2N-by-2N Schur matrices\n" \
    "    u: 2N-by-2N transformation matrix\n" \
    "    info: 0=success, 1-6 various errors"

#define DOC_SB02OU "Select unstable generalized eigenvalues for continuous-time Riccati.\n" \
    "\n" \
    "Callback function for DGGES eigenvalue selection. Returns True for\n" \
    "generalized eigenvalues with positive real part (unstable in continuous-time).\n" \
    "\n" \
    "Selection criterion: (ALPHAR < 0 AND BETA < 0) OR (ALPHAR > 0 AND BETA > 0)\n" \
    "This is equivalent to: Re(lambda) = ALPHAR/BETA > 0\n" \
    "\n" \
    "Used internally by SB02OD for continuous-time algebraic Riccati equation solver.\n" \
    "\n" \
    "Parameters:\n" \
    "  alphar (float): Real part of eigenvalue numerator\n" \
    "  alphai (float): Imaginary part of eigenvalue numerator (unused)\n" \
    "  beta (float): Eigenvalue denominator\n" \
    "\n" \
    "Returns:\n" \
    "  bool: True if unstable (Re(lambda) > 0), False otherwise"

#define DOC_SB02OV "Select unstable generalized eigenvalues for discrete-time Riccati.\n" \
    "\n" \
    "Callback function for DGGES eigenvalue selection. Returns True for\n" \
    "generalized eigenvalues with modulus >= 1 (unstable in discrete-time).\n" \
    "\n" \
    "Selection criterion: sqrt(ALPHAR^2 + ALPHAI^2) >= abs(BETA)\n" \
    "This is equivalent to: |lambda| = |ALPHAR + i*ALPHAI|/|BETA| >= 1\n" \
    "\n" \
    "Used internally by discrete-time algebraic Riccati equation solvers.\n" \
    "\n" \
    "Parameters:\n" \
    "  alphar (float): Real part of eigenvalue numerator\n" \
    "  alphai (float): Imaginary part of eigenvalue numerator\n" \
    "  beta (float): Eigenvalue denominator\n" \
    "\n" \
    "Returns:\n" \
    "  bool: True if unstable (|lambda| >= 1), False otherwise"

#define DOC_SB02OW "Select stable generalized eigenvalues for continuous-time Riccati.\n" \
    "\n" \
    "Callback function for DGGES eigenvalue selection. Returns True for\n" \
    "generalized eigenvalues with negative real part (stable in continuous-time).\n" \
    "\n" \
    "Selection criterion: (ALPHAR < 0 AND BETA > 0) OR (ALPHAR > 0 AND BETA < 0)\n" \
    "This is equivalent to: Re(lambda) = ALPHAR/BETA < 0\n" \
    "\n" \
    "Used internally by SB02OD for continuous-time algebraic Riccati equation solver.\n" \
    "\n" \
    "Parameters:\n" \
    "  alphar (float): Real part of eigenvalue numerator\n" \
    "  alphai (float): Imaginary part of eigenvalue numerator (unused)\n" \
    "  beta (float): Eigenvalue denominator\n" \
    "\n" \
    "Returns:\n" \
    "  bool: True if stable (Re(lambda) < 0), False otherwise"

#define DOC_SB02OX "Select stable generalized eigenvalues for discrete-time Riccati.\n" \
    "\n" \
    "Callback function for DGGES eigenvalue selection. Returns True for\n" \
    "generalized eigenvalues with modulus less than 1 (stable in discrete-time).\n" \
    "\n" \
    "Selection criterion: sqrt(ALPHAR^2 + ALPHAI^2) < abs(BETA)\n" \
    "This is equivalent to: |lambda| = |ALPHAR + i*ALPHAI|/|BETA| < 1\n" \
    "\n" \
    "Used internally by discrete-time algebraic Riccati equation solvers.\n" \
    "\n" \
    "Parameters:\n" \
    "  alphar (float): Real part of eigenvalue numerator\n" \
    "  alphai (float): Imaginary part of eigenvalue numerator\n" \
    "  beta (float): Eigenvalue denominator\n" \
    "\n" \
    "Returns:\n" \
    "  bool: True if stable (|lambda| < 1), False otherwise"

#define DOC_SB02OY "Construct extended Hamiltonian/symplectic matrix pairs for Riccati equations.\n" \
    "\n" \
    "Constructs extended matrix pairs for algebraic Riccati equations in optimal\n" \
    "control and spectral factorization problems, then compresses to 2N-by-2N.\n" \
    "\n" \
    "Parameters:\n" \
    "  type_str (str): 'O' optimal control, 'S' spectral factorization\n" \
    "  dico (str): 'C' continuous-time, 'D' discrete-time\n" \
    "  jobb (str): 'B' matrices B,R given, 'G' matrix G given\n" \
    "  fact (str): 'N' Q,R given, 'C' Q=C'C, 'D' R=D'D, 'B' both factored\n" \
    "  uplo (str): 'U' upper triangle, 'L' lower triangle\n" \
    "  jobl (str): 'Z' L is zero, 'N' L is nonzero\n" \
    "  jobe (str): 'I' E is identity, 'N' E is general\n" \
    "  n (int): Order of A, Q, E (>= 0)\n" \
    "  m (int): Order of R, columns of B (>= 0, for JOBB='B')\n" \
    "  p (int): Rows of C/D (for FACT='C','D','B' or TYPE='S')\n" \
    "  a (ndarray): N-by-N state matrix A (F-order)\n" \
    "  b (ndarray): N-by-M input matrix B or N-by-N matrix G (F-order)\n" \
    "  q (ndarray): N-by-N symmetric Q or P-by-N output matrix C (F-order)\n" \
    "  r (ndarray or None): M-by-M symmetric R or P-by-M matrix D (F-order)\n" \
    "  l (ndarray or None): N-by-M cross-weighting L (F-order)\n" \
    "  e (ndarray): N-by-N descriptor E (F-order)\n" \
    "  tol (float): Tolerance for singularity test (<=0 uses eps)\n" \
    "\n" \
    "Returns:\n" \
    "  For JOBB='B': (af, bf, rcond, info) - 2N-by-2N matrices, condition number\n" \
    "  For JOBB='G', DICO='C', JOBE='I': (af, info) - 2N-by-2N matrix only\n" \
    "  For JOBB='G' otherwise: (af, bf, info) - 2N-by-2N matrices"

#define DOC_SB02PD "Continuous-time algebraic Riccati equation solver using matrix sign function.\n" \
    "\n" \
    "Solves the real continuous-time matrix algebraic Riccati equation:\n" \
    "    op(A)'*X + X*op(A) + Q - X*G*X = 0\n" \
    "where op(A) = A or A', G and Q are symmetric.\n" \
    "\n" \
    "Optionally computes error bound and condition estimate.\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): N-by-N coefficient matrix A (F-order)\n" \
    "  g (ndarray): N-by-N symmetric matrix G (F-order)\n" \
    "  q (ndarray): N-by-N symmetric matrix Q (F-order)\n" \
    "  job (str): 'X' solution only (default), 'A' all (solution, rcond, ferr)\n" \
    "  trana (str): 'N' op(A)=A (default), 'T'/'C' op(A)=A'\n" \
    "  uplo (str): 'U' upper triangle (default), 'L' lower triangle of G, Q\n" \
    "\n" \
    "Returns:\n" \
    "  (X, rcond, ferr, wr, wi, info):\n" \
    "  - X: N-by-N symmetric solution matrix\n" \
    "  - rcond: Reciprocal condition number (if job='A')\n" \
    "  - ferr: Forward error bound (if job='A')\n" \
    "  - wr, wi: Real and imaginary parts of closed-loop eigenvalues (if job='A')\n" \
    "  - info: 0=success, 1=imaginary eigenvalues, 2=no convergence (approx computed),\n" \
    "          3=singular system, 4=Schur reduction failed"

#define DOC_SB02QD "Estimate conditioning and error bound for continuous-time Riccati equation.\n" \
    "\n" \
    "Estimates conditioning and computes forward error bound for:\n" \
    "    op(A)'*X + X*op(A) + Q - X*G*X = 0\n" \
    "where op(A) = A or A', Q and G are symmetric.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'C' condition only, 'E' error only, 'B' both\n" \
    "  fact (str): 'N' compute Schur, 'F' Schur factorization provided\n" \
    "  trana (str): 'N' for op(A)=A, 'T'/'C' for op(A)=A'\n" \
    "  uplo (str): 'U' for upper, 'L' for lower triangular Q, G\n" \
    "  lyapun (str): 'O' for original, 'R' for reduced Lyapunov\n" \
    "  a (ndarray): N-by-N matrix A (F-order)\n" \
    "  t (ndarray): N-by-N Schur matrix T (F-order, input if fact='F', output if 'N')\n" \
    "  u (ndarray): N-by-N orthogonal matrix U (F-order, input/output)\n" \
    "  g (ndarray): N-by-N symmetric matrix G (F-order)\n" \
    "  q (ndarray): N-by-N symmetric matrix Q (F-order)\n" \
    "  x (ndarray): N-by-N symmetric solution X (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (sep, rcond, ferr, t, u, info):\n" \
    "  - sep: Estimated separation sep(op(Ac), -op(Ac)')\n" \
    "  - rcond: Reciprocal condition number estimate\n" \
    "  - ferr: Forward error bound estimate\n" \
    "  - t: Updated Schur form T\n" \
    "  - u: Updated orthogonal matrix U\n" \
    "  - info: Exit code (0=success, >0=algorithm error)"

#define DOC_SB02RD "Solve algebraic Riccati equation using Schur vector method.\n" \
    "\n" \
    "Solves continuous-time Riccati equation:\n" \
    "  Q + A'*X + X*A - X*G*X = 0       (DICO='C')\n" \
    "or discrete-time Riccati equation:\n" \
    "  Q + A'*X*(I + G*X)^(-1)*A - X = 0  (DICO='D')\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'X' solution, 'C' condition, 'E' error bound, 'A' all\n" \
    "  dico (str): 'C' continuous-time, 'D' discrete-time\n" \
    "  hinv (str): 'D' direct, 'I' inverse (for discrete-time)\n" \
    "  trana (str): 'N' no transpose, 'T'/'C' transpose of A\n" \
    "  uplo (str): 'U' upper triangle, 'L' lower triangle\n" \
    "  scal (str): 'G' general scaling, 'N' no scaling\n" \
    "  sort (str): 'S' stable eigenvalues first, 'U' unstable first\n" \
    "  fact (str): 'N' compute Schur, 'F' Schur factorization provided\n" \
    "  lyapun (str): 'O' original equations, 'R' reduced equations\n" \
    "  A (ndarray): State matrix A (n x n, F-order)\n" \
    "  Q (ndarray): Symmetric matrix Q (n x n, F-order)\n" \
    "  G (ndarray): Symmetric matrix G = B*R^(-1)*B' (n x n, F-order)\n" \
    "  T (ndarray, optional): Schur form matrix (n x n, F-order)\n" \
    "  V (ndarray, optional): Schur vectors (n x n, F-order)\n" \
    "  X (ndarray, optional): Solution matrix input for JOB='C'/'E' (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (X, sep, rcond, ferr, wr, wi, S, info):\n" \
    "  - X: Solution matrix X (n x n)\n" \
    "  - sep: Separation or scaling factor\n" \
    "  - rcond: Reciprocal condition number\n" \
    "  - ferr: Forward error bound\n" \
    "  - wr, wi: Real and imaginary parts of eigenvalues (2*n,)\n" \
    "  - S: Ordered Schur form (2*n x 2*n)\n" \
    "  - info: Exit code (0=success, >0=error)"

#define DOC_SB02RU "Construct Hamiltonian or symplectic matrix for Riccati equation.\n" \
    "\n" \
    "For continuous-time (DICO='C'), constructs Hamiltonian matrix:\n" \
    "    S = [ op(A)   -G    ]\n" \
    "        [  -Q   -op(A)' ]\n" \
    "\n" \
    "For discrete-time (DICO='D'), constructs symplectic matrix.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' continuous-time, 'D' discrete-time\n" \
    "  hinv (str): 'D' or 'I' for discrete formula variant\n" \
    "  trana (str): 'N' op(A)=A, 'T'/'C' op(A)=A'\n" \
    "  uplo (str): 'U' upper tri stored, 'L' lower tri stored\n" \
    "  a (ndarray): N-by-N state matrix A (F-order)\n" \
    "  g (ndarray): N-by-N symmetric G matrix (F-order)\n" \
    "  q (ndarray): N-by-N symmetric Q matrix (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (s, rcond, pivotg, info): 2N-by-2N Hamiltonian/symplectic matrix,\n" \
    "                           condition number (discrete only), pivot growth,\n" \
    "                           exit code (0=success)"

#define DOC_SB02SD "Condition and error bound for discrete-time Riccati equation.\n" \
    "\n" \
    "Estimates conditioning and computes error bound for:\n" \
    "    X = op(A)'*X*(I_n + G*X)^-1*op(A) + Q\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'C' for rcond only, 'E' for ferr only, 'B' for both\n" \
    "  fact (str): 'N' to compute Schur, 'F' if Schur supplied\n" \
    "  trana (str): 'N' for op(A)=A, 'T'/'C' for op(A)=A'\n" \
    "  uplo (str): 'U' for upper, 'L' for lower triangular Q, G\n" \
    "  lyapun (str): 'O' for original, 'R' for reduced Lyapunov\n" \
    "  n (int): Order of matrices\n" \
    "  a (ndarray): N-by-N matrix A (F-order)\n" \
    "  t (ndarray): N-by-N Schur matrix T (F-order)\n" \
    "  u (ndarray): N-by-N orthogonal matrix U (F-order)\n" \
    "  g (ndarray): N-by-N symmetric matrix G (F-order)\n" \
    "  q (ndarray): N-by-N symmetric matrix Q (F-order)\n" \
    "  x (ndarray): N-by-N symmetric solution X (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (t, u, sepd, rcond, ferr, info): Updated T, U, separation, rcond, error, exit code"

#define DOC_SB03MD "Solve continuous or discrete Lyapunov equation.\n" \
    "\n" \
    "Solves op(A)'*X + X*op(A) = scale*C (continuous) or\n" \
    "op(A)'*X*op(A) - X = scale*C (discrete).\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous, 'D' for discrete\n" \
    "  job (str): 'X' solution, 'S' separation, 'B' both\n" \
    "  fact (str): 'N' compute Schur, 'F' provided\n" \
    "  trana (str): 'N' for A, 'T' for A'\n" \
    "  n (int): Order of matrices\n" \
    "  a (ndarray): Matrix A (n x n, F-order)\n" \
    "  c (ndarray): Symmetric RHS C (n x n, F-order)\n" \
    "  u (ndarray, optional): Schur transformation if FACT='F'\n" \
    "\n" \
    "Returns:\n" \
    "  (x, a, u, wr, wi, scale, sep, ferr, info):\n" \
    "    Solution, Schur form, transformation, eigenvalues, scale, info"

#define DOC_SB03MU "Solve small discrete-time Sylvester equation.\n" \
    "\n" \
    "Solves ISGN*op(TL)*X*op(TR) - X = SCALE*B where TL is N1-by-N1,\n" \
    "TR is N2-by-N2, N1,N2 in {0,1,2}, ISGN = 1 or -1.\n" \
    "\n" \
    "Parameters:\n" \
    "  ltranl (bool): If True, op(TL)=TL', else op(TL)=TL\n" \
    "  ltranr (bool): If True, op(TR)=TR', else op(TR)=TR\n" \
    "  isgn (int): Sign of equation (1 or -1)\n" \
    "  tl (ndarray): N1-by-N1 matrix TL (F-order)\n" \
    "  tr (ndarray): N2-by-N2 matrix TR (F-order)\n" \
    "  b (ndarray): N1-by-N2 RHS matrix B (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (X, scale, xnorm, info): Solution X, scale factor, inf-norm, exit code"

#define DOC_SB03MV "Solve 2x2 discrete-time Lyapunov equation.\n" \
    "\n" \
    "Solves op(T)'*X*op(T) - X = SCALE*B for symmetric X.\n" \
    "\n" \
    "Parameters:\n" \
    "  t (array): 2x2 matrix T\n" \
    "  b (array): 2x2 symmetric matrix B\n" \
    "  ltran (bool): If True, op(T)=T', else op(T)=T. Default: False\n" \
    "  lupper (bool): If True, upper triangle. Default: True\n" \
    "\n" \
    "Returns:\n" \
    "  (X, scale, xnorm, info): Solution X, scale factor, inf-norm, exit code"

#define DOC_SB03MW "Solve 2x2 continuous-time Lyapunov equation.\n" \
    "\n" \
    "Solves op(T)'*X + X*op(T) = SCALE*B for symmetric X.\n" \
    "\n" \
    "Parameters:\n" \
    "  t (array): 2x2 matrix T\n" \
    "  b (array): 2x2 symmetric matrix B\n" \
    "  ltran (bool): If True, op(T)=T', else op(T)=T. Default: False\n" \
    "  lupper (bool): If True, upper triangle. Default: True\n" \
    "\n" \
    "Returns:\n" \
    "  (X, scale, xnorm, info): Solution X, scale factor, inf-norm, exit code"

#define DOC_SB03MX "Solve discrete-time Lyapunov for quasi-triangular A.\n" \
    "\n" \
    "Solves op(A)'*X*op(A) - X = scale*C where A is Schur form.\n" \
    "\n" \
    "Parameters:\n" \
    "  trana (str): 'N' for op(A)=A, 'T'/'C' for op(A)=A'\n" \
    "  a (ndarray): Upper quasi-triangular matrix (n x n, F-order)\n" \
    "  c (ndarray): Symmetric RHS C, overwritten with solution X (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (X, scale, info): Solution X, scale factor, exit code"

#define DOC_SB03MY "Solve continuous-time Lyapunov for quasi-triangular A.\n" \
    "\n" \
    "Solves op(A)'*X + X*op(A) = scale*C where A is Schur form.\n" \
    "\n" \
    "Parameters:\n" \
    "  trana (str): 'N' for op(A)=A, 'T'/'C' for op(A)=A'\n" \
    "  a (ndarray): Upper quasi-triangular matrix (n x n, F-order)\n" \
    "  c (ndarray): Symmetric RHS C, overwritten with solution X (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (X, scale, info): Solution X, scale factor, exit code"

#define DOC_SB03OD "Solve stable Lyapunov equation for Cholesky factor.\n" \
    "\n" \
    "Solves either:\n" \
    "  A'*X + X*A = -scale²*B'*B   (continuous-time)\n" \
    "  A'*X*A - X = -scale²*B'*B   (discrete-time)\n" \
    "where X = U'*U (Cholesky factorization)\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  fact (str): 'N' compute Schur, 'F' Schur factorization provided\n" \
    "  trans (str): 'N' for no transpose, 'T' for transpose of A,B\n" \
    "  a (ndarray): State matrix A (n x n, F-order) or Schur factor S\n" \
    "  b (ndarray): Right-hand side B (m x n, F-order) if trans='N'\n" \
    "               or (n x m, F-order) if trans='T'\n" \
    "  q (ndarray): Schur vectors Q (n x n, F-order) if fact='F'\n" \
    "\n" \
    "Returns:\n" \
    "  (u, q_out, wr, wi, scale, info):\n" \
    "  - u: Upper triangular Cholesky factor U (n x n)\n" \
    "  - q_out: Orthogonal Schur vectors (if computed)\n" \
    "  - wr: Real parts of eigenvalues (n,)\n" \
    "  - wi: Imaginary parts of eigenvalues (n,)\n" \
    "  - scale: Scale factor (scale <= 1)\n" \
    "  - info: Exit code (0=success, 1=nearly singular, 2/3=unstable)"

#define DOC_SB03OR "Solve real quasi-triangular Sylvester equation.\n" \
    "\n" \
    "Solves for N-by-M matrix X (M = 1 or 2):\n" \
    "  Continuous: op(S)'*X + X*op(A) = scale*C\n" \
    "  Discrete:   op(S)'*X*op(A) - X = scale*C\n" \
    "\n" \
    "where S is N-by-N block upper triangular (real Schur form), A is M-by-M.\n" \
    "Solution X overwrites C.\n" \
    "\n" \
    "Parameters:\n" \
    "  discr (bool): True for discrete-time, False for continuous-time\n" \
    "  ltrans (bool): True for op(K) = K', False for op(K) = K\n" \
    "  n (int): Order of matrix S (>= 0)\n" \
    "  m (int): Order of matrix A (1 or 2)\n" \
    "  s (ndarray): N-by-N block upper triangular matrix (F-order)\n" \
    "  a (ndarray): M-by-M matrix (F-order)\n" \
    "  c (ndarray): N-by-M RHS matrix, overwritten with solution (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (scale, info):\n" \
    "    scale: Scale factor (0 < scale <= 1)\n" \
    "    info: 0=success, 1=perturbed to avoid overflow"

#define DOC_SB03OS "Complex triangular Lyapunov solver for Cholesky factor.\n" \
    "\n" \
    "Solves for X = op(U)^H * op(U):\n" \
    "  Continuous: op(S)^H*X + X*op(S) = -scale^2*op(R)^H*op(R)\n" \
    "  Discrete:   op(S)^H*X*op(S) - X = -scale^2*op(R)^H*op(R)\n" \
    "\n" \
    "where S and R are complex N-by-N upper triangular matrices.\n" \
    "\n" \
    "Parameters:\n" \
    "  discr (bool): True for discrete-time, False for continuous-time\n" \
    "  ltrans (bool): True for op(K) = K^H, False for op(K) = K\n" \
    "  n (int): Order of matrices (>= 0)\n" \
    "  s (ndarray): Complex N-by-N upper triangular S, modified on exit (F-order)\n" \
    "  r (ndarray): Complex N-by-N upper triangular R, overwritten with U (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (scale, info):\n" \
    "    scale: Scale factor (0 < scale <= 1)\n" \
    "    info: 0=success, 1=S not stable/convergent"

#define DOC_SB03OT "Solve reduced Lyapunov equation for triangular factors.\n" \
    "\n" \
    "Solves for X = op(U)'*op(U):\n" \
    "  Continuous: op(S)'*X + X*op(S) = -scale^2*op(R)'*op(R)\n" \
    "  Discrete:   op(S)'*X*op(S) - X = -scale^2*op(R)'*op(R)\n" \
    "\n" \
    "where S is block upper triangular (real Schur form), R is upper triangular.\n" \
    "Solution U overwrites R.\n" \
    "\n" \
    "Parameters:\n" \
    "  discr (bool): True for discrete-time, False for continuous-time\n" \
    "  ltrans (bool): True for op(K) = K', False for op(K) = K\n" \
    "  n (int): Order of matrices (>= 0)\n" \
    "  s (ndarray): N-by-N block upper triangular S, modified on exit (F-order)\n" \
    "  r (ndarray): N-by-N upper triangular R, overwritten with U (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (scale, info):\n" \
    "    scale: Scale factor (0 < scale <= 1)\n" \
    "    info: 0=success, 1=S not stable/convergent, 2=singular equation"

#define DOC_SB03OU "Solve Lyapunov equation for Cholesky factor.\n" \
    "\n" \
    "Computes upper triangular matrix U such that X = op(U)'*op(U) satisfies:\n" \
    "  Continuous: op(A)'*X + X*op(A) = -scale^2 * op(B)'*op(B)\n" \
    "  Discrete:   op(A)'*X*op(A) - X = -scale^2 * op(B)'*op(B)\n" \
    "\n" \
    "Parameters:\n" \
    "  discr (bool): If True, solve discrete equation\n" \
    "  ltrans (bool): If True, op(K)=K'; otherwise op(K)=K\n" \
    "  a (ndarray): N-by-N upper quasi-triangular matrix A in Schur form\n" \
    "  b (ndarray): M-by-N (or N-by-M if ltrans) matrix B\n" \
    "\n" \
    "Returns:\n" \
    "  (u, scale, info): Upper triangular Cholesky factor U, scale factor, exit code"

#define DOC_SB03OV "Construct complex plane rotation for Lyapunov solver.\n" \
    "\n" \
    "Parameters:\n" \
    "  a_re (float): Real part of complex number a\n" \
    "  a_im (float): Imaginary part of complex number a\n" \
    "  b (float): Real number b\n" \
    "  small (float): Threshold for unit matrix\n" \
    "\n" \
    "Returns:\n" \
    "  (d, c_re, c_im, s, info): d, complex cosine, real sine, exit code"

#define DOC_SB03OY "Solve 2x2 Lyapunov equation for Cholesky factor.\n" \
    "\n" \
    "Solves for Cholesky factor U of X where op(U)'*op(U) = X:\n" \
    "  Continuous: op(S)'*X + X*op(S) = -ISGN*scale^2*op(R)'*op(R)\n" \
    "  Discrete:   op(S)'*X*op(S) - X = -ISGN*scale^2*op(R)'*op(R)\n" \
    "\n" \
    "where S is 2x2 with complex conjugate eigenvalues, R is 2x2 upper triangular.\n" \
    "\n" \
    "Parameters:\n" \
    "  discr (bool): True for discrete-time, False for continuous-time\n" \
    "  ltrans (bool): True for op(K) = K', False for op(K) = K\n" \
    "  isgn (int): Sign (+1 or -1)\n" \
    "  s (ndarray): 2x2 matrix S, modified on exit (F-order)\n" \
    "  r (ndarray): 2x2 upper triangular matrix R (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, scale, info):\n" \
    "    a: 2x2 Cholesky factor U (upper triangular)\n" \
    "    scale: Scale factor (0 < scale <= 1)\n" \
    "    info: 0=success, 2=not stable/convergent, 3=singular, 4=real eigenvalues"

#define DOC_SB03OZ "Complex Lyapunov equation solver for Cholesky factor.\n" \
    "\n" \
    "Solves for X = op(U)^H * op(U):\n" \
    "  Continuous: op(A)^H*X + X*op(A) = -scale^2*op(B)^H*op(B)\n" \
    "  Discrete:   op(A)^H*X*op(A) - X = -scale^2*op(B)^H*op(B)\n" \
    "\n" \
    "where A is N-by-N complex, op(B) is M-by-N complex.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' continuous-time, 'D' discrete-time\n" \
    "  fact (str): 'N' compute Schur form, 'F' Schur form supplied\n" \
    "  trans (str): 'N' for op(K)=K, 'C' for op(K)=K^H\n" \
    "  n (int): Order of matrix A (>= 0)\n" \
    "  m (int): Rows of op(B) (>= 0)\n" \
    "  a (ndarray): Complex N-by-N matrix (F-order)\n" \
    "  q (ndarray): Complex N-by-N Schur vectors (F-order)\n" \
    "  b (ndarray): Complex M-by-N matrix (F-order)\n" \
    "  lzwork (int, optional): Complex workspace size (-1 for query)\n" \
    "\n" \
    "Returns:\n" \
    "  (scale, w, info):\n" \
    "    scale: Scale factor (0 < scale <= 1)\n" \
    "    w: Complex eigenvalues of A (length N)\n" \
    "    info: 0=success, 1=not stable/convergent, 2=Schur failed"

#define DOC_SB03PD "Solve discrete-time Lyapunov equation with separation estimation.\n" \
    "\n" \
    "Solves op(A)'*X*op(A) - X = scale*C and/or estimates separation.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'X' solution only, 'S' separation only, 'B' both\n" \
    "  fact (str): 'N' compute Schur, 'F' Schur provided\n" \
    "  trana (str): 'N' for op(A)=A, 'T'/'C' for op(A)=A'\n" \
    "  a (ndarray): N-by-N matrix A (F-order)\n" \
    "  c (ndarray): N-by-N symmetric RHS C (F-order)\n" \
    "  u (ndarray): N-by-N orthogonal U if fact='F' (optional)\n" \
    "\n" \
    "Returns:\n" \
    "  For job='X': (a, u, x, scale, wr, wi, info)\n" \
    "  For job='S': (a, u, None, scale, sepd, wr, wi, info)\n" \
    "  For job='B': (a, u, x, scale, sepd, ferr, wr, wi, info)"

#define DOC_SB03QD "Estimate conditioning and forward error bound for continuous-time Lyapunov.\n" \
    "\n" \
    "Estimates the conditioning and computes an error bound on the solution of:\n" \
    "  op(A)' * X + X * op(A) = scale * C\n" \
    "where op(A) = A or A^T, C is symmetric.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'C' condition only, 'E' error only, 'B' both\n" \
    "  fact (str): 'F' Schur form provided, 'N' compute Schur\n" \
    "  trana (str): 'N' for op(A)=A, 'T'/'C' for op(A)=A^T\n" \
    "  uplo (str): 'U' upper triangle of C, 'L' lower triangle\n" \
    "  lyapun (str): 'O' original equations, 'R' reduced equations\n" \
    "  scale (float): Scale factor from Lyapunov solver (0 <= scale <= 1)\n" \
    "  a (ndarray): N-by-N matrix A (F-order)\n" \
    "  t (ndarray): N-by-N Schur form matrix T (F-order)\n" \
    "  u (ndarray): N-by-N orthogonal Schur factor U (F-order)\n" \
    "  c (ndarray): N-by-N symmetric matrix C (F-order)\n" \
    "  x (ndarray): N-by-N symmetric solution matrix X (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (sep, rcond, ferr, t_out, u_out, info):\n" \
    "  sep: Separation estimate sep(op(A),-op(A)')\n" \
    "  rcond: Reciprocal condition number estimate\n" \
    "  ferr: Forward error bound\n" \
    "  t_out: Schur form of A\n" \
    "  u_out: Orthogonal transformation matrix\n" \
    "  info: 0=success, 1..n=DGEES fail, n+1=near-common eigenvalues"

#define DOC_SB03QX "Estimate forward error bound for continuous-time Lyapunov equation.\n" \
    "\n" \
    "Estimates forward error bound for solution X of op(A)'*X + X*op(A) = C.\n" \
    "\n" \
    "Parameters:\n" \
    "  trana (str): 'N' for op(A)=A, 'T'/'C' for op(A)=A'\n" \
    "  uplo (str): 'U' for upper, 'L' for lower triangle of R\n" \
    "  lyapun (str): 'O' for original (use U), 'R' for reduced form\n" \
    "  n (int): Order of matrices\n" \
    "  xanorm (float): Max-norm of solution matrix X\n" \
    "  t (ndarray): Schur form matrix T (n x n, F-order)\n" \
    "  u (ndarray): Orthogonal transformation U (n x n, F-order)\n" \
    "  r (ndarray): Absolute residual matrix R (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (ferr, r, info): Forward error bound, symmetrized R, exit code"

#define DOC_SB03QY "Estimate separation and 1-norm of Theta for continuous-time Lyapunov equation.\n" \
    "\n" \
    "Estimates sep(op(A), -op(A)') = min norm(op(A)'*X + X*op(A))/norm(X)\n" \
    "and/or 1-norm of Theta operator for op(A)'*X + X*op(A) = C.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'S' for separation, 'T' for Theta norm, 'B' for both\n" \
    "  trana (str): 'N' for op(A)=A, 'T'/'C' for op(A)=A'\n" \
    "  lyapun (str): 'O' for original (use U), 'R' for reduced form\n" \
    "  t (ndarray): Schur form matrix T (n x n, F-order)\n" \
    "  u (ndarray): Orthogonal transformation U (n x n, F-order)\n" \
    "  x (ndarray): Lyapunov solution matrix X (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (sep, thnorm, info): Separation, Theta norm, exit code (N+1=close eigenvalues)"

#define DOC_SB03RD "Solve continuous-time Lyapunov equation with separation estimation.\n" \
    "\n" \
    "Solves: op(A)' * X + X * op(A) = scale * C\n" \
    "where op(A) = A or A^T, C is symmetric.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'X' solution only, 'S' separation only, 'B' both\n" \
    "  fact (str): 'F' Schur form provided, 'N' compute Schur\n" \
    "  trana (str): 'N' for op(A)=A, 'T'/'C' for op(A)=A^T\n" \
    "  a (ndarray): N-by-N matrix A (F-order)\n" \
    "  c (ndarray): N-by-N symmetric matrix C (F-order)\n" \
    "  u (ndarray, optional): N-by-N orthogonal matrix (if fact='F')\n" \
    "\n" \
    "Returns:\n" \
    "  (a_out, u, x, scale, sep, ferr, wr, wi, info):\n" \
    "  a_out: Schur form of A\n" \
    "  u: Orthogonal Schur factor\n" \
    "  x: Solution matrix\n" \
    "  scale: Scale factor (0 < scale <= 1)\n" \
    "  sep: Separation estimate (job='S' or 'B')\n" \
    "  ferr: Forward error bound (job='B')\n" \
    "  wr, wi: Real/imaginary eigenvalue parts (fact='N')\n" \
    "  info: 0=success, -i=arg i invalid, 1..n=DGEES fail, n+1=singular"

#define DOC_SB03SD "Estimate conditioning and forward error for discrete-time Lyapunov.\n" \
    "\n" \
    "Estimates the conditioning and computes an error bound on the solution of:\n" \
    "  op(A)' * X * op(A) - X = scale * C\n" \
    "where op(A) = A or A^T, C is symmetric.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'C' condition only, 'E' error only, 'B' both\n" \
    "  fact (str): 'F' Schur form provided, 'N' compute Schur\n" \
    "  trana (str): 'N' for op(A)=A, 'T'/'C' for op(A)=A^T\n" \
    "  uplo (str): 'U' upper triangle of C, 'L' lower triangle\n" \
    "  lyapun (str): 'O' original equations, 'R' reduced equations\n" \
    "  n (int): Order of matrices\n" \
    "  scale (float): Scale factor from Lyapunov solver (0 <= scale <= 1)\n" \
    "  a (ndarray): N-by-N matrix A (F-order)\n" \
    "  t (ndarray): N-by-N Schur form matrix T (F-order)\n" \
    "  u (ndarray): N-by-N orthogonal Schur factor U (F-order)\n" \
    "  c (ndarray): N-by-N symmetric matrix C (F-order)\n" \
    "  x (ndarray): N-by-N symmetric solution matrix X (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (sepd, rcond, ferr, info):\n" \
    "  sepd: Separation estimate sepd(op(A),op(A)')\n" \
    "  rcond: Reciprocal condition number estimate\n" \
    "  ferr: Forward error bound\n" \
    "  info: 0=success, 1..n=DGEES fail, n+1=near-reciprocal eigenvalues"

#define DOC_SB03SX "Estimate forward error for discrete-time Lyapunov solution.\n" \
    "\n" \
    "Estimates forward error bound for solution X of op(A)'*X*op(A) - X = C.\n" \
    "\n" \
    "Parameters:\n" \
    "  trana (str): 'N' for op(A)=A, 'T'/'C' for op(A)=A'\n" \
    "  uplo (str): 'U' for upper, 'L' for lower triangle of R\n" \
    "  lyapun (str): 'O' for original, 'R' for reduced form\n" \
    "  n (int): Order of matrices\n" \
    "  xanorm (float): 1-norm of matrix A\n" \
    "  t (ndarray): Schur form matrix T (n x n, F-order)\n" \
    "  u (ndarray): Orthogonal transformation U (n x n, F-order)\n" \
    "  r (ndarray): Cholesky factor of solution X (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (ferr, r, info): Forward error bound, modified R, exit code"

#define DOC_SB03SY "Estimate separation and Theta norm for discrete-time Lyapunov.\n" \
    "\n" \
    "Estimates sepd(op(A),op(A)') and/or 1-norm of Theta operator.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'S' for separation, 'T' for Theta, 'B' for both\n" \
    "  trana (str): 'N' for op(A)=A, 'T'/'C' for op(A)=A'\n" \
    "  lyapun (str): 'O' for original, 'R' for reduced form\n" \
    "  n (int): Order of matrices\n" \
    "  t (ndarray): Schur form matrix T (n x n, F-order)\n" \
    "  u (ndarray): Orthogonal transformation U (n x n, F-order)\n" \
    "  xa (ndarray): Matrix X*op(A) or U'*X*U*op(T) (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (sepd, thnorm, info): Separation, Theta norm, exit code"

#define DOC_SB03TD "Solve continuous-time Lyapunov equation with conditioning/error bounds.\n" \
    "\n" \
    "Solves: op(A)' * X + X * op(A) = scale * C\n" \
    "where op(A) = A or A^T, C and X are N-by-N symmetric.\n" \
    "Optionally estimates conditioning and computes error bound.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'X' solution only, 'S' separation only, 'C' condition only,\n" \
    "             'E' error only, 'A' all of the above\n" \
    "  fact (str): 'F' Schur form provided, 'N' compute Schur\n" \
    "  trana (str): 'N' for op(A)=A, 'T'/'C' for op(A)=A^T\n" \
    "  uplo (str): 'U' upper triangle of C, 'L' lower triangle\n" \
    "  lyapun (str): 'O' original equations, 'R' reduced (Schur form)\n" \
    "  a (ndarray): N-by-N matrix A (if fact='N' or lyapun='O' and job!='X')\n" \
    "  c (ndarray): N-by-N symmetric matrix C (if job!='S')\n" \
    "  t (ndarray, optional): N-by-N Schur form T (if fact='F')\n" \
    "  u (ndarray, optional): N-by-N orthogonal U (if fact='F' and lyapun='O')\n" \
    "  x (ndarray, optional): N-by-N solution X (if job='C' or 'E')\n" \
    "  scale (float, optional): Scale factor (if job='C' or 'E')\n" \
    "\n" \
    "Returns:\n" \
    "  (x, t, u, wr, wi, scale, sep, rcond, ferr, info):\n" \
    "  x: Solution matrix (if job='X' or 'A')\n" \
    "  t: Schur form matrix T\n" \
    "  u: Orthogonal Schur factor U\n" \
    "  wr, wi: Eigenvalue real/imaginary parts (if fact='N')\n" \
    "  scale: Scale factor to prevent overflow\n" \
    "  sep: Separation estimate sep(op(A),-op(A)')\n" \
    "  rcond: Reciprocal condition number (if job='C' or 'A')\n" \
    "  ferr: Forward error bound (if job='E' or 'A')\n" \
    "  info: 0=success, -i=arg i invalid, 1..n=DGEES fail, n+1=close eigenvalues"

#define DOC_SB03UD "Solve discrete-time Lyapunov equation with conditioning/error bounds.\n" \
    "\n" \
    "Solves: op(A)' * X * op(A) - X = scale * C\n" \
    "where op(A) = A or A^T, C and X are N-by-N symmetric.\n" \
    "Optionally estimates conditioning and computes error bound.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'X' solution only, 'S' separation only, 'C' condition only,\n" \
    "             'E' error only, 'A' all of the above\n" \
    "  fact (str): 'F' Schur form provided, 'N' compute Schur\n" \
    "  trana (str): 'N' for op(A)=A, 'T'/'C' for op(A)=A^T\n" \
    "  uplo (str): 'U' upper triangle of C, 'L' lower triangle\n" \
    "  lyapun (str): 'O' original equations, 'R' reduced (Schur form)\n" \
    "  a (ndarray): N-by-N matrix A (if fact='N' or lyapun='O' and job!='X')\n" \
    "  c (ndarray): N-by-N symmetric matrix C (if job!='S')\n" \
    "  t (ndarray, optional): N-by-N Schur form T (if fact='F')\n" \
    "  u (ndarray, optional): N-by-N orthogonal U (if fact='F' and lyapun='O')\n" \
    "  x (ndarray, optional): N-by-N solution X (if job='C' or 'E')\n" \
    "  scale (float, optional): Scale factor (if job='C' or 'E')\n" \
    "\n" \
    "Returns:\n" \
    "  (x, scale, sepd, rcond, ferr, wr, wi, info):\n" \
    "  x: Solution matrix (if job='X' or 'A')\n" \
    "  scale: Scale factor to prevent overflow\n" \
    "  sepd: Separation estimate (if job='S', 'C', or 'A')\n" \
    "  rcond: Reciprocal condition number (if job='C' or 'A')\n" \
    "  ferr: Forward error bound (if job='E' or 'A')\n" \
    "  wr, wi: Eigenvalue real/imaginary parts (if fact='N')\n" \
    "  info: 0=success, -i=arg i invalid, 1..n=DGEES fail, n+1=near-reciprocal"

#define DOC_SB04MD "Solve continuous-time Sylvester equation AX + XB = C (Hessenberg-Schur).\n" \
    "\n" \
    "Solves the continuous-time Sylvester equation AX + XB = C using the\n" \
    "Hessenberg-Schur method. A is reduced to upper Hessenberg form,\n" \
    "B is reduced to real Schur form.\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): N-by-N coefficient matrix A (F-order)\n" \
    "  b (ndarray): M-by-M coefficient matrix B (F-order)\n" \
    "  c (ndarray): N-by-M coefficient matrix C (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (x, z, info): Solution matrix X, orthogonal matrix Z, exit code"

#define DOC_SB04MR "Solve linear system with compact storage (second subdiagonal zeros).\n" \
    "\n" \
    "Solves a linear algebraic system of order M whose coefficient matrix\n" \
    "has zeros below the second subdiagonal. Matrix stored compactly, row-wise.\n" \
    "\n" \
    "Parameters:\n" \
    "  m (int): Order of system (m >= 0)\n" \
    "  d (ndarray): Compact array (m*(m+1)/2 + 3*m elements, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (d, ipr, info): Updated d, solution indices, exit code (0=success, 1=singular)"

#define DOC_SB04MW "Solve linear system with compact storage (upper Hessenberg).\n" \
    "\n" \
    "Solves a linear algebraic system of order M whose coefficient matrix\n" \
    "is in upper Hessenberg form, stored compactly, row-wise.\n" \
    "\n" \
    "Parameters:\n" \
    "  m (int): Order of system (m >= 0)\n" \
    "  d (ndarray): Compact array (m*(m+1)/2 + 2*m elements, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (d, ipr, info): Updated d, solution indices, exit code (0=success, 1=singular)"

#define DOC_SB04ND "Solve continuous-time Sylvester equation AX + XB = C.\n" \
    "\n" \
    "Solves the continuous-time Sylvester equation AX + XB = C, where at least\n" \
    "one of the matrices A or B is in Schur form and the other is in Hessenberg\n" \
    "or Schur form (both either upper or lower).\n" \
    "\n" \
    "Uses the Hessenberg-Schur back substitution method proposed by\n" \
    "Golub, Nash and Van Loan.\n" \
    "\n" \
    "Parameters:\n" \
    "  abschu (str): 'A' = A Schur, B Hessenberg; 'B' = B Schur, A Hessenberg;\n" \
    "                'S' = Both in Schur form\n" \
    "  ula (str): 'U' for upper, 'L' for lower (A matrix)\n" \
    "  ulb (str): 'U' for upper, 'L' for lower (B matrix)\n" \
    "  a (ndarray): N-by-N coefficient matrix A (F-order)\n" \
    "  b (ndarray): M-by-M coefficient matrix B (F-order)\n" \
    "  c (ndarray): N-by-M coefficient matrix C (F-order)\n" \
    "  tol (float, optional): Tolerance for singularity test (default 0 = EPS)\n" \
    "\n" \
    "Returns:\n" \
    "  (x, info): Solution matrix X and exit code (0=success, 1=singular)"

#define DOC_SB04NV "Construct right-hand sides for Sylvester solver (2 RHS case).\n" \
    "\n" \
    "Constructs right-hand sides D for a system of equations in\n" \
    "Hessenberg form solved via sb04nx (case with 2 right-hand sides).\n" \
    "\n" \
    "Parameters:\n" \
    "  abschr (str): 'A' = AB contains A, 'B' = AB contains B\n" \
    "  ul (str): 'U' for upper Hessenberg, 'L' for lower Hessenberg\n" \
    "  indx (int): Position of first column/row of C to use (1-based)\n" \
    "  c (ndarray): N-by-M coefficient matrix C (F-order)\n" \
    "  ab (ndarray): Matrix A or B depending on abschr (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  d (ndarray): Right-hand side vector (2*N or 2*M elements)"

#define DOC_SB04NW "Construct right-hand side for Sylvester solver (1 RHS case).\n" \
    "\n" \
    "Constructs right-hand side D for a system of equations in\n" \
    "Hessenberg form solved via sb04ny (case with 1 right-hand side).\n" \
    "\n" \
    "Parameters:\n" \
    "  abschr (str): 'A' = AB contains A, 'B' = AB contains B\n" \
    "  ul (str): 'U' for upper Hessenberg, 'L' for lower Hessenberg\n" \
    "  indx (int): Position of column/row of C to use (1-based)\n" \
    "  c (ndarray): N-by-M coefficient matrix C (F-order)\n" \
    "  ab (ndarray): Matrix A or B depending on abschr (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  d (ndarray): Right-hand side vector (N or M elements)"

#define DOC_SB04NX "Solve Hessenberg system with two offdiagonals and two RHS.\n" \
    "\n" \
    "Solves a system of equations in Hessenberg form with two consecutive\n" \
    "offdiagonals and two right-hand sides.\n" \
    "\n" \
    "Parameters:\n" \
    "  rc (str): 'R' for row transforms, 'C' for column transforms\n" \
    "  ul (str): 'U' for upper Hessenberg, 'L' for lower Hessenberg\n" \
    "  a (ndarray): M-by-M Hessenberg matrix A (F-order)\n" \
    "  d (ndarray): Right-hand sides (2*M elements, F-order)\n" \
    "  lambd1, lambd2, lambd3, lambd4 (float): 2x2 block for diagonal\n" \
    "  tol (float): Tolerance for singularity test\n" \
    "\n" \
    "Returns:\n" \
    "  (d, info): Solution vector and exit code (0=success, 1=singular)"

#define DOC_SB04NY "Solve Hessenberg system with one offdiagonal and one RHS.\n" \
    "\n" \
    "Solves a system of equations in Hessenberg form with one offdiagonal\n" \
    "and one right-hand side.\n" \
    "\n" \
    "Parameters:\n" \
    "  rc (str): 'R' for row transforms, 'C' for column transforms\n" \
    "  ul (str): 'U' for upper Hessenberg, 'L' for lower Hessenberg\n" \
    "  a (ndarray): M-by-M Hessenberg matrix A (F-order)\n" \
    "  d (ndarray): Right-hand side (M elements, F-order)\n" \
    "  lambda_ (float): Value to add to diagonal elements\n" \
    "  tol (float): Tolerance for singularity test\n" \
    "\n" \
    "Returns:\n" \
    "  (d, info): Solution vector and exit code (0=success, 1=singular)"

#define DOC_SB04OD "Solve generalized Sylvester equations with separation estimation.\n" \
    "\n" \
    "Solves for R and L one of the generalized Sylvester equations:\n" \
    "  A * R - L * B = scale * C, D * R - L * E = scale * F  (equation 1)\n" \
    "or\n" \
    "  A' * R + D' * L = scale * C, R * B' + L * E' = scale * (-F)  (equation 2)\n" \
    "\n" \
    "Optionally reduces (A,D) and/or (B,E) to generalized Schur form and\n" \
    "computes a Dif estimate measuring separation of spectra.\n" \
    "\n" \
    "Parameters:\n" \
    "  reduce (str): 'R' reduce both, 'A' reduce (A,D), 'B' reduce (B,E), 'N' none\n" \
    "  trans (str): 'N' solve equation (1), 'T' solve equation (2)\n" \
    "  jobd (str): '1' one-norm Dif, '2' Frobenius Dif, 'D'/'F' solve+Dif, 'N' no Dif\n" \
    "  a (ndarray): M-by-M matrix A (F-order)\n" \
    "  b (ndarray): N-by-N matrix B (F-order)\n" \
    "  c (ndarray): M-by-N matrix C (F-order)\n" \
    "  d (ndarray): M-by-M matrix D (F-order)\n" \
    "  e (ndarray): N-by-N matrix E (F-order)\n" \
    "  f (ndarray): M-by-N matrix F (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, c, d, e, f, scale, dif, p, q, u, v, info):\n" \
    "    Modified matrices (Schur form), solution R in c, L in f,\n" \
    "    scale factor, Dif estimate, transformation matrices, exit code.\n" \
    "    P,Q for (A,D), U,V for (B,E); None if not computed.\n" \
    "    info: 0=success, 1=Schur fail, 2=not quasi-triangular, 3=singular"

#define DOC_SB04OW "Solve periodic Sylvester equation with matrices in periodic Schur form.\n" \
    "\n" \
    "Solves the periodic Sylvester equation:\n" \
    "  A * R - L * B = scale * C\n" \
    "  D * L - R * E = scale * F\n" \
    "\n" \
    "where R and L are unknown M-by-N matrices, (A, D), (B, E) and (C, F) are\n" \
    "given matrix pairs. (A, D) and (B, E) must be in periodic Schur form,\n" \
    "i.e., A, B are upper quasi triangular and D, E are upper triangular.\n" \
    "The solution (R, L) overwrites (C, F).\n" \
    "\n" \
    "Parameters:\n" \
    "  m (int): Order of A and D, row dimension of C, F, R, L. m > 0.\n" \
    "  n (int): Order of B and E, column dimension of C, F, R, L. n > 0.\n" \
    "  a (ndarray): M-by-M upper quasi triangular matrix A (F-order)\n" \
    "  b (ndarray): N-by-N upper quasi triangular matrix B (F-order)\n" \
    "  c (ndarray): M-by-N matrix C, contains R on exit (F-order)\n" \
    "  d (ndarray): M-by-M upper triangular matrix D (F-order)\n" \
    "  e (ndarray): N-by-N upper triangular matrix E (F-order)\n" \
    "  f (ndarray): M-by-N matrix F, contains L on exit (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (r, l, scale, info):\n" \
    "    r: Solution matrix R (modified c)\n" \
    "    l: Solution matrix L (modified f)\n" \
    "    scale: Scaling factor (0 <= scale <= 1)\n" \
    "    info: 0=success, >0=common eigenvalues in A*D and B*E"

#define DOC_SB04PD "Solve continuous-time or discrete-time Sylvester equations.\n" \
    "\n" \
    "Solves for X either the continuous-time Sylvester equation:\n" \
    "  op(A)*X + ISGN*X*op(B) = scale*C\n" \
    "\n" \
    "or the discrete-time Sylvester equation:\n" \
    "  op(A)*X*op(B) + ISGN*X = scale*C\n" \
    "\n" \
    "where op(M) = M or M**T, and ISGN = 1 or -1. If A and/or B are not\n" \
    "quasi-triangular, they are reduced to Schur canonical form.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  facta (str): 'F' Schur supplied, 'N' compute Schur, 'S' already Schur\n" \
    "  factb (str): 'F' Schur supplied, 'N' compute Schur, 'S' already Schur\n" \
    "  trana (str): 'N' for A, 'T' or 'C' for A**T\n" \
    "  tranb (str): 'N' for B, 'T' or 'C' for B**T\n" \
    "  isgn (int): Sign in equation (must be 1 or -1)\n" \
    "  a (ndarray): M-by-M matrix A (F-order)\n" \
    "  b (ndarray): N-by-N matrix B (F-order)\n" \
    "  c (ndarray): M-by-N right-hand side matrix C (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (x, a, u, b, v, scale, info):\n" \
    "    x: M-by-N solution matrix\n" \
    "    a: Schur form of A (if facta='N')\n" \
    "    u: Schur vectors of A (None if facta='S')\n" \
    "    b: Schur form of B (if factb='N')\n" \
    "    v: Schur vectors of B (None if factb='S')\n" \
    "    scale: Scale factor (0 < scale <= 1)\n" \
    "    info: 0=success, <0=arg error, >0=algorithm error"

#define DOC_SB04PX "Solve 2x2 Sylvester equation.\n" \
    "\n" \
    "Solves for N1-by-N2 matrix X (1 <= N1,N2 <= 2) in:\n" \
    "  op(TL)*X*op(TR) + ISGN*X = SCALE*B\n" \
    "\n" \
    "where op(T) = T or T', ISGN = 1 or -1.\n" \
    "\n" \
    "Parameters:\n" \
    "  ltranl (bool): True for op(TL) = TL', False for op(TL) = TL\n" \
    "  ltranr (bool): True for op(TR) = TR', False for op(TR) = TR\n" \
    "  isgn (int): Sign (+1 or -1)\n" \
    "  n1 (int): Rows of TL and B (1 or 2)\n" \
    "  n2 (int): Columns of TR and B (1 or 2)\n" \
    "  tl (ndarray): N1-by-N1 matrix TL (F-order)\n" \
    "  tr (ndarray): N2-by-N2 matrix TR (F-order)\n" \
    "  b (ndarray): N1-by-N2 right-hand side matrix (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (scale, x, xnorm, info):\n" \
    "    scale: Scale factor (0 < scale <= 1)\n" \
    "    x: N1-by-N2 solution matrix\n" \
    "    xnorm: Infinity norm of X\n" \
    "    info: 0=success, 1=TL and TR have common/nearby eigenvalues"

#define DOC_SB04PY "Solve discrete-time Sylvester equation with Schur matrices.\n" \
    "\n" \
    "Solves for X the discrete-time Sylvester equation:\n" \
    "  op(A)*X*op(B) + ISGN*X = scale*C\n" \
    "\n" \
    "where op(A) = A or A**T, A and B are both upper quasi-triangular\n" \
    "(Schur canonical form), and ISGN = 1 or -1. Solution X overwrites C.\n" \
    "\n" \
    "Parameters:\n" \
    "  trana (str): 'N' for A, 'T' or 'C' for A**T\n" \
    "  tranb (str): 'N' for B, 'T' or 'C' for B**T\n" \
    "  isgn (int): Sign of the equation (must be 1 or -1)\n" \
    "  a (ndarray): M-by-M upper quasi-triangular matrix A (F-order)\n" \
    "  b (ndarray): N-by-N upper quasi-triangular matrix B (F-order)\n" \
    "  c (ndarray): M-by-N right-hand side matrix C (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (x, scale, info): Solution X, scale factor, exit code\n" \
    "  scale: Scale factor (0 < scale <= 1) to prevent overflow\n" \
    "  info: 0=success, 1=A and -ISGN*B have almost reciprocal eigenvalues"

#define DOC_SB04QD "Solve discrete-time Sylvester equation X + AXB = C (Hessenberg-Schur).\n" \
    "\n" \
    "Solves the discrete-time Sylvester equation X + AXB = C using the\n" \
    "Hessenberg-Schur method. A is reduced to upper Hessenberg form,\n" \
    "B is reduced to real Schur form.\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): N-by-N coefficient matrix A (F-order)\n" \
    "  b (ndarray): M-by-M coefficient matrix B (F-order)\n" \
    "  c (ndarray): N-by-M coefficient matrix C (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (x, z, info): Solution matrix X, orthogonal matrix Z, exit code"

#define DOC_SB04QR "Solve linear system with compact storage (third subdiagonal pattern).\n" \
    "\n" \
    "Solves a linear algebraic system of order M whose coefficient matrix\n" \
    "has zeros below third subdiagonal with special pattern. Matrix stored compactly.\n" \
    "\n" \
    "Parameters:\n" \
    "  m (int): Order of system (m >= 0, m even)\n" \
    "  d (ndarray): Compact array (m*m/2 + 4*m elements, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (d, ipr, info): Updated d, solution indices, exit code (0=success, 1=singular)"

#define DOC_SB04RD "Solve discrete-time Sylvester equation (Hessenberg-Schur method).\n" \
    "\n" \
    "Solves X + A*X*B = C where at least one of A or B is in Schur form\n" \
    "and the other in Hessenberg or Schur form.\n" \
    "\n" \
    "Parameters:\n" \
    "  abschu (str): 'A' = A in Schur, B in Hessenberg\n" \
    "                'B' = B in Schur, A in Hessenberg\n" \
    "                'S' = both in Schur form\n" \
    "  ula (str): 'U' = A is upper, 'L' = A is lower\n" \
    "  ulb (str): 'U' = B is upper, 'L' = B is lower\n" \
    "  a (ndarray): N-by-N coefficient matrix A (F-order)\n" \
    "  b (ndarray): M-by-M coefficient matrix B (F-order)\n" \
    "  c (ndarray): N-by-M RHS matrix C (F-order)\n" \
    "  tol (float, optional): Tolerance for singularity test (default: machine eps)\n" \
    "\n" \
    "Returns:\n" \
    "  (x, info): Solution matrix X and exit code\n" \
    "    info=0: success, info=1: nearly singular matrix encountered"

#define DOC_SB04RV "Construct RHS for quasi-Hessenberg Sylvester solver (2 RHS case).\n" \
    "\n" \
    "Constructs right-hand sides D for a system of equations in quasi-Hessenberg\n" \
    "form solved via SB04RX (case with 2 right-hand sides).\n" \
    "\n" \
    "For the Sylvester equation X + AXB = C:\n" \
    "  - If abschr='B': ab contains B (m-by-m), ba contains A (n-by-n), D has 2*n elements\n" \
    "  - If abschr='A': ab contains A (n-by-n), ba contains B (m-by-m), D has 2*m elements\n" \
    "\n" \
    "Parameters:\n" \
    "  abschr (str): 'A' if ab contains A, 'B' if ab contains B\n" \
    "  ul (str): 'U' if ab is upper Hessenberg, 'L' if lower Hessenberg\n" \
    "  indx (int): Position of first column/row of C to use (1-based)\n" \
    "  c (ndarray): N-by-M coefficient/solution matrix (F-order)\n" \
    "  ab (ndarray): Hessenberg matrix A or B (F-order)\n" \
    "  ba (ndarray): Matrix B or A (the one not in ab) (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  d (ndarray): Right-hand side vector (2*n or 2*m elements, interleaved)"

#define DOC_SB04RW "Construct RHS for Hessenberg Sylvester solver (1 RHS case).\n" \
    "\n" \
    "Constructs right-hand side D for a system of equations in Hessenberg\n" \
    "form solved via SB04RY (case with 1 right-hand side).\n" \
    "\n" \
    "For the Sylvester equation X + AXB = C:\n" \
    "  - If abschr='B': ab contains B (m-by-m), ba contains A (n-by-n), D has n elements\n" \
    "  - If abschr='A': ab contains A (n-by-n), ba contains B (m-by-m), D has m elements\n" \
    "\n" \
    "Parameters:\n" \
    "  abschr (str): 'A' if ab contains A, 'B' if ab contains B\n" \
    "  ul (str): 'U' if ab is upper Hessenberg, 'L' if lower Hessenberg\n" \
    "  indx (int): Position of column/row of C to use (1-based)\n" \
    "  c (ndarray): N-by-M coefficient/solution matrix (F-order)\n" \
    "  ab (ndarray): Hessenberg matrix A or B (F-order)\n" \
    "  ba (ndarray): Matrix B or A (the one not in ab) (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  d (ndarray): Right-hand side vector (n or m elements)"

#define DOC_SB04RX "Solve quasi-Hessenberg system with two right-hand sides.\n" \
    "\n" \
    "Solves a system of equations in quasi-Hessenberg form (Hessenberg form\n" \
    "plus two consecutive offdiagonals) with two right-hand sides via QR\n" \
    "decomposition with Givens rotations.\n" \
    "\n" \
    "The system matrix H is formed as I (x) A * Lambda + Lambda (x) A\n" \
    "where Lambda is a 2x2 block and (x) denotes Kronecker product.\n" \
    "\n" \
    "Parameters:\n" \
    "  rc (str): 'R' for row transformations, 'C' for column transformations\n" \
    "  ul (str): 'U' if A is upper Hessenberg, 'L' if lower Hessenberg\n" \
    "  a (ndarray): M-by-M Hessenberg matrix A (F-order)\n" \
    "  lambd1 (float): Element (1,1) of 2x2 block\n" \
    "  lambd2 (float): Element (1,2) of 2x2 block\n" \
    "  lambd3 (float): Element (2,1) of 2x2 block\n" \
    "  lambd4 (float): Element (2,2) of 2x2 block\n" \
    "  d (ndarray): 2*M RHS vector, stored row-wise (F-order)\n" \
    "  tol (float): Tolerance for near-singularity test\n" \
    "\n" \
    "Returns:\n" \
    "  (d, info): Solution vector (2*M elements) and exit code\n" \
    "    info=0: success, info=1: system is (numerically) singular"

#define DOC_SB04RY "Solve Hessenberg system with one right-hand side.\n" \
    "\n" \
    "Solves a system of equations (I + lambda*A)*x = d in Hessenberg form\n" \
    "with one right-hand side via QR decomposition with Givens rotations.\n" \
    "\n" \
    "Parameters:\n" \
    "  rc (str): 'R' for row transformations, 'C' for column transformations\n" \
    "  ul (str): 'U' if A is upper Hessenberg, 'L' if lower Hessenberg\n" \
    "  a (ndarray): M-by-M Hessenberg matrix A (F-order)\n" \
    "  lambda (float): Scalar multiplier for A\n" \
    "  d (ndarray): M-element RHS vector (F-order)\n" \
    "  tol (float): Tolerance for near-singularity test\n" \
    "\n" \
    "Returns:\n" \
    "  (d, info): Solution vector (M elements) and exit code\n" \
    "    info=0: success, info=1: system is (numerically) singular"

#define DOC_SB06ND "Minimum norm feedback matrix for deadbeat control.\n" \
    "\n" \
    "Constructs the minimum norm feedback matrix F to perform 'deadbeat control'\n" \
    "on a (A,B)-pair reduced to staircase form by AB01OD such that R = A + BFU'\n" \
    "is nilpotent.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): State dimension, order of matrix A. n >= 0.\n" \
    "  m (int): Input dimension. m >= 0.\n" \
    "  kmax (int): Number of stairs in staircase form from AB01OD. 0 <= kmax <= n.\n" \
    "  a (ndarray): N-by-N state matrix in staircase form (F-order).\n" \
    "  b (ndarray): N-by-M input matrix in triangular form (F-order).\n" \
    "  kstair (ndarray): KMAX-element array of stair dimensions from AB01OD.\n" \
    "  u (ndarray): N-by-N transformation matrix or identity (F-order).\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, u, f, info): Modified A (U'AU + U'BF), modified B (U'B),\n" \
    "    modified U, deadbeat feedback matrix F, and exit code."

#define DOC_SB08CD "Left coprime factorization with inner denominator.\n" \
    "\n" \
    "Computes a left coprime factorization G = R^{-1}*Q with co-inner denominator R\n" \
    "of a given state-space representation (A,B,C,D). The factorization satisfies\n" \
    "R(s)*R'(-s) = I (continuous) or R(z)*R'(1/z) = I (discrete).\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  a (ndarray): N-by-N state matrix (F-order)\n" \
    "  b (ndarray): N-by-max(M,P) input matrix (F-order)\n" \
    "  c (ndarray): max(M,P)-by-N output matrix (F-order)\n" \
    "  d (ndarray): max(M,P)-by-max(M,P) feedthrough matrix (F-order)\n" \
    "  tol (float, optional): Tolerance for observability test (default: auto)\n" \
    "\n" \
    "Returns:\n" \
    "  (nq, nr, br, dr, iwarn, info): Order of Q/R, order of minimal R,\n" \
    "    output injection matrix Z'*H (N-by-P), lower triangular V (P-by-P),\n" \
    "    warning count, exit code"

#define DOC_SB08DD "Right coprime factorization with inner denominator.\n" \
    "\n" \
    "Computes a right coprime factorization G = N*M^(-1) with inner denominator\n" \
    "of a given state-space representation (A,B,C,D).\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  a (ndarray): N-by-N state matrix (F-order)\n" \
    "  b (ndarray): N-by-M input matrix (F-order)\n" \
    "  c (ndarray): P-by-N output matrix (F-order)\n" \
    "  d (ndarray): P-by-M feedthrough matrix (F-order)\n" \
    "  tol (float): Tolerance for rank determination\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, c, d, nq, nr, cr, dr, iwarn, info): Modified matrices, orders,\n" \
    "    denominator matrices CR and DR, warning and exit codes"

#define DOC_SB08ED "Left coprime factorization with prescribed stability degree.\n" \
    "\n" \
    "Constructs output injection matrix H and orthogonal transformation Z such that\n" \
    "Q = (Z'*(A+H*C)*Z, Z'*(B+H*D), C*Z, D) and R = (Z'*(A+H*C)*Z, Z'*H, C*Z, I)\n" \
    "provide stable left coprime factorization G = R^(-1) * Q.\n" \
    "\n" \
    "Undetectable parts are automatically deflated. Input arrays A, B, C, D are\n" \
    "modified in-place to contain the factorization results.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  a (ndarray): N-by-N state matrix (F-order, modified to contain AQ in Schur)\n" \
    "  b (ndarray): N-by-M input matrix (F-order, modified to contain BQ)\n" \
    "  c (ndarray): P-by-N output matrix (F-order, modified to contain CQ)\n" \
    "  d (ndarray): P-by-M feedthrough matrix (F-order, = DQ unchanged)\n" \
    "  alpha (ndarray): [stability_degree, stability_margin]. For continuous:\n" \
    "    eigenvalues outside alpha[1]-region get real parts = alpha[0] < 0.\n" \
    "    For discrete: 0 <= alpha[i] < 1.\n" \
    "  tol (float, optional): Tolerance for observability test (default: auto)\n" \
    "\n" \
    "Returns:\n" \
    "  (nq, nr, br, dr, iwarn, info):\n" \
    "    nq = order of factors Q and R, nr = order of minimal R,\n" \
    "    br = NQ-by-P output injection matrix Z'*H (first NR rows = BR of R),\n" \
    "    dr = P-by-P identity matrix, iwarn = warning count, info = exit code"

#define DOC_SB08FD "Right coprime factorization with prescribed stability degree.\n" \
    "\n" \
    "Constructs feedback matrix F and orthogonal transformation Z such that\n" \
    "Q = (Z'*(A+B*F)*Z, Z'*B, (C+D*F)*Z, D) and R = (Z'*(A+B*F)*Z, Z'*B, F*Z, I)\n" \
    "provide stable right coprime factorization G = Q * R^(-1).\n" \
    "\n" \
    "Unstabilizable parts are automatically deflated.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  n (int): State dimension (order of A)\n" \
    "  m (int): Number of inputs\n" \
    "  p (int): Number of outputs\n" \
    "  alpha (ndarray): [stability_degree, stability_margin]. For continuous:\n" \
    "    eigenvalues outside alpha[1]-region get real parts = alpha[0] < 0.\n" \
    "    For discrete: 0 <= alpha[i] < 1.\n" \
    "  a (ndarray): N-by-N state matrix (F-order)\n" \
    "  b (ndarray): N-by-M input matrix (F-order)\n" \
    "  c (ndarray): P-by-N output matrix (F-order)\n" \
    "  d (ndarray): P-by-M feedthrough matrix (F-order)\n" \
    "  tol (float, optional): Tolerance for controllability test\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, c, cr, dr, nq, nr, iwarn, info): Modified A,B,C in Schur form,\n" \
    "    feedback matrix CR (M-by-N), identity DR (M-by-M),\n" \
    "    order NQ, denominator order NR, warning count, exit code"

#define DOC_SB08GD "State-space representation from left coprime factorization.\n" \
    "\n" \
    "Constructs the state-space representation G = (A,B,C,D) from factors\n" \
    "Q = (AQR,BQ,CQR,DQ) and R = (AQR,BR,CQR,DR) of left coprime factorization\n" \
    "G = R^{-1} * Q.\n" \
    "\n" \
    "Formulas:\n" \
    "  A = AQR - BR * DR^{-1} * CQR\n" \
    "  B = BQ  - BR * DR^{-1} * DQ\n" \
    "  C = DR^{-1} * CQR\n" \
    "  D = DR^{-1} * DQ\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): N-by-N state matrix AQR (F-order, modified in-place)\n" \
    "  b (ndarray): N-by-M input matrix BQ (F-order, modified in-place)\n" \
    "  c (ndarray): P-by-N output matrix CQR (F-order, modified in-place)\n" \
    "  d (ndarray): P-by-M feedthrough matrix DQ (F-order, modified in-place)\n" \
    "  br (ndarray): N-by-P input matrix BR of system R (F-order)\n" \
    "  dr (ndarray): P-by-P matrix DR (F-order, returns LU factorization)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, c, d, dr, rcond, info): Modified A,B,C,D matrices, LU of DR,\n" \
    "    reciprocal condition number of DR, exit code\n" \
    "    info=0: success, info=1: DR singular, info=2: DR numerically singular (warning)"

#define DOC_SB08HD "State-space representation from right coprime factorization.\n" \
    "\n" \
    "Constructs the state-space representation G = (A,B,C,D) from factors\n" \
    "Q = (AQR,BQR,CQ,DQ) and R = (AQR,BQR,CR,DR) of right coprime factorization\n" \
    "G = Q * R^{-1}.\n" \
    "\n" \
    "Formulas:\n" \
    "  A = AQR - BQR * DR^{-1} * CR\n" \
    "  B = BQR * DR^{-1}\n" \
    "  C = CQ  - DQ * DR^{-1} * CR\n" \
    "  D = DQ * DR^{-1}\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): N-by-N state matrix AQR (F-order, modified in-place)\n" \
    "  b (ndarray): N-by-M input matrix BQR (F-order, modified in-place)\n" \
    "  c (ndarray): P-by-N output matrix CQ (F-order, modified in-place)\n" \
    "  d (ndarray): P-by-M feedthrough matrix DQ (F-order, modified in-place)\n" \
    "  cr (ndarray): M-by-N output matrix CR of system R (F-order)\n" \
    "  dr (ndarray): M-by-M matrix DR (F-order, returns LU factorization)\n" \
    "\n" \
    "Returns:\n" \
    "  (rcond, info): Reciprocal condition number of DR, exit code.\n" \
    "    info=0: success, info=1: DR singular, info=2: DR numerically singular (warning)\n" \
    "    Arrays a, b, c, d are modified in-place to contain G's matrices."

#define DOC_SB08MD "Spectral factorization of polynomials (continuous-time case).\n" \
    "\n" \
    "Computes a real polynomial E(s) such that:\n" \
    "  (a) E(-s) * E(s) = A(-s) * A(s)\n" \
    "  (b) E(s) is stable (all zeros have non-positive real parts)\n" \
    "\n" \
    "The input polynomial may be supplied as A(s) or as B(s) = A(-s) * A(s).\n" \
    "\n" \
    "Parameters:\n" \
    "  acona (str): 'A' to supply A(s) coefficients, 'B' to supply B(s) coefficients\n" \
    "  da (int): Degree of polynomials A(s) and E(s). DA >= 0.\n" \
    "  a (ndarray): Coefficients in increasing powers of s (if acona='A') or s**2\n" \
    "    (if acona='B'). Length DA+1.\n" \
    "\n" \
    "Returns:\n" \
    "  (e, b, res, info):\n" \
    "    e: Coefficients of spectral factor E(s) in increasing powers of s (length DA+1)\n" \
    "    b: Coefficients of B(s) = A(-s)*A(s) in increasing powers of s**2 (length DA+1)\n" \
    "    res: Accuracy estimate for E(s) coefficients\n" \
    "    info: 0=success, 1=all A(i)=0, 2=invalid B(s) for ACONA='B',\n" \
    "          3=no convergence in 30 iterations, 4=last iterate unstable"

#define DOC_SB08MY "Compute B(s) = A(s) * A(-s) for spectral factorization.\n" \
    "\n" \
    "Computes coefficients of B(s) = A(s) * A(-s) where A(s) is a polynomial\n" \
    "given in increasing powers of s. B(s) is returned in increasing powers\n" \
    "of s**2. Also computes an accuracy norm.\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): Coefficients of A(s) in increasing powers of s (length DA+1)\n" \
    "  epsb (float): Machine precision (e.g., np.finfo(float).eps)\n" \
    "\n" \
    "Returns:\n" \
    "  (b, epsb): Coefficients of B(s) in powers of s**2 (length DA+1),\n" \
    "    updated accuracy norm (3 * max_abs_sum * epsb)"

#define DOC_SB08ND "Spectral factorization of polynomials (discrete-time case).\n" \
    "\n" \
    "Computes a real polynomial E(z) such that:\n" \
    "  (a) E(1/z) * E(z) = A(1/z) * A(z)\n" \
    "  (b) E(z) is stable (all zeros have modulus <= 1)\n" \
    "\n" \
    "The input polynomial may be supplied as A(z) or as B(z) = A(1/z) * A(z).\n" \
    "\n" \
    "Parameters:\n" \
    "  acona (str): 'A' to supply A(z) coefficients, 'B' to supply B(z) coefficients\n" \
    "  da (int): Degree of polynomials A(z) and E(z). DA >= 0.\n" \
    "  a (ndarray): Coefficients in increasing powers of z. Length DA+1.\n" \
    "\n" \
    "Returns:\n" \
    "  (e, b, res, info):\n" \
    "    e: Coefficients of spectral factor E(z) in increasing powers of z (length DA+1)\n" \
    "    b: Coefficients of B(z) = A(1/z)*A(z) as defined in equation (1) (length DA+1)\n" \
    "    res: Accuracy estimate for E(z) coefficients\n" \
    "    info: 0=success, 2=invalid B(z) for ACONA='B',\n" \
    "          3=no convergence in 30 iterations, 4=last iterate unstable"

#define DOC_SB08NY "Compute B(z) = A(1/z) * A(z) for discrete-time spectral factorization.\n" \
    "\n" \
    "Computes the coefficients of B(z) = A(1/z) * A(z) where A(z) is a\n" \
    "polynomial given in increasing powers of z. The output B contains the\n" \
    "autocorrelation coefficients of A: b[k] = sum_{j=0}^{da-k} a[j]*a[j+k].\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): Coefficients of A(z) in increasing powers of z (length DA+1)\n" \
    "\n" \
    "Returns:\n" \
    "  (b, epsb): Autocorrelation coefficients (length DA+1),\n" \
    "    accuracy norm (3 * machine_eps * b[0])"

#define DOC_SB09MD "Evaluate closeness of two multivariable sequences.\n" \
    "\n" \
    "Compares M1(k) and M2(k) for k=1,...,N where each is NC-by-NB matrix.\n" \
    "\n" \
    "Computes:\n" \
    "  SS(i,j) = sum_{k=1}^N M1(i,j,k)^2  (sum of squares)\n" \
    "  SE(i,j) = sum_{k=1}^N (M1(i,j,k) - M2(i,j,k))^2  (quadratic error)\n" \
    "  PRE(i,j) = 100 * sqrt(SE(i,j)/SS(i,j))  (percentage relative error)\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Number of parameters (N >= 0)\n" \
    "  nc (int): Number of rows in M1(k), M2(k) (NC >= 0)\n" \
    "  nb (int): Number of columns in M1(k), M2(k) (NB >= 0)\n" \
    "  h1 (ndarray): NC-by-(N*NB) array with M1(k) sequence (F-order)\n" \
    "  h2 (ndarray): NC-by-(N*NB) array with M2(k) sequence (F-order)\n" \
    "  tol (float): Tolerance for computation (if < eps, eps is used)\n" \
    "\n" \
    "Returns:\n" \
    "  (ss, se, pre, info): Sum-of-squares matrix, quadratic error matrix,\n" \
    "    percentage relative error matrix, and exit code"

#define DOC_SB10AD "H-infinity optimal n-state controller synthesis.\n" \
    "\n" \
    "Computes an H-infinity optimal n-state controller K = (AK, BK, CK, DK)\n" \
    "using modified Glover-Doyle 1988 formulas and the closed-loop system\n" \
    "G = (AC, BC, CC, DC) for the plant P = (A, B, C, D).\n" \
    "\n" \
    "Plant partitioning:\n" \
    "  | A  | B1  B2  |\n" \
    "  |----|---------|  where B2 has NCON columns, C2 has NMEAS rows\n" \
    "  | C1 | D11 D12 |\n" \
    "  | C2 | D21 D22 |\n" \
    "\n" \
    "JOB modes:\n" \
    "  1: Bisection for gamma reduction\n" \
    "  2: Scan from gamma to 0\n" \
    "  3: Bisection then scanning\n" \
    "  4: Suboptimal controller only\n" \
    "\n" \
    "Parameters:\n" \
    "  job (int): Optimization mode (1-4)\n" \
    "  n (int): State dimension\n" \
    "  m (int): Number of inputs\n" \
    "  np (int): Number of outputs\n" \
    "  ncon (int): Number of control inputs (M2)\n" \
    "  nmeas (int): Number of measurements (NP2)\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (np x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (np x m, F-order)\n" \
    "  gamma (float): Initial H-infinity norm bound (gamma > 0)\n" \
    "  gtol (float): Tolerance for gamma (<=0 uses sqrt(eps))\n" \
    "  actol (float): Tolerance for stability (<=0 uses stability check)\n" \
    "\n" \
    "Returns:\n" \
    "  (ak, bk, ck, dk, ac, bc, cc, dc, gamma, rcond, info):\n" \
    "    Controller K, closed-loop G, achieved gamma, condition numbers, exit code"

#define DOC_SB10DD "H-infinity (sub)optimal controller for discrete-time system.\n" \
    "\n" \
    "Computes the matrices of an H-infinity controller K = [AK BK; CK DK]\n" \
    "for the discrete-time system P = [A B; C D] partitioned as\n" \
    "P = [A B1 B2; C1 D11 D12; C2 D21 D22] where B2 has NCON columns\n" \
    "and C2 has NMEAS rows.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Order of the system\n" \
    "  m (int): Column size of B\n" \
    "  np (int): Row size of C\n" \
    "  ncon (int): Number of control inputs\n" \
    "  nmeas (int): Number of measurements\n" \
    "  gamma (float): H-infinity norm bound (gamma > 0)\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (np x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (np x m, F-order)\n" \
    "  tol (float, optional): Tolerance for rank determination\n" \
    "\n" \
    "Returns:\n" \
    "  (ak, bk, ck, dk, x, z, rcond, info): Controller matrices,\n" \
    "    Riccati solutions, condition numbers, exit code"

#define DOC_SB10ED "H2 optimal n-state controller for discrete-time system.\n" \
    "\n" \
    "Computes the matrices of the H2 optimal controller K = [AK BK; CK DK]\n" \
    "for the discrete-time system P = [A B; C D] partitioned as\n" \
    "P = [A B1 B2; C1 0 D12; C2 D21 D22] where B2 has NCON columns\n" \
    "and C2 has NMEAS rows.\n" \
    "\n" \
    "Assumptions:\n" \
    "  - (A,B2) is stabilizable and (C2,A) is detectable\n" \
    "  - D12 is full column rank and D21 is full row rank\n" \
    "  - Discrete-time invariant zeros conditions (A3, A4)\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Order of the system (>= 0)\n" \
    "  m (int): Column size of B (>= 0)\n" \
    "  np (int): Row size of C (>= 0)\n" \
    "  ncon (int): Number of control inputs M2 (0 <= ncon <= m, ncon <= np-nmeas)\n" \
    "  nmeas (int): Number of measurements NP2 (0 <= nmeas <= np, nmeas <= m-ncon)\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (np x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (np x m, F-order)\n" \
    "  tol (float, optional): Tolerance for transformations. Default sqrt(eps).\n" \
    "\n" \
    "Returns:\n" \
    "  (ak, bk, ck, dk, rcond, info): Controller matrices, condition numbers, exit code\n" \
    "  rcond has 7 elements:\n" \
    "    [0]: control transformation TU\n" \
    "    [1]: measurement transformation TY\n" \
    "    [2]: Im2 + B2'*X2*B2\n" \
    "    [3]: Ip2 + C2*Y2*C2'\n" \
    "    [4]: X-Riccati equation\n" \
    "    [5]: Y-Riccati equation\n" \
    "    [6]: Im2 + DKHAT*D22\n" \
    "  info: 0=success, 1-2=rank deficiency, 3-4=D12/D21 rank, 5=SVD,\n" \
    "    6=X-Riccati, 7=B2'*X*B2 not PD, 8=Y-Riccati, 9=C2*Y*C2' not PD,\n" \
    "    10=Im2+DKHAT*D22 singular"

#define DOC_SB10FD "H-infinity (sub)optimal controller for continuous-time system.\n" \
    "\n" \
    "Computes the matrices of an H-infinity controller K = [AK BK; CK DK]\n" \
    "for the continuous-time system P = [A B; C D] partitioned as\n" \
    "P = [A B1 B2; C1 D11 D12; C2 D21 D22] where B2 has NCON columns\n" \
    "and C2 has NMEAS rows. Uses Glover-Doyle formulas for fixed gamma.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Order of the system\n" \
    "  m (int): Column size of B\n" \
    "  np (int): Row size of C\n" \
    "  ncon (int): Number of control inputs\n" \
    "  nmeas (int): Number of measurements\n" \
    "  gamma (float): H-infinity norm bound (gamma > 0)\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (np x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (np x m, F-order)\n" \
    "  tol (float, optional): Tolerance for transformations. Default sqrt(eps).\n" \
    "\n" \
    "Returns:\n" \
    "  (ak, bk, ck, dk, rcond, info): Controller matrices, condition numbers, exit code"

#define DOC_SB10HD "Compute H2 optimal n-state controller for continuous-time system.\n" \
    "\n" \
    "Computes controller K = (AK, BK, CK, DK) for a plant P = (A, B, C, D)\n" \
    "partitioned as P = [A B1 B2; C1 0 D12; C2 D21 D22].\n" \
    "\n" \
    "Assumptions:\n" \
    "  - (A,B2) is stabilizable and (C2,A) is detectable\n" \
    "  - D11 = 0\n" \
    "  - D12 is full column rank and D21 is full row rank\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): System order (>= 0)\n" \
    "  m (int): Number of inputs (>= 0)\n" \
    "  np (int): Number of outputs (>= 0)\n" \
    "  ncon (int): Number of control inputs M2 (0 <= ncon <= m, ncon <= np-nmeas)\n" \
    "  nmeas (int): Number of measurements NP2 (0 <= nmeas <= np, nmeas <= m-ncon)\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (np x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (np x m, F-order)\n" \
    "  tol (float): Tolerance for rank tests. If <= 0, sqrt(eps) is used.\n" \
    "\n" \
    "Returns:\n" \
    "  (ak, bk, ck, dk, rcond, info): Controller matrices, condition estimates, exit code\n" \
    "  rcond has 4 elements: control/measurement transform rcond, X/Y Riccati rcond\n" \
    "  info: 0=success, 1=D12 rank, 2=D21 rank, 3=SVD, 4=X-Riccati, 5=Y-Riccati"

#define DOC_SB10ID "Positive feedback controller for loop shaping design.\n" \
    "\n" \
    "Computes controller K = [AK BK; CK DK] for shaped plant G = [A B; C D]\n" \
    "in the McFarlane/Glover Loop Shaping Design Procedure.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Order of the plant (>= 0)\n" \
    "  m (int): Number of plant inputs (>= 0)\n" \
    "  np (int): Number of plant outputs (>= 0)\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (np x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (np x m, F-order)\n" \
    "  factor (float): =1 for optimal controller, >1 for suboptimal\n" \
    "\n" \
    "Returns:\n" \
    "  (ak, bk, ck, dk, nk, rcond, info): Controller matrices, order, cond numbers, exit code\n" \
    "  nk: Controller order (nk <= n)\n" \
    "  rcond: [X-Riccati rcond, Z-Riccati rcond]\n" \
    "  info: 0=success, 1=X-Riccati, 2=Z-Riccati, 3=iteration,\n" \
    "        4=Ip-D*Dk singular, 5=Im-Dk*D singular, 6=unstable"

#define DOC_SB10JD "Convert descriptor state-space to regular state-space.\n" \
    "\n" \
    "Converts E*dx/dt = A*x + B*u to dx/dt = Ad*x + Bd*u using SVD.\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (np x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (np x m, F-order)\n" \
    "  e (ndarray): Descriptor matrix E (n x n, F-order)\n" \
    "  n (int, optional): Order of descriptor system\n" \
    "  m (int, optional): Number of inputs\n" \
    "  np (int, optional): Number of outputs\n" \
    "\n" \
    "Returns:\n" \
    "  (ad, bd, cd, dd, nsys, info): Converted matrices, reduced order, exit code"

#define DOC_SB10KD "Discrete-time loop shaping controller design.\n" \
    "\n" \
    "Computes the positive feedback controller K = [Ak, Bk; Ck, Dk]\n" \
    "for the shaped plant G = [A, B; C, 0] using the McFarlane-Glover\n" \
    "method in discrete-time.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Order of the plant (N >= 0)\n" \
    "  m (int): Column size of matrix B (M >= 0)\n" \
    "  np (int): Row size of matrix C (NP >= 0)\n" \
    "  a (ndarray): N-by-N system state matrix (F-order)\n" \
    "  b (ndarray): N-by-M system input matrix (F-order)\n" \
    "  c (ndarray): NP-by-N system output matrix (F-order)\n" \
    "  factor (float): Performance factor (=1 optimal, >1 suboptimal)\n" \
    "\n" \
    "Returns:\n" \
    "  (ak, bk, ck, dk, rcond, info): Controller matrices,\n" \
    "    reciprocal condition numbers array(4), and exit code\n" \
    "    info=0: success, 1: P-Riccati failed, 2: Q-Riccati failed,\n" \
    "    3: X-Riccati failed, 4: eigenvalue computation failed,\n" \
    "    5: singular matrix, 6: closed-loop unstable"

#define DOC_SB10LD "Compute closed-loop system matrices.\n" \
    "\n" \
    "Computes the closed-loop system G = (AC, BC, CC, DC) from the\n" \
    "open-loop plant P = (A, B, C, D) and controller K = (AK, BK, CK, DK).\n" \
    "\n" \
    "The closed-loop has state dimension 2*n and computes:\n" \
    "  u = u1 + u2 where u2 = K * y2 (controller output)\n" \
    "  y = y1 + y2 where y2 is measured output\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): State dimension\n" \
    "  m (int): Number of inputs\n" \
    "  np (int): Number of outputs\n" \
    "  ncon (int): Number of control inputs (M2)\n" \
    "  nmeas (int): Number of measurements (NP2)\n" \
    "  a (ndarray): Plant state matrix A (n x n, F-order)\n" \
    "  b (ndarray): Plant input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Plant output matrix C (np x n, F-order)\n" \
    "  d (ndarray): Plant feedthrough D (np x m, F-order)\n" \
    "  ak (ndarray): Controller state matrix AK (n x n, F-order)\n" \
    "  bk (ndarray): Controller input matrix BK (n x nmeas, F-order)\n" \
    "  ck (ndarray): Controller output matrix CK (ncon x n, F-order)\n" \
    "  dk (ndarray): Controller feedthrough DK (ncon x nmeas, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (ac, bc, cc, dc, info): Closed-loop matrices and exit code\n" \
    "    info=0: success, 1: I-D22*DK singular, 2: I-DK*D22 singular"

#define DOC_SB10MD "D-step of D-K iteration for mu-synthesis (continuous-time).\n" \
    "\n" \
    "Estimates structured singular value mu(jw) at given frequencies and\n" \
    "optionally fits state-space realizations of D-scaling matrices.\n" \
    "\n" \
    "Parameters:\n" \
    "  nc (int): State dimension of closed-loop system\n" \
    "  mp (int): Number of rows in D11 block\n" \
    "  lendat (int): Number of frequency points (>= 2 if QUTOL >= 0)\n" \
    "  f (int): 1 = return D fitting function value, 0 = do not\n" \
    "  ord (ndarray): Preferred orders for D-scaling systems (modified on exit)\n" \
    "  nblock (ndarray): Block structure definition\n" \
    "  itype (ndarray): Block types (1=real, 2=complex)\n" \
    "  qutol (float): <0 = estimate mu only, >=0 = also fit D-scaling systems\n" \
    "  a (ndarray): NC-by-NC closed-loop state matrix (F-order)\n" \
    "  b (ndarray): NC-by-MP input matrix (F-order)\n" \
    "  c (ndarray): MP-by-NC output matrix (F-order)\n" \
    "  d (ndarray): MP-by-MP feedthrough matrix (F-order)\n" \
    "  omega (ndarray): LENDAT frequency values (rad/s)\n" \
    "\n" \
    "Returns:\n" \
    "  (totord, ad, bd, cd, dd, mju, f_value, info): Total order of D-scaling,\n" \
    "    state-space realization matrices, mu estimates, fitting function value,\n" \
    "    and exit code"

#define DOC_SB10PD "Normalize system for H-infinity controller design.\n" \
    "\n" \
    "Reduces D12 and D21 to unit diagonal form and transforms B, C, D11.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): System order\n" \
    "  m (int): Number of inputs\n" \
    "  np (int): Number of outputs\n" \
    "  ncon (int): Number of control inputs (M2)\n" \
    "  nmeas (int): Number of measurements (NP2)\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (np x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (np x m, F-order)\n" \
    "  tu (ndarray): Control transform TU (ncon x ncon, F-order, output)\n" \
    "  ty (ndarray): Measurement transform TY (nmeas x nmeas, F-order, output)\n" \
    "  tol (float, optional): Tolerance for rank tests. Default sqrt(eps).\n" \
    "\n" \
    "Returns:\n" \
    "  (b, c, d, tu, ty, rcond, info): Transformed matrices, transforms, cond numbers, exit code"

#define DOC_SB10RD "Compute H-infinity controller from state feedback and output injection.\n" \
    "\n" \
    "Computes controller K = (AK, BK, CK, DK) from state feedback F and\n" \
    "output injection H.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): System order\n" \
    "  m (int): Number of inputs\n" \
    "  np (int): Number of outputs\n" \
    "  ncon (int): Number of control inputs (M2)\n" \
    "  nmeas (int): Number of measurements (NP2)\n" \
    "  gamma (float): H-infinity norm bound\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (np x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (np x m, F-order)\n" \
    "  f (ndarray): State feedback matrix F (m x n, F-order)\n" \
    "  h (ndarray): Output injection matrix H (n x nmeas, F-order)\n" \
    "  tu (ndarray): Control transform TU (ncon x ncon, F-order)\n" \
    "  ty (ndarray): Measurement transform TY (nmeas x nmeas, F-order)\n" \
    "  x (ndarray): X-Riccati solution (n x n, F-order)\n" \
    "  y (ndarray): Y-Riccati solution (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (ak, bk, ck, dk, info): Controller matrices and exit code"

#define DOC_SB10SD "H2 optimal controller for normalized discrete-time systems.\n" \
    "\n" \
    "Computes the H2 optimal controller K = (AK, BK, CK, DK) for a normalized\n" \
    "discrete-time system obtained from SB10PD. Solves two discrete Riccati\n" \
    "equations with condition estimates.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): System order\n" \
    "  m (int): Number of inputs\n" \
    "  np (int): Number of outputs\n" \
    "  ncon (int): Number of control inputs (M2)\n" \
    "  nmeas (int): Number of measurements (NP2)\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (np x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (np x m, F-order)\n" \
    "  tol (float, optional): Tolerance for singularity tests. Default sqrt(eps).\n" \
    "\n" \
    "Returns:\n" \
    "  (ak, bk, ck, dk, x, y, rcond, info): Controller matrices AK (n x n),\n" \
    "    BK (n x nmeas), CK (ncon x n), DK (ncon x nmeas), Riccati solutions\n" \
    "    X and Y (n x n), condition estimates rcond(4), exit code"

#define DOC_SB10TD "Transform H2 controller from normalized to original system.\n" \
    "\n" \
    "Computes the H2 optimal discrete-time controller K = (AK, BK, CK, DK)\n" \
    "from the controller for the normalized system (as determined by SB10SD).\n" \
    "\n" \
    "Applies the transformations:\n" \
    "  BKHAT = BK * TY\n" \
    "  CKHAT = TU * CK\n" \
    "  DKHAT = TU * DK * TY\n" \
    "  CK = inv(I + DKHAT*D22) * CKHAT\n" \
    "  DK = inv(I + DKHAT*D22) * DKHAT\n" \
    "  AK = AK - BKHAT*D22*CK\n" \
    "  BK = BKHAT - BKHAT*D22*DK\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): System order\n" \
    "  m (int): Number of inputs\n" \
    "  np (int): Number of outputs\n" \
    "  ncon (int): Number of control inputs (M2)\n" \
    "  nmeas (int): Number of measurements (NP2)\n" \
    "  d (ndarray): Feedthrough matrix D (np x m, F-order). Only D22 used.\n" \
    "  tu (ndarray): Control transform TU (ncon x ncon, F-order)\n" \
    "  ty (ndarray): Measurement transform TY (nmeas x nmeas, F-order)\n" \
    "  ak (ndarray): Controller state matrix from SB10SD (n x n, F-order, in/out)\n" \
    "  bk (ndarray): Controller input matrix from SB10SD (n x nmeas, F-order, in/out)\n" \
    "  ck (ndarray): Controller output matrix from SB10SD (ncon x n, F-order, in/out)\n" \
    "  dk (ndarray): Controller feedthrough from SB10SD (ncon x nmeas, F-order, in/out)\n" \
    "  tol (float, optional): Tolerance for singularity test. Default sqrt(eps).\n" \
    "\n" \
    "Returns:\n" \
    "  (ak, bk, ck, dk, rcond, info): Transformed controller matrices,\n" \
    "    reciprocal condition number of (I + DKHAT*D22), exit code\n" \
    "    info=0: success, 1: (I + DKHAT*D22) singular or ill-conditioned"

#define DOC_SB10UD "Normalize D12 and D21 for H2 controller design.\n" \
    "\n" \
    "Reduces D12 and D21 matrices of a partitioned system to unit diagonal\n" \
    "form and transforms B and C matrices for H2 optimal controller computation.\n" \
    "Computes transformation matrices TU and TY.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): State dimension (>= 0)\n" \
    "  m (int): Total inputs (>= 0)\n" \
    "  np (int): Total outputs (>= 0)\n" \
    "  ncon (int): Control inputs M2 (0 <= NCON <= M)\n" \
    "  nmeas (int): Measured outputs NP2 (0 <= NMEAS <= NP)\n" \
    "  b (ndarray): N-by-M input matrix, modified on exit (F-order)\n" \
    "  c (ndarray): NP-by-N output matrix, modified on exit (F-order)\n" \
    "  d (ndarray): NP-by-M feedthrough matrix, modified on exit (F-order)\n" \
    "  tol (float, optional): Tolerance for rank determination (default: eps)\n" \
    "\n" \
    "Returns:\n" \
    "  (tu, ty, rcond, info):\n" \
    "    tu: M2-by-M2 control input transformation\n" \
    "    ty: NP2-by-NP2 measured output transformation\n" \
    "    rcond: Reciprocal condition numbers [rcond_D12, rcond_D21]\n" \
    "    info: 0=success, 1=D12 rank deficient, 2=D21 rank deficient, 3=both"

#define DOC_SB10VD "Compute state feedback and output injection for H2 controller.\n" \
    "\n" \
    "Solves X-Riccati and Y-Riccati equations for H2 optimal control.\n" \
    "\n" \
    "Parameters:\n" \
    "  ncon (int): Number of control inputs (M2)\n" \
    "  nmeas (int): Number of measurements (NP2)\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (np x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (f, h, x, y, xcond, ycond, info): State feedback F, output injection H,\n" \
    "    X-Riccati solution, Y-Riccati solution, condition estimates, exit code"

#define DOC_SB10WD "Compute H2 optimal controller from state feedback and output injection.\n" \
    "\n" \
    "Computes controller K = (AK, BK, CK, DK) from state feedback F and\n" \
    "output injection H as determined by SB10VD.\n" \
    "\n" \
    "Formulas:\n" \
    "  AK = A + H*C2 + B2*F + H*D22*F\n" \
    "  BK = -H*TY\n" \
    "  CK = TU*F\n" \
    "  DK = 0\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): System order\n" \
    "  m (int): Number of inputs\n" \
    "  np (int): Number of outputs\n" \
    "  ncon (int): Number of control inputs (M2)\n" \
    "  nmeas (int): Number of measurements (NP2)\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (np x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix D (np x m, F-order)\n" \
    "  f (ndarray): State feedback matrix F (ncon x n, F-order)\n" \
    "  h (ndarray): Output injection matrix H (n x nmeas, F-order)\n" \
    "  tu (ndarray): Control transform TU (ncon x ncon, F-order)\n" \
    "  ty (ndarray): Measurement transform TY (nmeas x nmeas, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (ak, bk, ck, dk, info): Controller matrices and exit code"

#define DOC_SB10YD "SB10YD: Frequency response fitting"

#define DOC_SB10ZD "Positive feedback controller for discrete-time system (D != 0).\n" \
    "\n" \
    "Computes the matrices of the positive feedback controller\n" \
    "K = [Ak, Bk; Ck, Dk] for the shaped plant G = [A, B; C, D]\n" \
    "in the Discrete-Time Loop Shaping Design Procedure.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Order of the plant (N >= 0)\n" \
    "  m (int): Column size of matrix B (M >= 0)\n" \
    "  np (int): Row size of matrix C (NP >= 0)\n" \
    "  a (ndarray): N-by-N system state matrix (F-order)\n" \
    "  b (ndarray): N-by-M system input matrix (F-order)\n" \
    "  c (ndarray): NP-by-N system output matrix (F-order)\n" \
    "  d (ndarray): NP-by-M feedthrough matrix (F-order)\n" \
    "  factor (float): Performance factor (=1 optimal, >1 suboptimal)\n" \
    "  tol (float): Tolerance for singularity tests (<=0 uses sqrt(eps))\n" \
    "\n" \
    "Returns:\n" \
    "  (ak, bk, ck, dk, rcond, info): Controller matrices (N-by-N, N-by-NP,\n" \
    "    M-by-N, M-by-NP), array of 6 reciprocal condition numbers, and exit code.\n" \
    "    info: 0=success, 1-2=Riccati failed, 3=iteration failed,\n" \
    "    4-9=singular matrices, 10=closed-loop unstable"

#define DOC_SB10ZP "Transform SISO system to stable and minimum phase.\n" \
    "\n" \
    "Computes a stable and minimum phase SISO system (Ao,Bo,Co,Do) from\n" \
    "a given SISO system (A,B,C,D). Unstable poles and zeros in the\n" \
    "right half-plane (continuous) or outside unit circle (discrete)\n" \
    "are reflected to stable locations.\n" \
    "\n" \
    "Parameters:\n" \
    "  discfl (int): 0 = continuous-time, 1 = discrete-time\n" \
    "  a (ndarray): N-by-N state matrix (F-order)\n" \
    "  b (ndarray): N input vector (F-order)\n" \
    "  c (ndarray): N output vector (F-order)\n" \
    "  d (ndarray): Scalar feedthrough (1-element array)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, c, d, n, info): Transformed system matrices, final order,\n" \
    "    and exit code (0=success, <0=param error, >0=algorithm error)"

#define DOC_SB16AD "Frequency-weighted controller reduction via balancing.\n" \
    "\n" \
    "Computes a reduced order controller (Acr,Bcr,Ccr,Dcr) for an original\n" \
    "state-space controller (Ac,Bc,Cc,Dc) using frequency-weighted B&T or SPA.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' continuous-time, 'D' discrete-time\n" \
    "  jobc (str): 'S' Enns', 'E' enhanced controllability Grammian\n" \
    "  jobo (str): 'S' Enns', 'E' enhanced observability Grammian\n" \
    "  jobmr (str): 'B' sqrt B&T, 'F' bal-free B&T, 'S' sqrt SPA, 'P' bal-free SPA\n" \
    "  weight (str): 'N' none, 'O' output, 'I' input, 'P' performance weighting\n" \
    "  equil (str): 'S' scale, 'N' no scaling\n" \
    "  ordsel (str): 'F' fixed order NCR, 'A' automatic selection\n" \
    "  n (int): Order of open-loop system (>= 0)\n" \
    "  m (int): Number of inputs (>= 0)\n" \
    "  p (int): Number of outputs (>= 0)\n" \
    "  nc (int): Order of controller (>= 0)\n" \
    "  ncr (int): Desired reduced order (ORDSEL='F')\n" \
    "  alpha (float): Stability boundary\n" \
    "  a (ndarray): N-by-N open-loop state matrix (F-order)\n" \
    "  b (ndarray): N-by-M open-loop input matrix (F-order)\n" \
    "  c (ndarray): P-by-N open-loop output matrix (F-order)\n" \
    "  d (ndarray): P-by-M open-loop feedthrough matrix (F-order)\n" \
    "  ac (ndarray): NC-by-NC controller state matrix (F-order)\n" \
    "  bc (ndarray): NC-by-P controller input matrix (F-order)\n" \
    "  cc (ndarray): M-by-NC controller output matrix (F-order)\n" \
    "  dc (ndarray): M-by-P controller feedthrough matrix (F-order)\n" \
    "  tol1 (float): Tolerance for order selection (ORDSEL='A')\n" \
    "  tol2 (float): Tolerance for minimal realization\n" \
    "\n" \
    "Returns:\n" \
    "  (acr, bcr, ccr, dcr, ncr_out, ncs, hsvc, iwarn, info): Reduced controller,\n" \
    "    actual order, stable part order, Hankel SVs, warning, exit code.\n" \
    "    info: 0=success, 1-7=algorithm errors"

#define DOC_SB16AY "Cholesky factors of frequency-weighted Grammians.\n" \
    "\n" \
    "Computes Cholesky factors Su and Ru of controllability Grammian P=Su*Su'\n" \
    "and observability Grammian Q=Ru'*Ru for frequency-weighted model reduction\n" \
    "of a controller (Ac,Bc,Cc,Dc) for plant (A,B,C,D).\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' continuous-time, 'D' discrete-time\n" \
    "  jobc (str): 'S' standard Grammian, 'E' enhanced Grammian\n" \
    "  jobo (str): 'S' standard Grammian, 'E' enhanced Grammian\n" \
    "  weight (str): 'N' none, 'O' left (output), 'I' right (input), 'P' both\n" \
    "  n (int): Plant state dimension (>= 0)\n" \
    "  m (int): Plant inputs (>= 0)\n" \
    "  p (int): Plant outputs (>= 0)\n" \
    "  nc (int): Controller state dimension (>= 0)\n" \
    "  ncs (int): Dimension of stable part (0 <= NCS <= NC)\n" \
    "  a (ndarray): N-by-N plant state matrix (F-order)\n" \
    "  b (ndarray): N-by-M plant input matrix (F-order)\n" \
    "  c (ndarray): P-by-N plant output matrix (F-order)\n" \
    "  d (ndarray): P-by-M plant feedthrough matrix (F-order)\n" \
    "  ac (ndarray): NC-by-NC controller state matrix (F-order)\n" \
    "  bc (ndarray): NC-by-P controller input matrix (F-order)\n" \
    "  cc (ndarray): M-by-NC controller output matrix (F-order)\n" \
    "  dc (ndarray): M-by-P controller feedthrough matrix (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (scalec, scaleo, s, r, info):\n" \
    "    scalec: Scale factor for controllability Grammian\n" \
    "    scaleo: Scale factor for observability Grammian\n" \
    "    s: NCS-by-NCS Cholesky factor Su\n" \
    "    r: NCS-by-NCS Cholesky factor Ru\n" \
    "    info: 0=success, 1=Lyapunov failed, 2=not stable"

#define DOC_SB16BD "Coprime factorization based state feedback controller reduction.\n" \
    "\n" \
    "Computes a reduced order controller (Ac,Bc,Cc,Dc) for a given open-loop\n" \
    "model (A,B,C,D) with state feedback gain F and observer gain G using\n" \
    "coprime factorization based model reduction methods (B&T or SPA).\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' continuous-time, 'D' discrete-time\n" \
    "  jobd (str): 'D' D matrix present, 'Z' D is zero\n" \
    "  jobmr (str): 'B' sqrt B&T, 'F' bal-free B&T, 'S' sqrt SPA, 'P' bal-free SPA\n" \
    "  jobcf (str): 'L' left coprime, 'R' right coprime factorization\n" \
    "  equil (str): 'S' perform scaling, 'N' no scaling\n" \
    "  ordsel (str): 'F' fixed order NCR, 'A' automatic order selection\n" \
    "  n (int): Order of open-loop system (>= 0)\n" \
    "  m (int): Number of inputs (>= 0)\n" \
    "  p (int): Number of outputs (>= 0)\n" \
    "  ncr (int): Desired controller order (0 <= NCR <= N, for ORDSEL='F')\n" \
    "  a (ndarray): N-by-N state matrix (F-order)\n" \
    "  b (ndarray): N-by-M input matrix (F-order)\n" \
    "  c (ndarray): P-by-N output matrix (F-order)\n" \
    "  d (ndarray): P-by-M feedthrough matrix (F-order)\n" \
    "  f (ndarray): M-by-N state feedback gain (F-order)\n" \
    "  g (ndarray): N-by-P observer gain (F-order)\n" \
    "  tol1 (float): Tolerance for order selection (ORDSEL='A'), 0 = default\n" \
    "  tol2 (float): Tolerance for minimal realization, 0 = default\n" \
    "\n" \
    "Returns:\n" \
    "  (ac, bc, cc, dc, hsv, ncr, iwarn, info): Reduced controller matrices,\n" \
    "    Hankel singular values, actual controller order, warning flag, exit code.\n" \
    "    info=0: success, 1: Schur of A+GC failed, 2: A+GC not stable,\n" \
    "    3: HSV computation failed, 4: Schur of A+BF failed, 5: A+BF not stable."

#define DOC_SB16CD "Frequency-weighted coprime factorization controller reduction.\n" \
    "\n" \
    "Computes reduced order controller (Ac,Bc,Cc) for open-loop model (A,B,C,D)\n" \
    "with state feedback gain F and observer gain G using coprime factorization\n" \
    "with frequency-weighted B&T model reduction.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' continuous-time, 'D' discrete-time\n" \
    "  jobd (str): 'D' D matrix present, 'Z' D is zero\n" \
    "  jobmr (str): 'B' sqrt B&T, 'F' balancing-free sqrt B&T\n" \
    "  jobcf (str): 'L' left coprime, 'R' right coprime factorization\n" \
    "  ordsel (str): 'F' fixed order NCR, 'A' automatic order selection\n" \
    "  a (ndarray): N-by-N state matrix (F-order)\n" \
    "  b (ndarray): N-by-M input matrix (F-order)\n" \
    "  c (ndarray): P-by-N output matrix (F-order)\n" \
    "  d (ndarray): P-by-M feedthrough matrix (F-order)\n" \
    "  f (ndarray): M-by-N state feedback gain (F-order)\n" \
    "  g (ndarray): N-by-P observer gain (F-order)\n" \
    "  ncr (int): Desired controller order (0 <= NCR <= N, for ORDSEL='F')\n" \
    "  tol (float): Tolerance for order selection (ORDSEL='A'), <=0 = default\n" \
    "\n" \
    "Returns:\n" \
    "  (ac, bc, cc, hsv, ncr, iwarn, info): Reduced controller matrices Ac, Bc, Cc\n" \
    "    (in modified a, g, f), Hankel singular values, actual order, warning, exit code.\n" \
    "    iwarn=1: NCR > minimal order, =2: repeated singular values at cutoff.\n" \
    "    info=0: success, 1: eigenvalue failure, 2: A+GC unstable, 3: A+BF unstable,\n" \
    "    4: obs Lyapunov singular, 5: ctrl Lyapunov singular, 6: HSV computation failed."

#define DOC_SB16CY "Cholesky factors of Grammians for coprime factors of controller.\n" \
    "\n" \
    "Computes Cholesky factors Su and Ru of controllability Grammian P=Su*Su'\n" \
    "and observability Grammian Q=Ru'*Ru for frequency-weighted model reduction\n" \
    "of coprime factors of state-feedback controller.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' continuous-time, 'D' discrete-time\n" \
    "  jobcf (str): 'L' left coprime, 'R' right coprime factorization\n" \
    "  n (int): Order of open-loop system (>= 0)\n" \
    "  m (int): Number of inputs (>= 0)\n" \
    "  p (int): Number of outputs (>= 0)\n" \
    "  a (ndarray): N-by-N state matrix (F-order)\n" \
    "  b (ndarray): N-by-M input matrix (F-order)\n" \
    "  c (ndarray): P-by-N output matrix (F-order)\n" \
    "  f (ndarray): M-by-N state feedback gain (F-order)\n" \
    "  g (ndarray): N-by-P observer gain (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (s, r, scalec, scaleo, info): Cholesky factors S and R,\n" \
    "    scaling factors, and exit code.\n" \
    "    info=0: success, 1: eigenvalue computation failed,\n" \
    "    2: A+GC not stable, 3: A+BF not stable,\n" \
    "    4: observability Lyapunov singular, 5: controllability Lyapunov singular."

#define DOC_SG02AD "Solve generalized algebraic Riccati equation for descriptor systems.\n" \
    "\n" \
    "Solves for X either:\n" \
    "  Continuous: Q + A'XE + E'XA - (L+E'XB)R^(-1)(L+E'XB)' = 0\n" \
    "  Discrete:   E'XE = A'XA - (L+A'XB)(R+B'XB)^(-1)(L+A'XB)' + Q\n" \
    "\n" \
    "Uses deflating subspaces with generalized Schur decomposition.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' continuous-time, 'D' discrete-time\n" \
    "  jobb (str): 'B' matrices B,R given, 'G' matrix G given\n" \
    "  fact (str): 'N' Q,R given, 'C' Q=C'C, 'D' R=D'D, 'B' both factored\n" \
    "  uplo (str): 'U' upper triangle, 'L' lower triangle\n" \
    "  jobl (str): 'Z' L is zero, 'N' L is nonzero\n" \
    "  scal (str): 'G' use scaling, 'N' no scaling\n" \
    "  sort (str): 'S' stable eigenvalues first, 'U' unstable first\n" \
    "  acc (str): 'R' iterative refinement, 'N' no refinement\n" \
    "  n (int): State dimension (>= 0)\n" \
    "  m (int): Number of inputs (>= 0, for JOBB='B')\n" \
    "  p (int): Number of outputs (for FACT='C','D','B')\n" \
    "  a (ndarray): N-by-N state matrix A (F-order)\n" \
    "  e (ndarray): N-by-N descriptor matrix E (F-order)\n" \
    "  b (ndarray): N-by-M input matrix B (F-order)\n" \
    "  q (ndarray): Weighting matrix Q or C (F-order)\n" \
    "  r (ndarray): Weighting matrix R or D (F-order)\n" \
    "  l (ndarray): Cross-weighting L (F-order)\n" \
    "  tol (float): Tolerance (<=0 uses eps)\n" \
    "\n" \
    "Returns:\n" \
    "  (x, rcondu, alfar, alfai, beta, s, t, u, iwarn, info)"

#define DOC_SG02CV "Compute residual matrix for Lyapunov equation.\n" \
    "\n" \
    "Computes residual matrix R for continuous or discrete-time Lyapunov equations:\n" \
    "  Continuous (JOBE='I'): R = op(A)'*X + X*op(A) + Q\n" \
    "  Continuous (JOBE='G'): R = op(A)'*X*op(E) + op(E)'*X*op(A) + Q\n" \
    "  Discrete   (JOBE='I'): R = op(A)'*X*op(A) - X + Q\n" \
    "  Discrete   (JOBE='G'): R = op(A)'*X*op(A) - op(E)'*X*op(E) + Q\n" \
    "\n" \
    "where X and Q are symmetric, A is upper real Schur, E is upper triangular.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' continuous-time, 'D' discrete-time\n" \
    "  job (str): 'R' residual only, 'N' or 'B' residual and norms\n" \
    "  jobe (str): 'G' general E, 'I' identity E (E not referenced)\n" \
    "  uplo (str): 'U' upper triangular, 'L' lower triangular\n" \
    "  trans (str): 'N' for op(W)=W, 'T' or 'C' for op(W)=W'\n" \
    "  a (ndarray): N-by-N upper real Schur matrix (F-order)\n" \
    "  e (ndarray or None): N-by-N upper triangular E (if jobe='G')\n" \
    "  x (ndarray): N-by-N symmetric matrix X (F-order)\n" \
    "  r (ndarray): N-by-N matrix Q on entry, R on exit (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (r, norms, info): Residual matrix R, Frobenius norms array, exit code.\n" \
    "    norms[0] = ||op(A)'*X*op(A)|| or ||op(A)'*X*op(E)|| depending on equation.\n" \
    "    norms[1] = ||op(E)'*X*op(E)|| (discrete, JOBE='G' only).\n" \
    "    info=0: success, <0: parameter error."

#define DOC_SG02CW "Compute residual matrix for Riccati equation with line search.\n" \
    "\n" \
    "Evaluates the residual matrix R for the generalized continuous-time or\n" \
    "discrete-time algebraic Riccati equation with optional computation of the\n" \
    "closed-loop system matrix C and norms. Supports multiple matrix product forms.\n" \
    "\n" \
    "For continuous-time (DICO='C'):\n" \
    "  R = op(A)'*X*op(E) + op(E)'*X*op(A) +/- op(E)'*X*G*X*op(E) + Q\n" \
    "\n" \
    "For discrete-time (DICO='D'):\n" \
    "  R = op(A)'*X*op(A) - op(E)'*X*op(E) +/- op(E)'*X*G*X*op(E) + Q\n" \
    "\n" \
    "where X is symmetric and the +/- depends on FLAG parameter.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' continuous-time, 'D' discrete-time\n" \
    "  job (str): 'R' residual only, 'A' residual and closed-loop C,\n" \
    "             'N' norms only, 'B' residual and norms\n" \
    "  jobe (str): 'G' general E, 'I' identity E\n" \
    "  flag (str): 'M' minus sign, 'P' plus sign in X*G*X term\n" \
    "  jobg (str): 'G' use G directly, 'D' G = B*R^(-1)*B',\n" \
    "              'F' G = F'*R*F, 'H' G = (K-F)*R^(-1)*(K-F)',\n" \
    "              'K' G = K'*R^(-1)*K\n" \
    "  uplo (str): 'U' upper triangular, 'L' lower triangular\n" \
    "  trans (str): 'N' for op(M)=M, 'T' or 'C' for op(M)=M'\n" \
    "  n (int): State dimension (>= 0)\n" \
    "  m (int): Number of control inputs (>= 0)\n" \
    "  a (ndarray): N-by-N state matrix (F-order)\n" \
    "  e (ndarray or None): N-by-N descriptor matrix (if jobe='G')\n" \
    "  g (ndarray): N-by-N or N-by-M matrix G or B (F-order)\n" \
    "  x (ndarray): N-by-N symmetric matrix X (F-order)\n" \
    "  f (ndarray or None): M-by-N matrix F (if jobg='F')\n" \
    "  k (ndarray or None): M-by-N matrix K (if jobg='H' or 'K')\n" \
    "  xe (ndarray or None): N-by-N matrix X*E (discrete, jobe='G')\n" \
    "  q (ndarray): N-by-N matrix Q on entry, R on exit (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (r, c, norms, info): Residual R, closed-loop C, norms array, exit code.\n" \
    "    c[i,j] = (A - G*X)[i,j] for continuous, (A - G*X*E^(-1)*A)[i,j] for discrete.\n" \
    "    norms[0] = ||R_low||_F / ||R_high||_F (residual / matrix norm).\n" \
    "    norms[1] = ||op(A)'*X*op(E)|| for continuous.\n" \
    "    info=0: success, <0: parameter error."

#define DOC_SG02CX "Find line search parameter minimizing Riccati residual norm.\n" \
    "\n" \
    "Finds alpha minimizing ||R(X + alpha*S)||_F where R(X) is the residual\n" \
    "of a (generalized) continuous-time algebraic Riccati equation.\n" \
    "\n" \
    "The function evaluates P(alpha) = ||(1-alpha)*R(X) +/- alpha^2*V||_F\n" \
    "where V = op(E)'*S*G*S*op(E) and S is the Newton step.\n" \
    "\n" \
    "Algorithm: Sets up a cubic polynomial P'(alpha) = 0 to find critical\n" \
    "points, then solves a 3x3 generalized eigenproblem via MC01XD.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobe (str): 'G' general E matrix, 'I' identity E\n" \
    "  flag (str): 'P' plus sign, 'M' minus sign in residual formula\n" \
    "  jobg (str): 'G' G given, 'D' D given (G=D*D'),\n" \
    "              'F' F given (V=F*F'), 'H' H,K given (V=H*K)\n" \
    "  uplo (str): 'U' upper triangle, 'L' lower triangle stored\n" \
    "  trans (str): 'N' op(W)=W, 'T'/'C' op(W)=W'\n" \
    "  n (int): Order of matrices E, R, S (>= 0)\n" \
    "  m (int): Columns in D/F/K' (>= 0 if jobg != 'G')\n" \
    "  e (ndarray): N-by-N matrix E (if jobe='G' and jobg='G' or 'D')\n" \
    "  r (ndarray): N-by-N symmetric residual R(X)\n" \
    "  s (ndarray): Newton step S; N-by-N symmetric if jobg='G'/'D'\n" \
    "  g (ndarray): Matrix G/D/F/H depending on jobg\n" \
    "\n" \
    "Returns:\n" \
    "  (alpha, rnorm, iwarn, info): Optimal alpha in [0,2], residual norm,\n" \
    "    warning code (2 = no optimum in [0,2]), exit code.\n" \
    "    info=0: success, <0: parameter error, 1: eigenproblem failed."

#define DOC_SG02ND "Compute optimal gain matrix for discrete/continuous Riccati problems.\n" \
    "\n" \
    "Computes the optimal gain matrix K for optimal control:\n" \
    "\n" \
    "Discrete-time:   K = (R + B'XB)^{-1} (B'X*op(A) + L')\n" \
    "Continuous-time: K = R^{-1} (B'X*op(E) + L')\n" \
    "\n" \
    "where X is the solution of the corresponding algebraic Riccati equation.\n" \
    "\n" \
    "R may be given in factored form. Optionally returns H = op(A)'XB + L\n" \
    "(discrete) or H = op(E)'XB + L (continuous), or F where FC = H.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' continuous-time, 'D' discrete-time\n" \
    "  jobe (str): 'G' general E matrix, 'I' identity E (only for DICO='C')\n" \
    "  job (str): 'K' K only, 'H' H and K, 'F' F if possible else H and K,\n" \
    "             'D'/'C' with pre-transformed B,L\n" \
    "  jobx (str): 'C' compute X*E or X*A, 'N' do not compute\n" \
    "  fact (str): 'N' R not factored, 'D' R=D'D, 'C' Cholesky, 'U' UdU'/LdL'\n" \
    "  uplo (str): 'U' upper triangle, 'L' lower triangle\n" \
    "  jobl (str): 'Z' L is zero, 'N' L is nonzero\n" \
    "  trans (str): 'N' op(W)=W, 'T'/'C' op(W)=W'\n" \
    "  n (int): Order of A and X (>= 0)\n" \
    "  m (int): Order of R and columns of B,L (>= 0)\n" \
    "  p (int): Rows of D (relevant for FACT='D')\n" \
    "  a (ndarray): N-by-N state matrix A (for DICO='D')\n" \
    "  e (ndarray): N-by-N matrix E (for DICO='C', JOBE='G')\n" \
    "  b (ndarray): N-by-M input matrix B\n" \
    "  r (ndarray): M-by-M matrix R (factored or not)\n" \
    "  ipiv (ndarray): Pivot indices (for FACT='U')\n" \
    "  l (ndarray): N-by-M cross-weighting L (for JOBL='N')\n" \
    "  x (ndarray): N-by-N solution matrix X\n" \
    "  rnorm (float): 1-norm of R (for FACT='U')\n" \
    "\n" \
    "Returns:\n" \
    "  (k, h, xe, oufact, rcond, info): M-by-N gain K, N-by-M matrix H or F,\n" \
    "    N-by-N product X*E or X*A, factorization info [2], reciprocal\n" \
    "    condition number, exit code.\n" \
    "    info=0: success, <0: parameter error, i: d[i]=0, m+1: singular,\n" \
    "    m+2: eigenvalue failure, m+3: X indefinite."

#define DOC_SG03AD "Solve generalized Lyapunov equation for descriptor systems.\n" \
    "\n" \
    "Solves for X either:\n" \
    "  Continuous: op(A)'*X*op(E) + op(E)'*X*op(A) = scale*Y\n" \
    "  Discrete:   op(A)'*X*op(A) - op(E)'*X*op(E) = scale*Y\n" \
    "\n" \
    "where op(M) = M or M^T and Y is symmetric.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' continuous-time, 'D' discrete-time\n" \
    "  job (str): 'X' solution only, 'S' separation only, 'B' both\n" \
    "  fact (str): 'N' factorization needed, 'F' factorization supplied\n" \
    "  trans (str): 'N' op(M)=M, 'T' op(M)=M^T\n" \
    "  uplo (str): 'U' upper triangle, 'L' lower triangle\n" \
    "  n (int): Order of matrices (>= 0)\n" \
    "  a (ndarray): N-by-N matrix A (F-order)\n" \
    "  e (ndarray): N-by-N matrix E (F-order)\n" \
    "  x (ndarray): N-by-N RHS matrix Y (F-order)\n" \
    "  q (ndarray, optional): Orthogonal Q (if FACT='F')\n" \
    "  z (ndarray, optional): Orthogonal Z (if FACT='F')\n" \
    "\n" \
    "Returns:\n" \
    "  (x, scale, sep, ferr, alphar, alphai, beta, a, e, q, z, info)"

#define DOC_SG03AX "Solve reduced generalized discrete-time Lyapunov equation.\n" \
    "\n" \
    "Solves A'*X*A - E'*X*E = scale*Y (trans='N') or\n" \
    "       A*X*A' - E*X*E' = scale*Y (trans='T').\n" \
    "\n" \
    "Parameters:\n" \
    "  trans (str): 'N' or 'T' - equation form\n" \
    "  a (ndarray): Upper quasitriangular matrix (n x n, F-order)\n" \
    "  e (ndarray): Upper triangular matrix (n x n, F-order)\n" \
    "  x (ndarray): RHS matrix Y, overwritten with solution (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (X, scale, info): Solution X, scale factor, exit code"

#define DOC_SG03AY "Solve reduced generalized continuous-time Lyapunov equation.\n" \
    "\n" \
    "Solves A'*X*E + E'*X*A = scale*Y (trans='N') or\n" \
    "       A*X*E' + E*X*A' = scale*Y (trans='T').\n" \
    "\n" \
    "Parameters:\n" \
    "  trans (str): 'N' or 'T' - equation form\n" \
    "  a (ndarray): Upper quasitriangular matrix (n x n, F-order)\n" \
    "  e (ndarray): Upper triangular matrix (n x n, F-order)\n" \
    "  x (ndarray): RHS matrix Y, overwritten with solution (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (X, scale, info): Solution X, scale factor, exit code"

#define DOC_SG03BD "Solve generalized Lyapunov equation for Cholesky factor.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  fact (str): 'N' to compute factorization, 'F' if factorization supplied\n" \
    "  trans (str): 'N' for op(K)=K, 'T' for op(K)=K^T\n" \
    "  n (int): Order of matrices A and E\n" \
    "  m (int): Number of rows in op(B)\n" \
    "  a (ndarray): Matrix A (n x n, F-order)\n" \
    "  e (ndarray): Matrix E (n x n, F-order)\n" \
    "  b (ndarray): Matrix B (size depends on trans, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (u, scale, alphar, alphai, beta, info): Cholesky factor, scale, eigenvalues, exit code"

#define DOC_SG03BR "Compute complex Givens rotation in real arithmetic.\n" \
    "\n" \
    "Parameters:\n" \
    "  xr (float): Real part of X\n" \
    "  xi (float): Imaginary part of X\n" \
    "  yr (float): Real part of Y\n" \
    "  yi (float): Imaginary part of Y\n" \
    "\n" \
    "Returns:\n" \
    "  (c, sr, si, zr, zi): Givens rotation parameters and result"

#define DOC_SG03BS "Solve generalized discrete-time Lyapunov equation (complex Hammarling).\n" \
    "\n" \
    "Computes Cholesky factor U of X = U^H*U or X = U*U^H for:\n" \
    "  TRANS='N': A^H * X * A - E^H * X * E = -SCALE^2 * B^H * B\n" \
    "  TRANS='C': A * X * A^H - E * X * E^H = -SCALE^2 * B * B^H\n" \
    "\n" \
    "where A, E, B are complex upper triangular matrices.\n" \
    "The pencil A-lambda*E must be d-stable (eigenvalues inside unit circle).\n" \
    "\n" \
    "Parameters:\n" \
    "  trans (str): 'N' for equation (1), 'C' for equation (2)\n" \
    "  a (ndarray): Complex upper triangular A (n x n, F-order)\n" \
    "  e (ndarray): Complex upper triangular E (n x n, F-order)\n" \
    "  b (ndarray): Complex upper triangular B (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (u, scale, info): Cholesky factor U, scale factor, exit code\n" \
    "  info=0: success, info=3: not d-stable, info=4: ZSTEIN failed"

#define DOC_SG03BT "Solve generalized continuous-time Lyapunov equation (complex Hammarling).\n" \
    "\n" \
    "Computes Cholesky factor U of X = U^H*U or X = U*U^H for:\n" \
    "  TRANS='N': A^H * X * E + E^H * X * A = -SCALE^2 * B^H * B\n" \
    "  TRANS='C': A * X * E^H + E * X * A^H = -SCALE^2 * B * B^H\n" \
    "\n" \
    "where A, E, B are complex upper triangular matrices.\n" \
    "The pencil A-lambda*E must be c-stable (eigenvalues with negative real parts).\n" \
    "\n" \
    "Parameters:\n" \
    "  trans (str): 'N' for equation (1), 'C' for equation (2)\n" \
    "  a (ndarray): Complex upper triangular A (n x n, F-order)\n" \
    "  e (ndarray): Complex upper triangular E (n x n, F-order)\n" \
    "  b (ndarray): Complex upper triangular B (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (u, scale, info): Cholesky factor U, scale factor, exit code\n" \
    "  info=0: success, info=3: not c-stable"

#define DOC_SG03BU "Solve generalized discrete-time Lyapunov equation for Cholesky factor.\n" \
    "\n" \
    "Parameters:\n" \
    "  trans (str): 'N' for TRANS='N', 'T' for TRANS='T'\n" \
    "  a (ndarray): Quasitriangular matrix A (n x n, F-order)\n" \
    "  e (ndarray): Upper triangular matrix E (n x n, F-order)\n" \
    "  b (ndarray): Upper triangular matrix B (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (u, scale, info): Cholesky factor U, scale factor, exit code"

#define DOC_SG03BV "Solve generalized continuous-time Lyapunov equation for Cholesky factor.\n" \
    "\n" \
    "Parameters:\n" \
    "  trans (str): 'N' for TRANS='N', 'T' for TRANS='T'\n" \
    "  a (ndarray): Quasitriangular matrix A (n x n, F-order)\n" \
    "  e (ndarray): Upper triangular matrix E (n x n, F-order)\n" \
    "  b (ndarray): Upper triangular matrix B (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (u, scale, info): Cholesky factor U, scale factor, exit code"

#define DOC_SG03BW "Solve generalized Sylvester equation for small systems.\n" \
    "\n" \
    "Parameters:\n" \
    "  trans (str): 'N' or 'T' - equation form\n" \
    "  a (ndarray): Upper quasitriangular matrix (m x m, F-order)\n" \
    "  e (ndarray): Upper triangular matrix (m x m, F-order)\n" \
    "  c (ndarray): Matrix C (n x n, F-order)\n" \
    "  d (ndarray): Matrix D (n x n, F-order)\n" \
    "  x (ndarray): Right-hand side Y, overwritten with solution (m x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (x, scale, info): Solution matrix, scale factor, and exit code"

#define DOC_SG03BX "Solve 2x2 generalized Lyapunov equation.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  trans (str): 'N' for op(K)=K, 'T' for op(K)=K^T\n" \
    "  a (ndarray): Matrix A (2 x 2, F-order)\n" \
    "  e (ndarray): Upper triangular matrix E (2 x 2, F-order)\n" \
    "  b (ndarray): Upper triangular matrix B (2 x 2, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (u, scale, m1, m2, info): Cholesky factor, scale, auxiliary matrices, exit code"

#define DOC_SG03BY "Compute complex Givens rotation in real arithmetic.\n" \
    "\n" \
    "Computes parameters for the complex Givens rotation:\n" \
    "\n" \
    "    (  CR-CI*I   SR-SI*I )   ( XR+XI*I )   ( Z )\n" \
    "    (                    ) * (         ) = (   )\n" \
    "    ( -SR-SI*I   CR+CI*I )   ( YR+YI*I )   ( 0 )\n" \
    "\n" \
    "where CR, CI, SR, SI, XR, XI, YR, YI are real numbers and I is the\n" \
    "imaginary unit. Z is a non-negative real number.\n" \
    "\n" \
    "The routine avoids overflow using max-norm scaling.\n" \
    "\n" \
    "Parameters:\n" \
    "  xr (float): Real part of first complex input\n" \
    "  xi (float): Imaginary part of first complex input\n" \
    "  yr (float): Real part of second complex input\n" \
    "  yi (float): Imaginary part of second complex input\n" \
    "\n" \
    "Returns:\n" \
    "  (cr, ci, sr, si, z): Givens rotation parameters and result norm"

#define DOC_SG03BZ "Solve stable generalized continuous/discrete-time Lyapunov equation (complex).\n" \
    "\n" \
    "Computes the Cholesky factor U of X = op(U)^H * op(U), where X solves:\n" \
    "\n" \
    "Continuous-time (DICO='C'):\n" \
    "  op(A)^H * X * op(E) + op(E)^H * X * op(A) = -SCALE^2 * op(B)^H * op(B)\n" \
    "\n" \
    "Discrete-time (DICO='D'):\n" \
    "  op(A)^H * X * op(A) - op(E)^H * X * op(E) = -SCALE^2 * op(B)^H * op(B)\n" \
    "\n" \
    "where op(K) = K (TRANS='N') or op(K) = K^H (TRANS='C').\n" \
    "\n" \
    "The pencil A - lambda*E must be c-stable (continuous, Re(eig) < 0) or\n" \
    "d-stable (discrete, |eig| < 1).\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' for continuous-time, 'D' for discrete-time\n" \
    "  fact (str): 'N' compute Schur factorization, 'F' supplied\n" \
    "  trans (str): 'N' for op(K)=K, 'C' for op(K)=K^H\n" \
    "  a (ndarray): Complex matrix A (n x n, F-order)\n" \
    "  e (ndarray): Complex matrix E (n x n, F-order)\n" \
    "  q (ndarray): Unitary transformation Q (n x n, F-order)\n" \
    "  z (ndarray): Unitary transformation Z (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (m x n or n x m depending on TRANS)\n" \
    "\n" \
    "Returns:\n" \
    "  (u, scale, alpha, beta, info): Cholesky factor U, scale factor,\n" \
    "    eigenvalue numerators, eigenvalue denominators, exit code\n" \
    "  info=0: success, info=4: ZGGES failed, info=5: not c-stable,\n" \
    "  info=6: not d-stable, info=7: ZSTEIN failed"

#define DOC_TB01ID "Balance system matrices (A,B,C) using diagonal similarity.\n" \
    "\n" \
    "Reduces 1-norm of system matrix S=[A B; C 0] by balancing.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'A' all, 'B' B+A, 'C' C+A, 'N' A only\n" \
    "  a (ndarray): N-by-N state matrix A (F-order)\n" \
    "  b (ndarray): N-by-M input matrix B (F-order)\n" \
    "  c (ndarray): P-by-N output matrix C (F-order)\n" \
    "  maxred (float): Max reduction (<=0 uses 10.0)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, c, maxred, scale, info): Balanced matrices,\n" \
    "    norm ratio, scale factors, exit code"

#define DOC_TB01IZ "Balance complex system matrices (A,B,C) using diagonal similarity.\n" \
    "\n" \
    "Reduces 1-norm of system matrix S=[A B; C 0] by balancing.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'A' all, 'B' B+A, 'C' C+A, 'N' A only\n" \
    "  a (ndarray): N-by-N complex state matrix A (F-order)\n" \
    "  b (ndarray): N-by-M complex input matrix B (F-order)\n" \
    "  c (ndarray): P-by-N complex output matrix C (F-order)\n" \
    "  maxred (float): Max reduction (<=0 uses 10.0)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, c, maxred, scale, info): Balanced matrices,\n" \
    "    norm ratio, scale factors, exit code"

#define DOC_TB01KD "Stable/unstable decomposition of state-space system.\n" \
    "\n" \
    "Reduces (A,B,C) to block-diagonal form with eigenvalues in specified domain in leading block.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' continuous, 'D' discrete\n" \
    "  stdom (str): 'S' stable domain, 'U' unstable domain\n" \
    "  joba (str): 'S' A is Schur, 'G' A is general\n" \
    "  alpha (float): Domain boundary\n" \
    "  a (ndarray): State matrix (n x n)\n" \
    "  b (ndarray): Input matrix (n x m)\n" \
    "  c (ndarray): Output matrix (p x n)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, c, ndim, u, wr, wi, info): Block-diagonal A, transformed B/C, counts, eigenvalues"

#define DOC_TB01KX "Block-diagonalize state-space system with Schur form input.\n" \
    "\n" \
    "Computes additive spectral decomposition by block-diagonalizing A (in real Schur form) via Sylvester equation. The leading NDIM eigenvalues must be distinct from trailing eigenvalues.\n" \
    "\n" \
    "Transformation: A <- V*A*U, B <- V*B, C <- C*U, where V = inv(U)\n" \
    "\n" \
    "Parameters:\n" \
    "  ndim (int): Dimension of leading block (0 <= ndim <= n)\n" \
    "  a (ndarray): State matrix in Schur form (n x n, F-order)\n" \
    "  b (ndarray): Input matrix (n x m, F-order)\n" \
    "  c (ndarray): Output matrix (p x n, F-order)\n" \
    "  u (ndarray): Initial transformation matrix (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, c, u, v, info): Block-diagonal A, transformed B/C/U, inverse V, exit code\n" \
    "  info=0: success, info=1: Sylvester equation failed (close eigenvalues)"

#define DOC_TB01LD "Reduce state matrix to ordered Schur form.\n" \
    "\n" \
    "Reduces (A,B,C) so leading block of A has eigenvalues in specified domain.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' continuous, 'D' discrete\n" \
    "  stdom (str): 'S' stable domain, 'U' unstable domain\n" \
    "  joba (str): 'S' A is Schur, 'G' A is general\n" \
    "  alpha (float): Domain boundary\n" \
    "  a (ndarray): State matrix (n x n)\n" \
    "  b (ndarray): Input matrix (n x m)\n" \
    "  c (ndarray): Output matrix (p x n)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, c, ndim, u, wr, wi, info): Transformed matrices, counts, eigenvalues"

#define DOC_TB01MD "Reduce (B,A) to upper/lower controller Hessenberg form.\n" \
    "\n" \
    "Computes unitary transformation U to reduce (B,A) pair to controller\n" \
    "Hessenberg form. For UPLO='U', U'B is upper triangular and U'AU is\n" \
    "upper Hessenberg. For UPLO='L', the structure is reversed.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobu (str): 'N'=no U, 'I'=initialize U to identity, 'U'=update given U\n" \
    "  uplo (str): 'U'=upper Hessenberg, 'L'=lower Hessenberg\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  u (ndarray, optional): Transformation matrix (n x n), required if jobu='U'\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, u, info): Transformed A=U'AU, B=U'B, transformation U, exit code"

#define DOC_TB01ND "Reduce (A,C) to upper/lower observer Hessenberg form.\n" \
    "\n" \
    "Computes unitary transformation U to reduce (A,C) pair to observer\n" \
    "Hessenberg form. For UPLO='U', CU is upper triangular and U'AU is\n" \
    "upper Hessenberg. For UPLO='L', the structure is reversed.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobu (str): 'N'=no U, 'I'=initialize U to identity, 'U'=update given U\n" \
    "  uplo (str): 'U'=upper Hessenberg, 'L'=lower Hessenberg\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  c (ndarray): Output matrix C (p x n, F-order)\n" \
    "  u (ndarray, optional): Transformation matrix (n x n), required if jobu='U'\n" \
    "\n" \
    "Returns:\n" \
    "  (a, c, u, info): Transformed A=U'AU, C=CU, transformation U, exit code"

#define DOC_TB01PD "Compute minimal realization of state-space system.\n" \
    "\n" \
    "Reduces the linear time-invariant system (A,B,C) to a minimal\n" \
    "(controllable and observable) or controllable/observable form.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'M'=minimal, 'C'=controllable, 'O'=observable\n" \
    "  equil (str): 'S'=scale first, 'N'=no balancing\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (p x n, F-order)\n" \
    "  tol (float): Tolerance for rank determination (0 for default)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, c, nr, nblk, info):\n" \
    "    - a: Reduced state matrix (nr x nr block)\n" \
    "    - b: Reduced input matrix\n" \
    "    - c: Reduced output matrix\n" \
    "    - nr: Order of reduced system\n" \
    "    - nblk: Staircase block structure\n" \
    "    - info: Exit code (0 = success)"

#define DOC_TB01PX "Compute minimal/controllable/observable realization (variant).\n" \
    "\n" \
    "Finds a reduced (controllable, observable, or minimal) state-space\n" \
    "representation (Ar,Br,Cr) with Ar in upper block Hessenberg staircase form.\n" \
    "\n" \
    "Two-phase reduction:\n" \
    "  Phase 1 (job='M' or 'C'): Remove uncontrollable part\n" \
    "  Phase 2 (job='M' or 'O'): Remove unobservable part\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'M'=minimal, 'C'=controllable, 'O'=observable\n" \
    "  equil (str): 'S'=balance (scale) first, 'N'=no balancing\n" \
    "  n (int): Order of original system (n >= 0)\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  p (int): Number of outputs (p >= 0)\n" \
    "  a (ndarray): State matrix A (n x n, F-order)\n" \
    "  b (ndarray): Input matrix B (n x m, F-order)\n" \
    "  c (ndarray): Output matrix C (p x n, F-order)\n" \
    "  tol (float, optional): Tolerance for rank determination (default: 0.0)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, c, nr, infred, iwork, info):\n" \
    "    - a: Reduced state matrix (nr x nr block in upper block Hessenberg)\n" \
    "    - b: Reduced input matrix (nr x m)\n" \
    "    - c: Reduced output matrix (p x nr)\n" \
    "    - nr: Order of reduced system\n" \
    "    - infred: Reduction info (4-element array):\n" \
    "        [0]: Order reduction in Phase 1 (>=0) or -1 if not performed\n" \
    "        [1]: Order reduction in Phase 2 (>=0) or -1 if not performed\n" \
    "        [2]: Number of nonzero subdiagonals of Ar\n" \
    "        [3]: Number of blocks in staircase form\n" \
    "    - iwork: Block dimensions (infred[3] elements)\n" \
    "    - info: Exit code (0 = success)"

#define DOC_TB01TD "Balance state-space (A,B,C,D) by permutations and scalings.\n" \
    "\n" \
    "Reduces (A,B,C,D) to balanced form using state permutations and\n" \
    "state/input/output scalings. Uses DGEBAL for A, then scales B,C,D.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Order of A (n >= 0)\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  p (int): Number of outputs (p >= 0)\n" \
    "  a (ndarray): State matrix (n x n, F-order), modified in-place\n" \
    "  b (ndarray): Input matrix (n x m, F-order), modified in-place\n" \
    "  c (ndarray): Output matrix (p x n, F-order), modified in-place\n" \
    "  d (ndarray): Feedthrough matrix (p x m, F-order), modified in-place\n" \
    "\n" \
    "Returns:\n" \
    "  (low, igh, scstat, scin, scout, info):\n" \
    "    - low: Lower index of balanced submatrix (1-based)\n" \
    "    - igh: Upper index of balanced submatrix (1-based)\n" \
    "    - scstat: State transformation info (n elements)\n" \
    "    - scin: Input scalings (m elements)\n" \
    "    - scout: Output scalings (p elements)\n" \
    "    - info: Exit code (0 = success, <0 = invalid parameter)"

#define DOC_TB01TY "Balance rows/columns of a matrix block using integer powers of BASE.\n" \
    "\n" \
    "Scales each non-zero row (MODE=0) or column (MODE!=0) by BASE^IEXPT\n" \
    "so the 1-norm is in range (SIZE/BASE, SIZE]. No rounding errors.\n" \
    "\n" \
    "Parameters:\n" \
    "  mode (int): 0 = balance rows, != 0 = balance columns\n" \
    "  ioff (int): Row offset (0-based)\n" \
    "  joff (int): Column offset (0-based)\n" \
    "  nrow (int): Number of rows in block\n" \
    "  ncol (int): Number of columns in block\n" \
    "  size (float): Target 1-norm (absolute value used)\n" \
    "  x (ndarray): Matrix to balance (F-order), modified in-place\n" \
    "\n" \
    "Returns:\n" \
    "  (x, bvect, info): Balanced matrix, scale factors, exit code"

#define DOC_TB01UD "Reduce state-space system to controllability staircase form.\n" \
    "\n" \
    "Finds controllable realization via orthogonal similarity transformations.\n" \
    "The transformed system has Acont in upper block Hessenberg form.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobz (str): 'N'=no Z, 'F'=factored form, 'I'=identity initialized\n" \
    "  n (int): Order of state matrix A\n" \
    "  m (int): Number of inputs (columns of B)\n" \
    "  p (int): Number of outputs (rows of C)\n" \
    "  a (ndarray): State dynamics matrix (n x n, F-order)\n" \
    "  b (ndarray): Input matrix (n x m, F-order)\n" \
    "  c (ndarray): Output matrix (p x n, F-order)\n" \
    "  tol (float, optional): Tolerance for rank determination (default: 0.0)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, c, ncont, indcon, nblk, z, tau, info): Transformed system and metadata"

#define DOC_TB01UX "Reduce state-space system to observability staircase form.\n" \
    "\n" \
    "Computes orthogonal Z to decompose (A,B,C) into observable/unobservable parts:\n" \
    "  Z'*A*Z = [Ano *; 0 Ao], Z'*B = [Bno; Bo], C*Z = [0 Co]\n" \
    "where (Ao,Bo,Co) is observable with order NOBSV.\n" \
    "\n" \
    "Parameters:\n" \
    "  compz (str): 'N'=no Z, 'I'=identity initialized\n" \
    "  n (int): Order of state matrix A\n" \
    "  m (int): Number of inputs (columns of B)\n" \
    "  p (int): Number of outputs (rows of C)\n" \
    "  a (ndarray): State dynamics matrix (n x n, F-order)\n" \
    "  b (ndarray): Input matrix (n x m, F-order)\n" \
    "  c (ndarray): Output matrix (p x n, F-order)\n" \
    "  tol (float, optional): Tolerance for rank determination (default: 0.0)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, c, nobsv, nlblck, ctau, z, info): Transformed system and metadata"

#define DOC_TB01UY "Controllable realization for M1+M2 input system.\n" \
    "\n" \
    "Reduces system (A,[B1,B2],C) to controllability staircase form with\n" \
    "alternating rank detection between B1 (M1 columns) and B2 (M2 columns).\n" \
    "Produces orthogonal canonical form: Ac = Z'*A*Z, [Bc1,Bc2] = Z'*[B1,B2], Cc = C*Z.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobz (str): 'N'=no Z, 'F'=factored form, 'I'=identity initialized\n" \
    "  n (int): Order of state matrix A\n" \
    "  m1 (int): Number of B1 columns (first input group)\n" \
    "  m2 (int): Number of B2 columns (second input group)\n" \
    "  p (int): Number of outputs (rows of C)\n" \
    "  a (ndarray): State dynamics matrix (n x n, F-order)\n" \
    "  b (ndarray): Compound input matrix [B1,B2] (n x (m1+m2), F-order)\n" \
    "  c (ndarray): Output matrix (p x n, F-order)\n" \
    "  tol (float, optional): Tolerance for rank determination (default: 0.0)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, c, ncont, indcon, nblk, z, tau, info): Transformed system and metadata\n" \
    "    ncont: Controllable subsystem order\n" \
    "    indcon: Controllability index (always even)\n" \
    "    nblk: Block dimensions (2*n array, INDCON/2 odd+even pairs)"

#define DOC_TB01VD "Convert discrete-time system to output normal form.\n" \
    "\n" \
    "Converts a stable discrete-time system (A, B, C, D) with initial state x0\n" \
    "into the output normal form, producing parameter vector THETA.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): System order\n" \
    "  m (int): Number of inputs\n" \
    "  l (int): Number of outputs\n" \
    "  a (ndarray): State matrix (n x n, F-order), modified on exit\n" \
    "  b (ndarray): Input matrix (n x m, F-order), modified on exit\n" \
    "  c (ndarray): Output matrix (l x n, F-order), modified on exit\n" \
    "  d (ndarray): Feedthrough matrix (l x m, F-order), not modified\n" \
    "  x0 (ndarray): Initial state (n,), modified on exit\n" \
    "  apply (str, optional): 'A' = apply bijective mapping, 'N' = no mapping (default)\n" \
    "\n" \
    "Returns:\n" \
    "  (theta, a, b, c, scale, info): Parameter vector, transformed matrices, scale factor, exit code\n" \
    "  info=0: success, info=1: Lyapunov scale=0, info=2: unstable A, info=3: QR failed"

#define DOC_TB01VY "Convert output normal form to state-space representation.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): System order\n" \
    "  m (int): Number of inputs\n" \
    "  l (int): Number of outputs\n" \
    "  theta (ndarray): Parameter vector (N*(L+M+1)+L*M, F-order)\n" \
    "  apply (str, optional): 'A' = apply bijective mapping, 'N' = no mapping (default)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, c, d, x0, info): State-space matrices, initial state, exit code"

#define DOC_TB01WD "Reduce state matrix to real Schur form via orthogonal transformation.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Order of state matrix A\n" \
    "  m (int): Number of inputs (columns of B)\n" \
    "  p (int): Number of outputs (rows of C)\n" \
    "  a (ndarray): State dynamics matrix (n x n, F-order)\n" \
    "  b (ndarray): Input matrix (n x m, F-order)\n" \
    "  c (ndarray): Output matrix (p x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, c, u, wr, wi, info): Transformed system, Schur vectors, eigenvalues, exit code"

#define DOC_TB01WX "Reduce state matrix to upper Hessenberg form via orthogonal similarity.\n" \
    "\n" \
    "Reduces: A <- U'*A*U (Hessenberg), B <- U'*B, C <- C*U\n" \
    "\n" \
    "Parameters:\n" \
    "  compu (str): 'N' = no U, 'I' = initialize U to identity, 'U' = update given U1\n" \
    "  n (int): Order of state matrix A\n" \
    "  m (int): Number of inputs (columns of B)\n" \
    "  p (int): Number of outputs (rows of C)\n" \
    "  a (ndarray): State dynamics matrix (n x n, F-order)\n" \
    "  b (ndarray): Input matrix (n x m, F-order)\n" \
    "  c (ndarray): Output matrix (p x n, F-order)\n" \
    "  u (ndarray, optional): For COMPU='U', orthogonal matrix to update (n x n, F-order)\n" \
    "  ldwork (int, optional): Workspace size (-1 for query, default: automatic)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, c, u, dwork, info): Hessenberg A, transformed B and C, transformation U, workspace, exit code"

#define DOC_TB01XD "Special similarity transformation of dual state-space system.\n" \
    "\n" \
    "Applies: A <-- P*A'*P, B <-- P*C', C <-- B'*P, D <-- D' (optional)\n" \
    "where P has 1s on secondary diagonal (anti-identity matrix).\n" \
    "\n" \
    "Parameters:\n" \
    "  jobd (str): 'D' if D is present, 'Z' if D is zero matrix\n" \
    "  n (int): Order of state matrix A (n >= 0)\n" \
    "  m (int): Number of columns of B (m >= 0)\n" \
    "  p (int): Number of rows of C (p >= 0)\n" \
    "  kl (int): Number of subdiagonals of A (0 <= kl <= max(0,n-1))\n" \
    "  ku (int): Number of superdiagonals of A (0 <= ku <= max(0,n-1))\n" \
    "  a (ndarray): State matrix (n x n, F-order)\n" \
    "  b (ndarray): Input matrix (n x max(m,p), F-order)\n" \
    "  c (ndarray): Output matrix (max(m,p) x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix (max(m,p) x max(m,p), F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, c, d, info): Transformed matrices and exit code"

#define DOC_TB01XZ "Special similarity transformation of dual state-space system (complex).\n" \
    "\n" \
    "Applies: A <-- P*A'*P, B <-- P*C', C <-- B'*P, D <-- D' (optional)\n" \
    "where P has 1s on secondary diagonal (anti-identity matrix).\n" \
    "Complex version of tb01xd.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobd (str): 'D' if D is present, 'Z' if D is zero matrix\n" \
    "  n (int): Order of state matrix A (n >= 0)\n" \
    "  m (int): Number of columns of B (m >= 0)\n" \
    "  p (int): Number of rows of C (p >= 0)\n" \
    "  kl (int): Number of subdiagonals of A (0 <= kl <= max(0,n-1))\n" \
    "  ku (int): Number of superdiagonals of A (0 <= ku <= max(0,n-1))\n" \
    "  a (ndarray): Complex state matrix (n x n, F-order, dtype=complex)\n" \
    "  b (ndarray): Complex input matrix (n x max(m,p), F-order, dtype=complex)\n" \
    "  c (ndarray): Complex output matrix (max(m,p) x n, F-order, dtype=complex)\n" \
    "  d (ndarray): Complex feedthrough matrix (max(m,p) x max(m,p), F-order, dtype=complex)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, c, d, info): Transformed complex matrices and exit code"

#define DOC_TB01ZD "Controllable realization for single-input system.\n" \
    "\n" \
    "Finds controllable realization for the single-input system\n" \
    "dX/dt = A*X + B*U, Y = C*X by reducing to orthogonal canonical form.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobz (str): 'N'=no Z, 'F'=factored form, 'I'=identity initialized\n" \
    "  n (int): Order of state matrix A\n" \
    "  p (int): Number of outputs (rows of C)\n" \
    "  a (ndarray): State dynamics matrix (n x n, F-order)\n" \
    "  b (ndarray): Input vector (n, F-order)\n" \
    "  c (ndarray): Output matrix (p x n, F-order)\n" \
    "  tol (float, optional): Tolerance for controllability (default: 0.0)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, c, ncont, z, tau, info): Transformed system and metadata"

#define DOC_TB03AD "Convert state-space to polynomial matrix fraction.\n" \
    "\n" \
    "Computes left inv(P(s))*Q(s) or right Q(s)*inv(P(s)) polynomial matrix\n" \
    "representation with same transfer function as state-space (A,B,C,D).\n" \
    "\n" \
    "Parameters:\n" \
    "  leri (str): 'L' for left, 'R' for right fraction\n" \
    "  equil (str): 'S' to balance (A,B,C), 'N' for no balancing\n" \
    "  n (int): Order of state-space representation\n" \
    "  m (int): Number of inputs\n" \
    "  p (int): Number of outputs\n" \
    "  a (ndarray): State matrix (n, n) F-order\n" \
    "  b (ndarray): Input matrix (n, max(m,p)) F-order\n" \
    "  c (ndarray): Output matrix (max(m,p), n) F-order\n" \
    "  d (ndarray): Feedthrough matrix (max(m,p), max(m,p)) F-order\n" \
    "  tol (float): Tolerance for rank decisions (default 0.0)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, c, nr, index, pcoeff, qcoeff, vcoeff, iwork, info)"

#define DOC_TB03AY "Compute polynomial matrix V(s) block by block.\n" \
    "\n" \
    "Internal helper for TB03AD. Calculates V(s) one block at a time.\n" \
    "\n" \
    "Parameters:\n" \
    "  nr (int): Total state dimension\n" \
    "  a (ndarray): State matrix in upper block Hessenberg form (F-order)\n" \
    "  nblk (ndarray): Block sizes array (int32)\n" \
    "  vcoeff (ndarray): V(s) coefficient array (ldvco1, ldvco2, indblk+1)\n" \
    "  pcoeff (ndarray): P(s) coefficient array (workspace)\n" \
    "\n" \
    "Returns:\n" \
    "  (vcoeff, pcoeff, info): Modified arrays and exit code"

#define DOC_TB04AD "Convert state-space to transfer function representation.\n" \
    "\n" \
    "Computes the transfer function matrix T(s) as rows over common\n" \
    "denominators or columns over common denominators.\n" \
    "\n" \
    "T(s) = C * (sI - A)^(-1) * B + D\n" \
    "\n" \
    "Parameters:\n" \
    "  rowcol (str): 'R' rows over denominators, 'C' columns over denominators\n" \
    "  a (ndarray): N-by-N state matrix A (F-order)\n" \
    "  b (ndarray): N-by-M input matrix B (F-order)\n" \
    "  c (ndarray): P-by-N output matrix C (F-order)\n" \
    "  d (ndarray): P-by-M feedthrough matrix D (F-order)\n" \
    "  tol1 (float): Tolerance for rank determination (default 0.0)\n" \
    "  tol2 (float): Tolerance for controllability (default 0.0)\n" \
    "\n" \
    "Returns:\n" \
    "  (nr, index, dcoeff, ucoeff, info):\n" \
    "    nr: Order of controllable part\n" \
    "    index: Degrees of denominator polynomials\n" \
    "    dcoeff: Denominator polynomial coefficients\n" \
    "    ucoeff: Numerator polynomial coefficients\n" \
    "    info: 0=success"

#define DOC_TB04BD "Transfer function matrix via pole-zero method.\n" \
    "\n" \
    "Computes the transfer function matrix G of a state-space representation\n" \
    "(A,B,C,D) using the pole-zeros method. Each element is returned in\n" \
    "minimal form with numerator and denominator polynomials.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobd (str): 'D' if D matrix present, 'Z' if D is zero\n" \
    "  order (str): 'I' increasing powers, 'D' decreasing powers\n" \
    "  equil (str): 'S' equilibrate (A,B,C), 'N' no equilibration\n" \
    "  n (int): System order (n >= 0)\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  p (int): Number of outputs (p >= 0)\n" \
    "  md (int): Max polynomial degree + 1 (md >= 1, upper bound n+1)\n" \
    "  a (ndarray): State matrix (n x n, F-order)\n" \
    "  b (ndarray): Input matrix (n x m, F-order), destroyed on exit\n" \
    "  c (ndarray): Output matrix (p x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix (p x m, F-order)\n" \
    "  tol (float): Controllability tolerance (default 0.0)\n" \
    "\n" \
    "Returns:\n" \
    "  (ign, igd, gn, gd, info):\n" \
    "    ign: Numerator degrees (p x m, int32)\n" \
    "    igd: Denominator degrees (p x m, int32)\n" \
    "    gn: Numerator coefficients (p*m*md,)\n" \
    "    gd: Denominator coefficients (p*m*md,)\n" \
    "    info: 0=success, 1=QR failed for zeros, 2=QR failed for poles"

#define DOC_TB04BV "Separate strictly proper part from constant part of transfer function matrix.\n" \
    "\n" \
    "Separates proper P-by-M transfer function G into G = G0 + D where\n" \
    "G0 is strictly proper (deg(num) < deg(den)) and D is constant.\n" \
    "\n" \
    "Parameters:\n" \
    "  order (str): 'I' increasing, 'D' decreasing polynomial order\n" \
    "  p (int): Number of outputs (p >= 0)\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  md (int): Max polynomial degree + 1\n" \
    "  ign (ndarray): Numerator degrees (p x m, int32, F-order), modified\n" \
    "  igd (ndarray): Denominator degrees (p x m, int32, F-order)\n" \
    "  gn (ndarray): Numerator coefficients (p*m*md,), modified\n" \
    "  gd (ndarray): Denominator coefficients (p*m*md,)\n" \
    "  tol (float): Tolerance for negligible coefficients (default 0.0)\n" \
    "\n" \
    "Returns:\n" \
    "  (ign, gn, d, info):\n" \
    "    ign: Updated numerator degrees of G0\n" \
    "    gn: Updated numerator coefficients of G0\n" \
    "    d: Feedthrough matrix D (p x m)\n" \
    "    info: 0=success, 1=not proper, 2=null denominator"

#define DOC_TB04BW "Add real matrix D to rational matrix G (G + D).\n" \
    "\n" \
    "Computes the sum G + D where G is a P-by-M rational matrix (polynomial\n" \
    "ratios) and D is a P-by-M real matrix. The result replaces G in-place.\n" \
    "\n" \
    "If g(i,j) = 0 (both degrees zero and numerator zero), its denominator is\n" \
    "assumed to be 1.\n" \
    "\n" \
    "Parameters:\n" \
    "  order (str): 'I' increasing, 'D' decreasing polynomial order\n" \
    "  p (int): Number of outputs (p >= 0)\n" \
    "  m (int): Number of inputs (m >= 0)\n" \
    "  md (int): Max polynomial degree + 1\n" \
    "  ign (ndarray): Numerator degrees (p x m, int32, F-order), modified\n" \
    "  igd (ndarray): Denominator degrees (p x m, int32, F-order)\n" \
    "  gn (ndarray): Numerator coefficients (p*m*md,), modified\n" \
    "  gd (ndarray): Denominator coefficients (p*m*md,)\n" \
    "  d (ndarray): Real matrix D (p x m, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (ign, gn, info):\n" \
    "    ign: Updated numerator degrees of G + D\n" \
    "    gn: Updated numerator coefficients of G + D\n" \
    "    info: 0=success, -i=parameter i invalid"

#define DOC_TB04BX "Compute gain of SISO system from state-space, poles, and zeros.\n" \
    "\n" \
    "Computes the gain of a single-input single-output linear system,\n" \
    "given its state-space representation (A,b,c,d), and its poles and\n" \
    "zeros. The matrix A must be in upper Hessenberg form.\n" \
    "\n" \
    "The gain formula is:\n" \
    "  g = (c*(S0*I - A)^(-1)*b + d) * prod(S0 - Pi) / prod(S0 - Zi)\n" \
    "\n" \
    "where Pi are poles, Zi are zeros, and S0 is chosen away from all\n" \
    "poles and zeros.\n" \
    "\n" \
    "Parameters:\n" \
    "  ip (int): Number of system poles (ip >= 0)\n" \
    "  iz (int): Number of system zeros (iz >= 0)\n" \
    "  a (ndarray): IP-by-IP state matrix in upper Hessenberg form (F-order)\n" \
    "  b (ndarray): System input vector (ip,)\n" \
    "  c (ndarray): System output vector (ip,)\n" \
    "  d (float): System feedthrough scalar\n" \
    "  pr (ndarray): Real parts of poles (ip,)\n" \
    "  pi (ndarray): Imaginary parts of poles (ip,)\n" \
    "  zr (ndarray): Real parts of zeros (iz,)\n" \
    "  zi (ndarray): Imaginary parts of zeros (iz,)\n" \
    "\n" \
    "Returns:\n" \
    "  gain (float): Computed gain of the linear system"

#define DOC_TB04CD "State-space to minimal pole-zero-gain form.\n" \
    "\n" \
    "Computes the transfer function matrix G of a state-space representation\n" \
    "(A,B,C,D) using the pole-zeros method. Each SISO channel is returned in\n" \
    "minimal pole-zero-gain form.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobd (str): 'D' if D matrix present, 'Z' if D is zero\n" \
    "  equil (str): 'S' equilibrate (A,B,C), 'N' no equilibration\n" \
    "  a (ndarray): State matrix (n x n, F-order), modified if equil='S'\n" \
    "  b (ndarray): Input matrix (n x m, F-order), destroyed on exit\n" \
    "  c (ndarray): Output matrix (p x n, F-order), modified if equil='S'\n" \
    "  d (ndarray): Feedthrough matrix (p x m, F-order), not used if jobd='Z'\n" \
    "  npz (int): Max poles/zeros per SISO channel (upper bound n)\n" \
    "  tol (float): Controllability tolerance (default 0.0)\n" \
    "\n" \
    "Returns:\n" \
    "  (nz, np, zerosr, zerosi, polesr, polesi, gains, info):\n" \
    "    nz: Number of zeros per element (p x m, int32)\n" \
    "    np: Number of poles per element (p x m, int32)\n" \
    "    zerosr: Real parts of zeros (p*m*npz,)\n" \
    "    zerosi: Imaginary parts of zeros (p*m*npz,)\n" \
    "    polesr: Real parts of poles (p*m*npz,)\n" \
    "    polesi: Imaginary parts of poles (p*m*npz,)\n" \
    "    gains: Gain of each element (p x m)\n" \
    "    info: 0=success, 1=QR failed for zeros, 2=QR failed for poles"

#define DOC_TB05AD "Frequency response matrix of state-space system.\n" \
    "\n" \
    "Computes G(freq) = C * (freq*I - A)^(-1) * B.\n" \
    "\n" \
    "Parameters:\n" \
    "  baleig (str): 'N' no balance, 'C' condition est, 'B'/'E' balance+eig, 'A' all\n" \
    "  inita (str): 'G' general matrix, 'H' Hessenberg form\n" \
    "  a (ndarray): N-by-N state matrix A (F-order)\n" \
    "  b (ndarray): N-by-M input matrix B (F-order)\n" \
    "  c (ndarray): P-by-N output matrix C (F-order)\n" \
    "  freq (complex): Evaluation frequency\n" \
    "\n" \
    "Returns:\n" \
    "  (g, rcond, evre, evim, hinvb, info): Frequency response G(freq),\n" \
    "    condition estimate, eigenvalues (real/imag), H^(-1)*B, exit code"

#define DOC_TC01OD "Find dual polynomial matrix representation.\n" \
    "\n" \
    "Finds the dual right (left) polynomial matrix representation of a given\n" \
    "left (right) polynomial matrix representation by transposing both the\n" \
    "numerator Q(s) and denominator P(s) polynomial matrices.\n" \
    "\n" \
    "Parameters:\n" \
    "  leri (str): 'L' for left matrix fraction, 'R' for right matrix fraction\n" \
    "  m (int): Number of system inputs\n" \
    "  p (int): Number of system outputs\n" \
    "  pcoeff (ndarray): Denominator polynomial coefficients (porm x porm x indlim, F-order)\n" \
    "  qcoeff (ndarray): Numerator polynomial coefficients (max(m,p) x max(m,p) x indlim, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (pcoeff, qcoeff, info): Transposed coefficient matrices, exit code"

#define DOC_TC04AD "Convert polynomial matrix representation to state-space.\n" \
    "\n" \
    "Finds a state-space representation (A,B,C,D) with the same transfer\n" \
    "matrix T(s) as a given left or right polynomial matrix representation:\n" \
    "  T(s) = inv(P(s))*Q(s) for left,  T(s) = Q(s)*inv(P(s)) for right\n" \
    "\n" \
    "Uses Wolovich's Observable Structure Theorem to construct observable\n" \
    "companion form. For right matrix fractions, converts via duality.\n" \
    "\n" \
    "Parameters:\n" \
    "  leri (str): 'L' for left PMR inv(P(s))*Q(s), 'R' for right PMR Q(s)*inv(P(s))\n" \
    "  m (int): Number of system inputs (m >= 0)\n" \
    "  p (int): Number of system outputs (p >= 0)\n" \
    "  index (ndarray): Row/column degrees of P(s), int32 array of length max(m,p)\n" \
    "  pcoeff (ndarray): Denominator coefficients (porm x porm x kpcoef, F-order)\n" \
    "  qcoeff (ndarray): Numerator coefficients (porm x porp x kpcoef, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (n, rcond, a, b, c, d, info): State order, condition number,\n" \
    "    state-space matrices (A n-by-n, B n-by-m, C p-by-n, D p-by-m), exit code\n" \
    "    info=1 means P(s) is not row/column proper"

#define DOC_TC05AD "Evaluate transfer matrix at complex frequency.\n" \
    "\n" \
    "Computes the frequency response matrix T(s) of a left or right polynomial\n" \
    "matrix representation at a specified complex frequency s = SVAL:\n" \
    "  T(s) = inv(P(s))*Q(s) for left PMR\n" \
    "  T(s) = Q(s)*inv(P(s)) for right PMR\n" \
    "\n" \
    "For standard frequency response, use SVAL = (0, omega) for frequency omega.\n" \
    "\n" \
    "Parameters:\n" \
    "  leri (str): 'L' for left PMR, 'R' for right PMR\n" \
    "  m (int): Number of system inputs (m >= 0)\n" \
    "  p (int): Number of system outputs (p >= 0)\n" \
    "  sval (complex): Complex frequency at which to evaluate\n" \
    "  index (ndarray): Polynomial degrees, int32 array of length max(m,p)\n" \
    "  pcoeff (ndarray): Denominator coefficients (porm x porm x kpcoef, F-order)\n" \
    "  qcoeff (ndarray): Numerator coefficients (porm x porp x kpcoef, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (rcond, cfreqr, info): Reciprocal condition of P(SVAL),\n" \
    "    complex frequency response matrix (p x m), exit code.\n" \
    "    info=1 means P(SVAL) is exactly or nearly singular."

#define DOC_TD03AD "Polynomial matrix representation for proper transfer matrix.\n" \
    "\n" \
    "Finds a relatively prime left or right polynomial matrix representation\n" \
    "for a proper transfer matrix T(s) given as row or column polynomial\n" \
    "vectors over common denominator polynomials.\n" \
    "\n" \
    "For LERI='L': T(s) = inv(P(s)) * Q(s)\n" \
    "For LERI='R': T(s) = Q(s) * inv(P(s))\n" \
    "\n" \
    "Also computes minimal state-space representation (A,B,C,D) en route.\n" \
    "\n" \
    "Parameters:\n" \
    "  rowcol (str): 'R' = T(s) factorized by rows, 'C' = by columns\n" \
    "  leri (str): 'L' = left PMR inv(P)*Q, 'R' = right PMR Q*inv(P)\n" \
    "  equil (str): 'S' = perform balancing, 'N' = no balancing\n" \
    "  m (int): Number of system inputs\n" \
    "  p (int): Number of system outputs\n" \
    "  indexd (ndarray): Degrees of denominator polynomials, int32 array\n" \
    "  dcoeff (ndarray): Denominator coefficients (porm x kdcoef, F-order)\n" \
    "  ucoeff (ndarray): Numerator coefficients (p x m x kdcoef, F-order)\n" \
    "  tol (float): Tolerance for rank determination (0 = default)\n" \
    "\n" \
    "Returns:\n" \
    "  (nr, a, b, c, d, indexp, pcoeff, qcoeff, vcoeff, iwork, info):\n" \
    "    nr (int): Minimal order\n" \
    "    a (ndarray): State matrix (lda x n)\n" \
    "    b (ndarray): Input matrix (ldb x max(m,p))\n" \
    "    c (ndarray): Output matrix (ldc x n)\n" \
    "    d (ndarray): Feedthrough matrix (ldd x max(m,p))\n" \
    "    indexp (ndarray): Degrees of P(s) polynomials\n" \
    "    pcoeff (ndarray): Denominator matrix P(s) coefficients\n" \
    "    qcoeff (ndarray): Numerator matrix Q(s) coefficients\n" \
    "    vcoeff (ndarray): Intermediate matrix V(s) coefficients\n" \
    "    iwork (ndarray): Diagonal block orders of A\n" \
    "    info (int): Exit code, 0 = success"

#define DOC_TD03AY "Calculate state-space from polynomial row vectors over diagonal denominators.\n" \
    "\n" \
    "Helper for TD04AD. Calculates state-space (A,B,C,D) for transfer matrix\n" \
    "T(s) = inv(D(s)) * U(s) where D(s) is diagonal.\n" \
    "\n" \
    "Parameters:\n" \
    "  mwork (int): Number of inputs\n" \
    "  pwork (int): Number of outputs\n" \
    "  index (ndarray): Degrees of diagonal D(s) elements, int32 array\n" \
    "  dcoeff (ndarray): Denominator coefficients (pwork x kdcoef, F-order)\n" \
    "  ucoeff (ndarray): Numerator coefficients (pwork x mwork x kdcoef, F-order)\n" \
    "  n (int): State-space order = sum(index)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, b, c, d, info): State-space matrices, exit code.\n" \
    "    info > 0 means row info has near-zero leading coefficient."

#define DOC_TD04AD "Minimal state-space representation from transfer function.\n" \
    "\n" \
    "Finds a minimal state-space representation (A,B,C,D) for a proper\n" \
    "transfer matrix T(s) given as polynomial vectors over denominators.\n" \
    "\n" \
    "For ROWCOL='R': T(s) = inv(D(s)) * U(s)\n" \
    "For ROWCOL='C': T(s) = U(s) * inv(D(s))\n" \
    "\n" \
    "Parameters:\n" \
    "  rowcol (str): 'R' = rows over common denominators,\n" \
    "                'C' = columns over common denominators\n" \
    "  m (int): Number of system inputs\n" \
    "  p (int): Number of system outputs\n" \
    "  index (ndarray): Degrees of denominator polynomials, int32 array\n" \
    "  dcoeff (ndarray): Denominator coefficients (porm x kdcoef, F-order)\n" \
    "  ucoeff (ndarray): Numerator coefficients (p x m x kdcoef, F-order)\n" \
    "  tol (float): Tolerance for rank determination (0 = default)\n" \
    "\n" \
    "Returns:\n" \
    "  (nr, a, b, c, d, info): Minimal order, state-space matrices, exit code.\n" \
    "    info > 0 means row info has near-zero leading coefficient."

#define DOC_TD05AD "Evaluate transfer function G(jW) at specified frequency.\n" \
    "\n" \
    "Given a complex valued rational transfer function G(jW), calculates its\n" \
    "complex value or magnitude/phase for a specified frequency W.\n" \
    "\n" \
    "        B(1)+B(2)*(jW)+B(3)*(jW)^2+...+B(MP1)*(jW)^(MP1-1)\n" \
    "G(jW) = --------------------------------------------------\n" \
    "        A(1)+A(2)*(jW)+A(3)*(jW)^2+...+A(NP1)*(jW)^(NP1-1)\n" \
    "\n" \
    "Parameters:\n" \
    "  unitf (str): 'R' = frequency in radians/second, 'H' = frequency in hertz\n" \
    "  output (str): 'C' = Cartesian (real/imag), 'P' = Polar (dB/degrees)\n" \
    "  np1 (int): Denominator order + 1 (>= 1)\n" \
    "  mp1 (int): Numerator order + 1 (>= 1)\n" \
    "  w (float): Frequency value\n" \
    "  a (ndarray): Denominator coefficients, ascending powers of jW\n" \
    "  b (ndarray): Numerator coefficients, ascending powers of jW\n" \
    "\n" \
    "Returns:\n" \
    "  (valr, vali, info): Output values and exit code.\n" \
    "    For OUTPUT='C': valr = Re(G), vali = Im(G)\n" \
    "    For OUTPUT='P': valr = magnitude in dB, vali = phase in degrees\n" \
    "    info=1 means W is a pole or all A coefficients are zero."

#define DOC_TF01MD "Output response sequence of discrete-time state-space system.\n" \
    "\n" \
    "Simulates: x(k+1) = A*x(k) + B*u(k), y(k) = C*x(k) + D*u(k)\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): State matrix (n x n, F-order)\n" \
    "  b (ndarray): Input matrix (n x m, F-order)\n" \
    "  c (ndarray): Output matrix (p x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix (p x m, F-order)\n" \
    "  u (ndarray): Input sequence (m x ny, F-order)\n" \
    "  x (ndarray): Initial state vector (n,), modified in-place\n" \
    "\n" \
    "Returns:\n" \
    "  (y, x_final, info): Output sequence (p x ny), final state, exit code"

#define DOC_TF01MX "Output sequence of linear time-invariant open-loop system.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Order of matrix A\n" \
    "  m (int): Number of system inputs\n" \
    "  p (int): Number of system outputs\n" \
    "  ny (int): Number of output vectors to compute\n" \
    "  s (ndarray): System matrix [A B; C D] (n+p x n+m, F-order)\n" \
    "  u (ndarray): Input sequence (ny x m, F-order)\n" \
    "  x (ndarray): Initial state vector (n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (y, x, info): Output sequence, final state, exit code"

#define DOC_TF01MY "Output sequence of discrete-time state-space system (variant).\n" \
    "\n" \
    "Simulates: x(k+1) = A*x(k) + B*u(k), y(k) = C*x(k) + D*u(k)\n" \
    "\n" \
    "This variant differs from TF01MD in that input/output trajectories\n" \
    "are stored row-wise: U is NY-by-M (rows are u(k)'), Y is NY-by-P\n" \
    "(rows are y(k)').\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): State matrix (n x n, F-order)\n" \
    "  b (ndarray): Input matrix (n x m, F-order)\n" \
    "  c (ndarray): Output matrix (p x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix (p x m, F-order)\n" \
    "  u (ndarray): Input sequence (ny x m, F-order), rows are u(k)'\n" \
    "  x (ndarray): Initial state vector (n,), modified in-place\n" \
    "\n" \
    "Returns:\n" \
    "  (y, x_final, info): Output sequence (ny x p), final state, exit code"

#define DOC_TF01ND "Output response sequence with Hessenberg state matrix.\n" \
    "\n" \
    "Simulates: x(k+1) = A*x(k) + B*u(k), y(k) = C*x(k) + D*u(k)\n" \
    "where A is an upper or lower Hessenberg matrix.\n" \
    "\n" \
    "Processing time is approximately half that of TF01MD due to\n" \
    "exploiting the Hessenberg structure of A.\n" \
    "\n" \
    "Parameters:\n" \
    "  uplo (str): 'U' for upper Hessenberg A, 'L' for lower Hessenberg A\n" \
    "  a (ndarray): Hessenberg state matrix (n x n, F-order)\n" \
    "  b (ndarray): Input matrix (n x m, F-order)\n" \
    "  c (ndarray): Output matrix (p x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix (p x m, F-order)\n" \
    "  u (ndarray): Input sequence (m x ny, F-order)\n" \
    "  x (ndarray): Initial state vector (n,), modified in-place\n" \
    "\n" \
    "Returns:\n" \
    "  (y, x_final, info): Output sequence (p x ny), final state, exit code"

#define DOC_TF01OD "Block Hankel expansion of multivariable parameter sequence.\n" \
    "\n" \
    "Constructs block Hankel matrix T from sequence M(1),...,M(NR+NC-1):\n" \
    "\n" \
    "     | M(1)   M(2)   ...  M(NC)     |\n" \
    " T = | M(2)   M(3)   ...  M(NC+1)   |\n" \
    "     | ...                 ...      |\n" \
    "     | M(NR)  M(NR+1)...  M(NR+NC-1)|\n" \
    "\n" \
    "Parameters:\n" \
    "  h (ndarray): Parameter sequence (NH1 x (NR+NC-1)*NH2, F-order).\n" \
    "               M(k) stored in columns (k-1)*NH2+1 to k*NH2.\n" \
    "  nr (int): Number of block rows in T (nr >= 0)\n" \
    "  nc (int): Number of block columns in T (nc >= 0)\n" \
    "\n" \
    "Returns:\n" \
    "  (t, info): Block Hankel matrix (NH1*NR x NH2*NC), exit code"

#define DOC_TF01PD "Block Toeplitz expansion of multivariable parameter sequence.\n" \
    "\n" \
    "Constructs block Toeplitz matrix T from sequence M(1),...,M(NR+NC-1):\n" \
    "\n" \
    "     | M(NC)     M(NC-1)   ...  M(1)      |\n" \
    " T = | M(NC+1)   M(NC)     ...  M(2)      |\n" \
    "     | ...                      ...       |\n" \
    "     | M(NR+NC-1) M(NR+NC-2) ... M(NR)    |\n" \
    "\n" \
    "Parameters:\n" \
    "  h (ndarray): Parameter sequence (NH1 x (NR+NC-1)*NH2, F-order).\n" \
    "               M(k) stored in columns (k-1)*NH2+1 to k*NH2.\n" \
    "  nr (int): Number of block rows in T (nr >= 0)\n" \
    "  nc (int): Number of block columns in T (nc >= 0)\n" \
    "\n" \
    "Returns:\n" \
    "  (t, info): Block Toeplitz matrix (NH1*NR x NH2*NC), exit code"

#define DOC_TF01QD "Markov parameters from transfer function matrix.\n" \
    "\n" \
    "Computes N Markov parameters M(1), M(2), ..., M(N) from a multivariable\n" \
    "system whose transfer function matrix G(z) is given in ARMA form.\n" \
    "\n" \
    "The (i,j)-th element of G(z) has the form:\n" \
    "         MA(1)z^{-1} + MA(2)z^{-2} + ... + MA(r)z^{-r}\n" \
    " G_ij = ----------------------------------------------\n" \
    "        1 + AR(1)z^{-1} + AR(2)z^{-2} + ... + AR(r)z^{-r}\n" \
    "\n" \
    "where r is the order of the element.\n" \
    "\n" \
    "Parameters:\n" \
    "  nc (int): Number of outputs (rows in G(z)), nc >= 0\n" \
    "  nb (int): Number of inputs (columns in G(z)), nb >= 0\n" \
    "  n (int): Number of Markov parameters to compute, n >= 0\n" \
    "  iord (ndarray): Orders of transfer function elements (nc*nb,), int32\n" \
    "  ar (ndarray): Denominator coefficients, float64\n" \
    "  ma (ndarray): Numerator coefficients, float64\n" \
    "\n" \
    "Returns:\n" \
    "  (h, info): Markov parameters (nc x n*nb), exit code"

#define DOC_TF01RD "Compute Markov parameters from state-space representation.\n" \
    "\n" \
    "Computes N Markov parameters M(1), M(2), ..., M(N) from system\n" \
    "matrices (A, B, C), where M(k) = C * A^(k-1) * B.\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): State matrix (na x na, F-order)\n" \
    "  b (ndarray): Input matrix (na x nb, F-order)\n" \
    "  c (ndarray): Output matrix (nc x na, F-order)\n" \
    "  n (int): Number of Markov parameters to compute\n" \
    "\n" \
    "Returns:\n" \
    "  (h, info): Markov parameters (nc x n*nb), exit code"

#define DOC_TG01AD "Balance the matrices of a descriptor system pencil.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'A' all, 'B' B+A+E, 'C' C+A+E, 'N' A+E only\n" \
    "  l (int): Number of rows of A, B, E\n" \
    "  n (int): Number of columns of A, E, C\n" \
    "  m (int): Number of columns of B\n" \
    "  p (int): Number of rows of C\n" \
    "  thresh (float): Threshold for small elements\n" \
    "  a (ndarray): State dynamics matrix (l x n, F-order)\n" \
    "  e (ndarray): Descriptor matrix (l x n, F-order)\n" \
    "  b (ndarray): Input/state matrix (l x m, F-order)\n" \
    "  c (ndarray): State/output matrix (p x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, b, c, lscale, rscale, info): Balanced system and scaling factors"

#define DOC_TG01AZ "Balance the matrices of a complex descriptor system pencil.\n" \
    "\n" \
    "Complex version of tg01ad. Balances system pencil via diagonal\n" \
    "similarity transformations (Dl*A*Dr - lambda Dl*E*Dr, Dl*B, C*Dr).\n" \
    "Magnitude is computed as |real| + |imag| for thresholding.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'A' all, 'B' B+A+E, 'C' C+A+E, 'N' A+E only\n" \
    "  l (int): Number of rows of A, B, E\n" \
    "  n (int): Number of columns of A, E, C\n" \
    "  m (int): Number of columns of B\n" \
    "  p (int): Number of rows of C\n" \
    "  thresh (float): Threshold for small elements\n" \
    "  a (ndarray): Complex state dynamics matrix (l x n, F-order)\n" \
    "  e (ndarray): Complex descriptor matrix (l x n, F-order)\n" \
    "  b (ndarray): Complex input/state matrix (l x m, F-order)\n" \
    "  c (ndarray): Complex state/output matrix (p x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, b, c, lscale, rscale, info): Balanced system and scaling factors"

#define DOC_TG01BD "Reduce descriptor system to generalized Hessenberg form.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobe (str): 'G' = E general, 'U' = E upper triangular\n" \
    "  compq (str): 'N', 'I', or 'V' - Q computation mode\n" \
    "  compz (str): 'N', 'I', or 'V' - Z computation mode\n" \
    "  ilo (int): Lower active row/column index (1-based)\n" \
    "  ihi (int): Upper active row/column index (1-based)\n" \
    "  a (ndarray): State dynamics matrix (n x n, F-order)\n" \
    "  e (ndarray): Descriptor matrix (n x n, F-order)\n" \
    "  b (ndarray): Input/state matrix (n x m, F-order)\n" \
    "  c (ndarray): State/output matrix (p x n, F-order)\n" \
    "  q (ndarray): Orthogonal matrix Q (n x n, F-order)\n" \
    "  z (ndarray): Orthogonal matrix Z (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, b, c, q, z, info): Transformed system in Hessenberg form"

#define DOC_TG01CD "Reduce descriptor system pair (A-lambda E, B) to QR-coordinate form.\n" \
    "\n" \
    "Computes orthogonal matrix Q such that Q'*E is upper trapezoidal.\n" \
    "\n" \
    "Parameters:\n" \
    "  compq (str): 'N' no Q, 'I' init Q to I, 'U' update Q1*Q\n" \
    "  a (ndarray): State dynamics matrix (l x n, F-order)\n" \
    "  e (ndarray): Descriptor matrix (l x n, F-order)\n" \
    "  b (ndarray): Input/state matrix (l x m, F-order)\n" \
    "  q (ndarray, optional): Input Q for compq='U' (l x l, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, b, q, info): Transformed system with Q'*E upper trapezoidal"

#define DOC_TG01DD "Orthogonal reduction of descriptor system (C,A-lambda E) to RQ-coordinate form.\n" \
    "\n" \
    "Computes orthogonal matrix Z such that E*Z is upper trapezoidal.\n" \
    "\n" \
    "Parameters:\n" \
    "  compz (str): 'N' no Z, 'I' init Z to I, 'U' update Z1*Z\n" \
    "  a (ndarray): State dynamics matrix (l x n, F-order)\n" \
    "  e (ndarray): Descriptor matrix (l x n, F-order)\n" \
    "  c (ndarray): State/output matrix (p x n, F-order)\n" \
    "  z (ndarray, optional): Input Z for compz='U' (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, c, z, info): Transformed system with E*Z upper trapezoidal"

#define DOC_TG01ED "Orthogonal reduction of descriptor system to SVD coordinate form.\n" \
    "\n" \
    "Computes orthogonal matrices Q and Z such that Q'*A*Z and Q'*E*Z are\n" \
    "in SVD coordinate form, with E transformed to diagonal form containing\n" \
    "its singular values.\n" \
    "\n" \
    "Parameters:\n" \
    "  joba (str): 'N' = no A22 reduction, 'R' = reduce A22 to SVD form\n" \
    "  a (ndarray): State dynamics matrix (l x n, F-order)\n" \
    "  e (ndarray): Descriptor matrix (l x n, F-order)\n" \
    "  b (ndarray): Input/state matrix (l x m, F-order)\n" \
    "  c (ndarray): State/output matrix (p x n, F-order)\n" \
    "  tol (float): Tolerance for rank determination (< 1.0; 0 for default)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, b, c, q, z, ranke, rnka22, info):\n" \
    "    Transformed matrices, orthogonal Q and Z, ranks of E and A22, exit code"

#define DOC_TG01FD "Orthogonal reduction of descriptor system to SVD-like form.\n" \
    "\n" \
    "Parameters:\n" \
    "  compq (str): 'N', 'I', or 'U' - Q computation mode\n" \
    "  compz (str): 'N', 'I', or 'U' - Z computation mode\n" \
    "  joba (str): 'N', 'R', or 'T' - A22 reduction mode\n" \
    "  l (int): Number of rows of A, B, E\n" \
    "  n (int): Number of columns of A, E, C\n" \
    "  m (int): Number of columns of B\n" \
    "  p (int): Number of rows of C\n" \
    "  a (ndarray): State dynamics matrix (l x n, F-order)\n" \
    "  e (ndarray): Descriptor matrix (l x n, F-order)\n" \
    "  b (ndarray): Input/state matrix (l x m, F-order)\n" \
    "  c (ndarray): State/output matrix (p x n, F-order)\n" \
    "  tol (float): Tolerance for rank determination\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, b, c, q, z, ranke, rnka22, info): Transformed system and ranks"

#define DOC_TG01FZ "Unitary reduction of complex descriptor system to SVD-like form.\n" \
    "\n" \
    "Computes unitary matrices Q and Z for the complex descriptor system\n" \
    "(A-lambda*E,B,C) such that Q'*A*Z and Q'*E*Z are in SVD-like form.\n" \
    "This is the complex version of tg01fd.\n" \
    "\n" \
    "Parameters:\n" \
    "  compq (str): 'N', 'I', or 'U' - Q computation mode\n" \
    "  compz (str): 'N', 'I', or 'U' - Z computation mode\n" \
    "  joba (str): 'N', 'R', or 'T' - A22 reduction mode\n" \
    "  l (int): Number of rows of A, B, E\n" \
    "  n (int): Number of columns of A, E, C\n" \
    "  m (int): Number of columns of B\n" \
    "  p (int): Number of rows of C\n" \
    "  a (ndarray): Complex state dynamics matrix (l x n, F-order)\n" \
    "  e (ndarray): Complex descriptor matrix (l x n, F-order)\n" \
    "  b (ndarray): Complex input/state matrix (l x m, F-order)\n" \
    "  c (ndarray): Complex state/output matrix (p x n, F-order)\n" \
    "  tol (float): Tolerance for rank determination\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, b, c, q, z, ranke, rnka22, info): Transformed system and ranks"

#define DOC_TG01GD "Reduced descriptor representation without non-dynamic modes.\n" \
    "\n" \
    "Finds a reduced descriptor representation (Ar-lambda*Er,Br,Cr,Dr)\n" \
    "without non-dynamic modes. Optionally transforms Er's leading\n" \
    "diagonal block to identity (standard form).\n" \
    "\n" \
    "Parameters:\n" \
    "  jobs (str): 'S' = make E11 identity, 'D' = keep E upper triangular\n" \
    "  l (int): Number of rows of A, B, E\n" \
    "  n (int): Number of columns of A, E, C\n" \
    "  m (int): Number of columns of B\n" \
    "  p (int): Number of rows of C\n" \
    "  a (ndarray): State dynamics matrix (l x n, F-order)\n" \
    "  e (ndarray): Descriptor matrix (l x n, F-order)\n" \
    "  b (ndarray): Input/state matrix (l x m, F-order)\n" \
    "  c (ndarray): State/output matrix (p x n, F-order)\n" \
    "  d (ndarray): Feedthrough matrix (p x m, F-order)\n" \
    "  tol (float): Tolerance for rank determination\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, b, c, d, lr, nr, ranke, infred, info):\n" \
    "    Reduced system (LR x NR), rank of E, reduction info, exit code"

#define DOC_TG01HD "Orthogonal reduction of descriptor system to controllability staircase form.\n" \
    "\n" \
    "Computes orthogonal transformations Q and Z to reduce descriptor system\n" \
    "(A-lambda*E,B,C) to the form where controllable and uncontrollable parts\n" \
    "are separated. JOBCON selects which uncontrollable eigenvalues to separate:\n" \
    "'C' = both finite and infinite, 'F' = finite only, 'I' = infinite only.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobcon (str): 'C', 'F', or 'I' - eigenvalue separation mode\n" \
    "  compq (str): 'N', 'I', or 'U' - Q computation mode\n" \
    "  compz (str): 'N', 'I', or 'U' - Z computation mode\n" \
    "  n (int): Order of state matrices A and E\n" \
    "  m (int): Number of inputs (columns of B)\n" \
    "  p (int): Number of outputs (rows of C)\n" \
    "  a (ndarray): State matrix (n x n, F-order)\n" \
    "  e (ndarray): Descriptor matrix (n x n, F-order)\n" \
    "  b (ndarray): Input matrix (n x m, F-order)\n" \
    "  c (ndarray): Output matrix (p x n, F-order)\n" \
    "  tol (float): Tolerance for rank determination (< 1.0; 0 for default)\n" \
    "  q (ndarray, optional): Input Q for COMPQ='U' (n x n, F-order)\n" \
    "  z (ndarray, optional): Input Z for COMPZ='U' (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, b, c, q, z, ncont, niucon, nrblck, rtau, info):\n" \
    "    Transformed Q'*A*Z, Q'*E*Z, Q'*B, C*Z, orthogonal matrices Q and Z,\n" \
    "    controllable order NCONT, uncontrollable infinite eigenvalue count NIUCON,\n" \
    "    number of staircase blocks NRBLCK, block sizes RTAU, and exit code"

#define DOC_TG01HU "Orthogonal reduction to multi-input controllability staircase form.\n" \
    "\n" \
    "Reduces descriptor system (A-lambda*E,[B1,B2],C) with separate input\n" \
    "groups B1 (M1 columns) and B2 (M2 columns) to controllability staircase\n" \
    "form, separating controllable and uncontrollable parts.\n" \
    "\n" \
    "Parameters:\n" \
    "  compq (str): 'N', 'I', or 'U' - Q computation mode\n" \
    "  compz (str): 'N', 'I', or 'U' - Z computation mode\n" \
    "  l (int): Number of rows of A, B, E\n" \
    "  n (int): Number of columns of A, E, C\n" \
    "  m1 (int): Number of columns of B1 (first input group)\n" \
    "  m2 (int): Number of columns of B2 (second input group)\n" \
    "  p (int): Number of rows of C\n" \
    "  n1 (int): Order of subsystem to reduce (0 <= n1 <= min(l,n))\n" \
    "  lbe (int): Number of sub-diagonals in E1 (0 <= lbe < n1)\n" \
    "  a (ndarray): State matrix (l x n, F-order)\n" \
    "  e (ndarray): Descriptor matrix (l x n, F-order)\n" \
    "  b (ndarray): Input matrix [B1,B2] (l x (m1+m2), F-order)\n" \
    "  c (ndarray): Output matrix (p x n, F-order)\n" \
    "  tol (float): Tolerance for rank determination\n" \
    "  q (ndarray, optional): Input Q for COMPQ='U' (l x l, F-order)\n" \
    "  z (ndarray, optional): Input Z for COMPZ='U' (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, b, c, q, z, nr, nrblck, rtau, info):\n" \
    "    Transformed system, orthogonal matrices Q and Z,\n" \
    "    controllable order NR, number of staircase blocks,\n" \
    "    block sizes RTAU, and exit code"

#define DOC_TG01HX "Orthogonal reduction to controllability staircase form.\n" \
    "\n" \
    "Reduces descriptor system (A-lambda*E,B,C) to controllability\n" \
    "staircase form, separating controllable and uncontrollable parts.\n" \
    "\n" \
    "Parameters:\n" \
    "  compq (str): 'N', 'I', or 'U' - Q computation mode\n" \
    "  compz (str): 'N', 'I', or 'U' - Z computation mode\n" \
    "  l (int): Number of rows of A, B, E\n" \
    "  n (int): Number of columns of A, E, C\n" \
    "  m (int): Number of columns of B\n" \
    "  p (int): Number of rows of C\n" \
    "  n1 (int): Order of subsystem to reduce (0 <= n1 <= min(l,n))\n" \
    "  lbe (int): Number of sub-diagonals in E1 (0 <= lbe < n1)\n" \
    "  a (ndarray): State matrix (l x n, F-order)\n" \
    "  e (ndarray): Descriptor matrix (l x n, F-order)\n" \
    "  b (ndarray): Input matrix (l x m, F-order)\n" \
    "  c (ndarray): Output matrix (p x n, F-order)\n" \
    "  tol (float): Tolerance for rank determination\n" \
    "  q (ndarray, optional): Input Q for COMPQ='U' (l x l, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, b, c, q, z, nr, nrblck, rtau, info):\n" \
    "    Transformed system, orthogonal matrices Q and Z,\n" \
    "    controllable order NR, number of staircase blocks,\n" \
    "    block sizes RTAU, and exit code"

#define DOC_TG01HY "Blocked version of orthogonal reduction to controllability staircase form.\n" \
    "\n" \
    "Reduces descriptor system (A-lambda*E,B,C) to controllability\n" \
    "staircase form using block algorithms for better performance on large systems.\n" \
    "Separates controllable and uncontrollable parts.\n" \
    "\n" \
    "Parameters:\n" \
    "  compq (str): 'N', 'I', or 'U' - Q computation mode\n" \
    "  compz (str): 'N', 'I', or 'U' - Z computation mode\n" \
    "  l (int): Number of rows of A, B, E\n" \
    "  n (int): Number of columns of A, E, C\n" \
    "  m (int): Number of columns of B\n" \
    "  p (int): Number of rows of C\n" \
    "  n1 (int): Order of subsystem to reduce (0 <= n1 <= min(l,n))\n" \
    "  lbe (int): Number of sub-diagonals in E1 (0 <= lbe < n1)\n" \
    "  a (ndarray): State matrix (l x n, F-order)\n" \
    "  e (ndarray): Descriptor matrix (l x n, F-order)\n" \
    "  b (ndarray): Input matrix (l x m, F-order)\n" \
    "  c (ndarray): Output matrix (p x n, F-order)\n" \
    "  tol (float): Tolerance for rank determination\n" \
    "  q (ndarray, optional): Input Q for COMPQ='U' (l x l, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, b, c, q, z, nr, nrblck, rtau, info):\n" \
    "    Transformed system, orthogonal matrices Q and Z,\n" \
    "    controllable order NR, number of staircase blocks,\n" \
    "    block sizes RTAU, and exit code"

#define DOC_TG01ID "Orthogonal reduction of descriptor system to observability staircase form.\n" \
    "\n" \
    "Computes orthogonal transformation matrices Q and Z which reduce the\n" \
    "N-th order descriptor system (A-lambda*E,B,C) to the form:\n" \
    "  Q'*A*Z = (Ano, *; 0, Ao), Q'*E*Z = (Eno, *; 0, Eo),\n" \
    "  Q'*B = (Bno; Bo), C*Z = (0, Co)\n" \
    "where (Ao-lambda*Eo,Bo,Co) is finite and/or infinite observable.\n" \
    "\n" \
    "JOBOBS selects which unobservable eigenvalues to separate:\n" \
    "  'O' = both finite and infinite\n" \
    "  'F' = finite only\n" \
    "  'I' = nonzero finite and infinite\n" \
    "\n" \
    "Parameters:\n" \
    "  jobobs (str): 'O', 'F', or 'I' - eigenvalue separation mode\n" \
    "  compq (str): 'N', 'I', or 'U' - Q computation mode\n" \
    "  compz (str): 'N', 'I', or 'U' - Z computation mode\n" \
    "  a (ndarray): State matrix (n x n, F-order)\n" \
    "  e (ndarray): Descriptor matrix (n x n, F-order)\n" \
    "  b (ndarray): Input matrix (n x m, F-order)\n" \
    "  c (ndarray): Output matrix (p x n, F-order)\n" \
    "  tol (float): Tolerance for rank determination (< 1.0; 0 for default)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, b, c, q, z, nobsv, niuobs, nlblck, ctau, info):\n" \
    "    Transformed Q'*A*Z, Q'*E*Z, Q'*B, C*Z, orthogonal matrices Q and Z,\n" \
    "    observable order NOBSV, unobservable infinite eigenvalue count NIUOBS,\n" \
    "    number of staircase blocks NLBLCK, block sizes CTAU, and exit code"

#define DOC_TG01JD "Find irreducible descriptor representation.\n" \
    "\n" \
    "Finds a reduced (controllable, observable, or irreducible) descriptor\n" \
    "representation (Ar-lambda*Er,Br,Cr) for an original descriptor\n" \
    "representation (A-lambda*E,B,C). The pencil Ar-lambda*Er is in upper\n" \
    "block Hessenberg form with either Ar or Er upper triangular.\n" \
    "\n" \
    "The reduction is performed in 4 phases:\n" \
    "  Phase 1: Eliminate finite uncontrollable eigenvalues\n" \
    "  Phase 2: Eliminate infinite and finite nonzero uncontrollable eigenvalues\n" \
    "  Phase 3: Eliminate finite unobservable eigenvalues\n" \
    "  Phase 4: Eliminate infinite and finite nonzero unobservable eigenvalues\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'I' = irreducible, 'C' = controllable, 'O' = observable\n" \
    "  systyp (str): 'R' = rational, 'S' = proper/standard, 'P' = polynomial\n" \
    "  equil (str): 'S' = scale system, 'N' = no scaling\n" \
    "  a (ndarray): State matrix (n x n, F-order)\n" \
    "  e (ndarray): Descriptor matrix (n x n, F-order)\n" \
    "  b (ndarray): Input matrix (n x m, F-order; n x max(m,p) if job != 'C')\n" \
    "  c (ndarray): Output matrix (p x n, F-order; max(m,p) x n if job != 'C')\n" \
    "  tol (float): Tolerance for rank determination (< 1.0; <= 0 for default)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, b, c, nr, infred, iwork, info):\n" \
    "    Reduced Ar, Er, Br, Cr matrices; reduced order NR;\n" \
    "    INFRED array (phase reductions in [0-3], A/E sub-diagonals in [4-5],\n" \
    "    block count in [6]); IWORK with block dimensions; exit code"

#define DOC_TG01JY "Find irreducible descriptor representation with singularity check.\n" \
    "\n" \
    "Finds a reduced (controllable, observable, or irreducible) descriptor\n" \
    "representation (Ar-lambda*Er,Br,Cr) for a regular descriptor system\n" \
    "(A-lambda*E,B,C). Extended version of TG01JD with options for checking\n" \
    "pencil singularity and restoring original matrices for maximum accuracy.\n" \
    "\n" \
    "The reduction is performed in 4 phases:\n" \
    "  Phase 1: Eliminate finite uncontrollable eigenvalues\n" \
    "  Phase 2: Eliminate infinite and finite nonzero uncontrollable eigenvalues\n" \
    "  Phase 3: Eliminate finite unobservable eigenvalues\n" \
    "  Phase 4: Eliminate infinite and finite nonzero unobservable eigenvalues\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'I' = irreducible, 'C' = controllable, 'O' = observable\n" \
    "  systyp (str): 'R' = rational, 'S' = proper/standard, 'P' = polynomial\n" \
    "  equil (str): 'S' = scale system, 'N' = no scaling\n" \
    "  cksing (str): 'C' = check pencil singularity, 'N' = no check\n" \
    "  restor (str): 'R' = restore original for max accuracy, 'N' = no restore\n" \
    "  a (ndarray): State matrix (n x n, F-order)\n" \
    "  e (ndarray): Descriptor matrix (n x n, F-order)\n" \
    "  b (ndarray): Input matrix (n x m, F-order; n x max(m,p) if job != 'C')\n" \
    "  c (ndarray): Output matrix (p x n, F-order; max(m,p) x n if job != 'C')\n" \
    "  tol (ndarray, optional): Tolerance array (3 elements):\n" \
    "    tol[0]: tolerance for controllability/observability\n" \
    "    tol[1]: tolerance for pencil regularity check\n" \
    "    tol[2]: threshold for pencil singularity check\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, b, c, nr, infred, iwork, info):\n" \
    "    Reduced Ar, Er, Br, Cr matrices; reduced order NR;\n" \
    "    INFRED array (phase reductions in [0-3], A/E sub-diagonals in [4-5],\n" \
    "    block count in [6]); IWORK with block dimensions; exit code"

#define DOC_TG01KD "Orthogonal equivalence transformation of SISO descriptor system.\n" \
    "\n" \
    "For a single-input single-output descriptor system (A, E, B, C) with E\n" \
    "upper triangular, computes (Q'*A*Z, Q'*E*Z, Q'*B, C*Z) via orthogonal\n" \
    "equivalence transformation so that Q'*B has only the first element\n" \
    "nonzero and Q'*E*Z remains upper triangular.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobe (str): 'U' = E is upper triangular, 'I' = E is identity\n" \
    "  compc (str): 'C' = transform C, 'N' = do not transform C\n" \
    "  compq (str): 'N', 'I', or 'U' - Q computation mode\n" \
    "  compz (str): 'N', 'I', or 'U' - Z computation mode\n" \
    "  a (ndarray): State matrix (n x n, F-order)\n" \
    "  e (ndarray): Descriptor matrix (n x n, F-order, if JOBE='U')\n" \
    "  b (ndarray): Input vector (n, F-order)\n" \
    "  c (ndarray): Output vector (n, F-order)\n" \
    "  q (ndarray, optional): Input Q for COMPQ='U' (n x n, F-order)\n" \
    "  z (ndarray, optional): Input Z for COMPZ='U' (n x n, F-order)\n" \
    "  incc (int, optional): Stride of C (default 1)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, b, c, q, z, info):\n" \
    "    Transformed Q'*A*Z, Q'*E*Z, Q'*B, C*Z, orthogonal Q and Z, exit code"

#define DOC_TG01KZ "Complex unitary equivalence transformation of SISO descriptor system.\n" \
    "\n" \
    "For a single-input single-output complex descriptor system (A, E, B, C)\n" \
    "with E upper triangular, computes (Q'*A*Z, Q'*E*Z, Q'*B, C*Z) via unitary\n" \
    "equivalence transformation so that Q'*B has only the first element\n" \
    "nonzero and Q'*E*Z remains upper triangular.\n" \
    "\n" \
    "Uses Givens rotations to annihilate the last N-1 elements of B in\n" \
    "reverse order while preserving the upper triangular form of E.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobe (str): 'U' = E is upper triangular, 'I' = E is identity\n" \
    "  compc (str): 'C' = transform C, 'N' = do not transform C\n" \
    "  compq (str): 'N', 'I', or 'U' - Q computation mode\n" \
    "  compz (str): 'N', 'I', or 'U' - Z computation mode\n" \
    "  a (ndarray): Complex state matrix (n x n, F-order)\n" \
    "  e (ndarray): Complex descriptor matrix (n x n, F-order, if JOBE='U')\n" \
    "  b (ndarray): Complex input vector (n, F-order)\n" \
    "  c (ndarray): Complex output vector (n, F-order)\n" \
    "  q (ndarray, optional): Complex input Q for COMPQ='U' (n x n, F-order)\n" \
    "  z (ndarray, optional): Complex input Z for COMPZ='U' (n x n, F-order)\n" \
    "  incc (int, optional): Stride of C (default 1)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, b, c, q, z, info):\n" \
    "    Transformed Q'*A*Z, Q'*E*Z, Q'*B, C*Z, unitary Q and Z, exit code"

#define DOC_TG01LD "Finite-infinite decomposition of descriptor system.\n" \
    "\n" \
    "Computes orthogonal transformation matrices Q and Z which reduce the\n" \
    "regular pole pencil A-lambda*E of the descriptor system (A-lambda*E,B,C)\n" \
    "to finite-infinite separated form:\n" \
    "  JOB='F': Q'*A*Z = (Af, *; 0, Ai), Q'*E*Z = (Ef, *; 0, Ei)\n" \
    "  JOB='I': Q'*A*Z = (Ai, *; 0, Af), Q'*E*Z = (Ei, *; 0, Ef)\n" \
    "where Af-lambda*Ef (Ef nonsingular) contains finite eigenvalues,\n" \
    "and Ai-lambda*Ei (Ai nonsingular) contains infinite eigenvalues\n" \
    "in staircase form.\n" \
    "\n" \
    "Optionally reduces Af to upper Hessenberg form (JOBA='H').\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'F' = finite-infinite, 'I' = infinite-finite separation\n" \
    "  joba (str): 'H' = reduce Af to Hessenberg, 'N' = keep unreduced\n" \
    "  compq (str): 'N' = no Q, 'I' = initialize Q to I, 'U' = update Q1*Q\n" \
    "  compz (str): 'N' = no Z, 'I' = initialize Z to I, 'U' = update Z1*Z\n" \
    "  n (int): Order of A and E (n >= 0)\n" \
    "  m (int): Number of columns of B (m >= 0)\n" \
    "  p (int): Number of rows of C (p >= 0)\n" \
    "  a (ndarray): State matrix (n x n, F-order)\n" \
    "  e (ndarray): Descriptor matrix (n x n, F-order)\n" \
    "  b (ndarray): Input matrix (n x m, F-order)\n" \
    "  c (ndarray): Output matrix (p x n, F-order)\n" \
    "  tol (float): Tolerance for rank decisions (use <= 0 for default)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, b, c, q, z, nf, nd, niblck, iblck, info):\n" \
    "    Transformed Q'*A*Z, Q'*E*Z, Q'*B, C*Z, orthogonal Q and Z,\n" \
    "    NF = number of finite eigenvalues, ND = non-dynamic infinite count,\n" \
    "    NIBLCK = number of infinite blocks minus one, IBLCK = block dimensions,\n" \
    "    INFO = 0 success, 1 = pencil not regular"

#define DOC_TG01LY "Finite-infinite decomposition of structured descriptor system.\n" \
    "\n" \
    "Reduces a regular pole pencil A-lambda*E of descriptor system (A-lambda*E,B,C)\n" \
    "from structured form to finite-infinite separated form. Input matrices must\n" \
    "have special structure where E has rank RANKE with nonzero block in top-left.\n" \
    "\n" \
    "Parameters:\n" \
    "  compq (bool): True to accumulate Q transformations, False otherwise\n" \
    "  compz (bool): True to accumulate Z transformations, False otherwise\n" \
    "  ranke (int): Rank of E matrix (0 <= ranke <= n)\n" \
    "  rnka22 (int): Rank of A22 block (0 <= rnka22 <= n - ranke)\n" \
    "  a (ndarray): State matrix (n x n, F-order)\n" \
    "  e (ndarray): Descriptor matrix (n x n, F-order)\n" \
    "  b (ndarray): Input matrix (n x m, F-order)\n" \
    "  c (ndarray): Output matrix (p x n, F-order)\n" \
    "  q (ndarray): Left transformation matrix (n x n if compq, else 1 x 1)\n" \
    "  z (ndarray): Right transformation matrix (n x n if compz, else 1 x 1)\n" \
    "  tol (float, optional): Tolerance for rank decisions (default: n^2 * eps)\n" \
    "  ldwork (int, optional): Workspace size (-1 for query)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, b, c, q, z, nf, niblck, iblck, info):\n" \
    "    Transformed Q'*A*Z, Q'*E*Z, Q'*B, C*Z, orthogonal Q and Z,\n" \
    "    NF = number of finite eigenvalues,\n" \
    "    NIBLCK = number of infinite blocks,\n" \
    "    IBLCK = block dimensions array,\n" \
    "    INFO = 0 success, 1 = pencil not regular, <0 = param error"

#define DOC_TG01MD "Finite-infinite generalized real Schur form decomposition.\n" \
    "\n" \
    "Computes orthogonal transformation matrices Q and Z which reduce the\n" \
    "regular pole pencil A-lambda*E of the descriptor system (A-lambda*E,B,C)\n" \
    "to generalized real Schur form with finite-infinite separation:\n" \
    "  JOB='F': Q'*A*Z = (Af, *; 0, Ai), Q'*E*Z = (Ef, *; 0, Ei)\n" \
    "  JOB='I': Q'*A*Z = (Ai, *; 0, Af), Q'*E*Z = (Ei, *; 0, Ef)\n" \
    "where (Af,Ef) is in generalized real Schur form with Ef nonsingular\n" \
    "and upper triangular, Af in real Schur form. The subpencil\n" \
    "Af-lambda*Ef contains the finite eigenvalues.\n" \
    "The pair (Ai,Ei) has both Ai and Ei upper triangular. The subpencil\n" \
    "Ai-lambda*Ei (Ai nonsingular, Ei nilpotent) contains infinite\n" \
    "eigenvalues in block staircase form.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'F' = finite-infinite, 'I' = infinite-finite separation\n" \
    "  n (int): Order of A and E (n >= 0)\n" \
    "  m (int): Number of columns of B (m >= 0)\n" \
    "  p (int): Number of rows of C (p >= 0)\n" \
    "  a (ndarray): State matrix (n x n, F-order)\n" \
    "  e (ndarray): Descriptor matrix (n x n, F-order)\n" \
    "  b (ndarray): Input matrix (n x m, F-order)\n" \
    "  c (ndarray): Output matrix (p x n, F-order)\n" \
    "  tol (float): Tolerance for rank decisions (use <= 0 for default)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, b, c, alphar, alphai, beta, q, z, nf, nd, niblck, iblck, info):\n" \
    "    Transformed Q'*A*Z, Q'*E*Z, Q'*B, C*Z;\n" \
    "    ALPHAR, ALPHAI, BETA = generalized eigenvalue components\n" \
    "      (eigenvalues = (ALPHAR + i*ALPHAI) / BETA);\n" \
    "    Orthogonal Q and Z;\n" \
    "    NF = number of finite eigenvalues;\n" \
    "    ND = non-dynamic infinite eigenvalue count;\n" \
    "    NIBLCK = number of infinite blocks minus one;\n" \
    "    IBLCK = block dimensions;\n" \
    "    INFO = 0 success, 1 = pencil not regular, 2 = QZ failed"

#define DOC_TG01ND "Finite-infinite block-diagonal decomposition of descriptor system.\n" \
    "\n" \
    "Computes equivalence transformation matrices Q and Z which reduce the\n" \
    "regular pole pencil A-lambda*E of the descriptor system (A-lambda*E,B,C)\n" \
    "to fully block-diagonal form:\n" \
    "  JOB='F': Q*A*Z = diag(Af, Ai), Q*E*Z = diag(Ef, Ei)\n" \
    "  JOB='I': Q*A*Z = diag(Ai, Af), Q*E*Z = diag(Ei, Ef)\n" \
    "where (Af,Ef) is in generalized real Schur form with Ef nonsingular\n" \
    "and upper triangular, and Af in real Schur form. The subpencil\n" \
    "Af-lambda*Ef contains the finite eigenvalues. The pair (Ai,Ei) is in\n" \
    "generalized real Schur form with Ai nonsingular and Ei nilpotent\n" \
    "(upper triangular in block staircase form).\n" \
    "\n" \
    "Unlike tg01md, this routine produces FULLY block-diagonal form by\n" \
    "solving generalized Sylvester equations to eliminate off-diagonal blocks.\n" \
    "\n" \
    "Parameters:\n" \
    "  job (str): 'F' = finite-infinite, 'I' = infinite-finite separation\n" \
    "  jobt (str): 'D' = direct Q, Z; 'I' = inverse inv(Q), inv(Z)\n" \
    "  n (int): Order of A and E (n >= 0)\n" \
    "  m (int): Number of columns of B (m >= 0)\n" \
    "  p (int): Number of rows of C (p >= 0)\n" \
    "  a (ndarray): State matrix (n x n, F-order)\n" \
    "  e (ndarray): Descriptor matrix (n x n, F-order)\n" \
    "  b (ndarray): Input matrix (n x m, F-order)\n" \
    "  c (ndarray): Output matrix (p x n, F-order)\n" \
    "  tol (float): Tolerance for rank decisions (use <= 0 for default)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, b, c, alphar, alphai, beta, q, z, nf, nd, niblck, iblck, info):\n" \
    "    Block-diagonal Q*A*Z, Q*E*Z, transformed Q*B, C*Z;\n" \
    "    ALPHAR, ALPHAI, BETA = generalized eigenvalue components\n" \
    "      (eigenvalues = (ALPHAR + i*ALPHAI) / BETA);\n" \
    "    Transformation matrices Q, Z (or inv(Q), inv(Z) if JOBT='I');\n" \
    "    NF = number of finite eigenvalues;\n" \
    "    ND = non-dynamic infinite eigenvalue count;\n" \
    "    NIBLCK = number of infinite blocks minus one;\n" \
    "    IBLCK = block dimensions;\n" \
    "    INFO = 0 success, 1 = not regular, 2 = QZ failed, 3 = Sylvester failed"

#define DOC_TG01NX "Block-diagonal decomposition of descriptor system in generalized Schur form.\n" \
    "\n" \
    "Computes equivalence transformation matrices Q and Z which reduce\n" \
    "the regular pole pencil A-lambda*E of the descriptor system (A-lambda*E,B,C),\n" \
    "with (A,E) in generalized real Schur form, to block-diagonal form:\n" \
    "  Q*A*Z = diag(A1, A2),  Q*E*Z = diag(E1, E2)\n" \
    "where (A1,E1) and (A2,E2) have no common generalized eigenvalues.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobt (str): 'D' = compute Q, Z; 'I' = compute inv(Q), inv(Z)\n" \
    "  n (int): Order of A, E; rows of B; columns of C\n" \
    "  m (int): Number of columns of B\n" \
    "  p (int): Number of rows of C\n" \
    "  ndim (int): Dimension of leading diagonal blocks (0 <= ndim <= n)\n" \
    "  a (ndarray): State matrix in Schur form (n x n, F-order)\n" \
    "  e (ndarray): Descriptor matrix, upper triangular (n x n, F-order)\n" \
    "  b (ndarray): Input matrix (n x m, F-order)\n" \
    "  c (ndarray): Output matrix (p x n, F-order)\n" \
    "  q (ndarray): Left transformation matrix Q1 (n x n, F-order)\n" \
    "  z (ndarray): Right transformation matrix Z1 (n x n, F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, b, c, q, z, info):\n" \
    "    Block-diagonal A and E, transformed B and C, updated Q and Z, exit code"

#define DOC_TG01OA "Orthogonal equivalence transformation of SISO descriptor system.\n" \
    "\n" \
    "For a single-input single-output descriptor system given by the\n" \
    "system matrix [D C; B A-s*E] with E upper triangular, computes a\n" \
    "transformed system (Q'*A*Z, Q'*E*Z, Q'*B, C*Z) via orthogonal\n" \
    "equivalence transformation so that Q'*B has only the first element\n" \
    "nonzero and Q'*E*Z remains upper triangular.\n" \
    "\n" \
    "Uses Givens rotations to annihilate the last N-1 elements of B in\n" \
    "reverse order while preserving the upper triangular form of E.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobe (str): 'U' = E is upper triangular, 'I' = E is identity\n" \
    "  dcba (ndarray): System matrix [D C; B A] ((n+1) x (n+1), F-order)\n" \
    "  e (ndarray): Descriptor matrix E (n x n, F-order, if JOBE='U')\n" \
    "\n" \
    "Returns:\n" \
    "  (dcba, e, info):\n" \
    "    dcba: Transformed [D C*Z; Q'*B Q'*A*Z]\n" \
    "    e: Transformed Q'*E*Z (if JOBE='U')\n" \
    "    info: 0 = success, <0 = invalid parameter"

#define DOC_TG01OB "Complex SISO descriptor system unitary equivalence transformation.\n" \
    "\n" \
    "For a single-input single-output complex descriptor system stored in\n" \
    "compact form [D C; B A] with E upper triangular, computes a transformed\n" \
    "system via unitary equivalence transformation so that the transformed B\n" \
    "has only the first element nonzero and E remains upper triangular.\n" \
    "\n" \
    "Uses Givens rotations to annihilate the last N-1 elements of B in\n" \
    "reverse order while preserving the upper triangular form of E.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobe (str): 'U' = E is upper triangular, 'I' = E is identity\n" \
    "  dcba (ndarray): System matrix [(N+1) x (N+1), F-order] containing:\n" \
    "    [D  C]  where D is scalar, C is 1xN, B is Nx1, A is NxN\n" \
    "    [B  A]\n" \
    "  e (ndarray): Complex descriptor matrix (N x N, F-order, if JOBE='U')\n" \
    "\n" \
    "Returns:\n" \
    "  (dcba, e, info):\n" \
    "    Transformed [D C*Z; Q'*B Q'*A*Z], transformed Q'*E*Z, exit code.\n" \
    "    D is unchanged. Q'*B has only first element nonzero."

#define DOC_TG01OD "Reduce SISO descriptor system so feedthrough has large magnitude.\n" \
    "\n" \
    "Computes for a single-input single-output descriptor system, given by\n" \
    "the system matrix:\n" \
    "    [ D     C    ]\n" \
    "    [ B  A - s*E ]\n" \
    "with E nonsingular, a reduced system matrix:\n" \
    "    [ d     c    ]\n" \
    "    [ b  a - s*e ]\n" \
    "such that d has a \"sufficiently\" large magnitude.\n" \
    "\n" \
    "Uses Householder transformations and Givens rotations. If E is a general\n" \
    "matrix, it is first triangularized using QR decomposition.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobe (str): 'G' = E is general nonsingular, 'I' = E is identity\n" \
    "  dcba (ndarray): System matrix [D C; B A] ((n+1) x (n+1), F-order)\n" \
    "  e (ndarray): Descriptor matrix E (n x n, F-order, if JOBE='G')\n" \
    "  tol (float, optional): Tolerance for d magnitude check (default: EPS^(3/4))\n" \
    "\n" \
    "Returns:\n" \
    "  (dcba, e, nz, g, info):\n" \
    "    dcba: Leading (nz+1) x (nz+1) contains reduced [d c; b a]\n" \
    "    e: Leading nz x nz contains reduced e (if JOBE='G')\n" \
    "    nz: Order of reduced system\n" \
    "    g: Gain of reduced system\n" \
    "    info: 0 = success, <0 = invalid parameter"

#define DOC_TG01OZ "Reduce complex SISO descriptor system so feedthrough has large magnitude.\n" \
    "\n" \
    "Computes for a single-input single-output complex descriptor system,\n" \
    "given by the system matrix:\n" \
    "    [ D     C    ]\n" \
    "    [ B  A - s*E ]\n" \
    "with E nonsingular, a reduced system matrix:\n" \
    "    [ d     c    ]\n" \
    "    [ b  a - s*e ]\n" \
    "such that d has a sufficiently large magnitude.\n" \
    "\n" \
    "Uses Householder transformations and Givens rotations. If E is a general\n" \
    "matrix, it is first triangularized using QR decomposition.\n" \
    "\n" \
    "Parameters:\n" \
    "  jobe (str): 'G' = E is general nonsingular, 'I' = E is identity\n" \
    "  dcba (ndarray): Complex system matrix [D C; B A] ((n+1) x (n+1), F-order)\n" \
    "  e (ndarray): Complex descriptor matrix E (n x n, F-order, if JOBE='G')\n" \
    "  tol (float, optional): Tolerance for d magnitude check (default: EPS^(3/4))\n" \
    "\n" \
    "Returns:\n" \
    "  (dcba, e, nz, g, info):\n" \
    "    dcba: Leading (nz+1) x (nz+1) contains reduced [d c; b a]\n" \
    "    e: Leading nz x nz contains reduced e (if JOBE='G')\n" \
    "    nz: Order of reduced system\n" \
    "    g: Complex gain of reduced system\n" \
    "    info: 0 = success, <0 = invalid parameter"

#define DOC_TG01PD "Bi-domain spectral splitting of a subpencil of a descriptor system.\n" \
    "\n" \
    "Computes orthogonal transformation matrices Q and Z which reduce the\n" \
    "regular pole pencil A-lambda*E of the descriptor system (A-lambda*E,B,C)\n" \
    "to generalized real Schur form with ordered generalized eigenvalues.\n" \
    "The pair (A,E) is reduced so that eigenvalues in the domain of interest\n" \
    "are in the leading diagonal blocks.\n" \
    "\n" \
    "For continuous-time (DICO='C'):\n" \
    "  Domain of interest: Re(lambda) < ALPHA (STDOM='S') or Re(lambda) > ALPHA (STDOM='U')\n" \
    "For discrete-time (DICO='D'):\n" \
    "  Domain of interest: |lambda| < ALPHA (STDOM='S') or |lambda| > ALPHA (STDOM='U')\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' = continuous-time, 'D' = discrete-time\n" \
    "  stdom (str): 'S' = stability domain, 'U' = instability domain\n" \
    "  jobae (str): 'G' = general (A,E), 'S' = (A,E) in Schur form\n" \
    "  compq (str): 'I' = initialize Q to identity, 'U' = update existing Q (only if JOBAE='S')\n" \
    "  compz (str): 'I' = initialize Z to identity, 'U' = update existing Z (only if JOBAE='S')\n" \
    "  n (int): Order of matrices A and E\n" \
    "  m (int): Number of columns of B\n" \
    "  p (int): Number of rows of C\n" \
    "  nlow (int): Lower boundary index (min(1,n) if JOBAE='G')\n" \
    "  nsup (int): Upper boundary index (n if JOBAE='G')\n" \
    "  alpha (float): Domain boundary (>= 0 for discrete-time)\n" \
    "  a (ndarray): N-by-N state matrix (F-order)\n" \
    "  e (ndarray): N-by-N descriptor matrix (F-order)\n" \
    "  b (ndarray): N-by-M input matrix (F-order)\n" \
    "  c (ndarray): P-by-N output matrix (F-order)\n" \
    "  q (ndarray, optional): N-by-N orthogonal matrix (if COMPQ='U')\n" \
    "  z (ndarray, optional): N-by-N orthogonal matrix (if COMPZ='U')\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, b, c, q, z, ndim, alphar, alphai, beta, info):\n" \
    "    a: Transformed Q'*A*Z in Schur form\n" \
    "    e: Transformed Q'*E*Z upper triangular\n" \
    "    b: Transformed Q'*B\n" \
    "    c: Transformed C*Z\n" \
    "    q: Left orthogonal transformation matrix\n" \
    "    z: Right orthogonal transformation matrix\n" \
    "    ndim: Number of eigenvalues in domain of interest\n" \
    "    alphar: Real parts of generalized eigenvalues\n" \
    "    alphai: Imaginary parts of generalized eigenvalues\n" \
    "    beta: Denominators of generalized eigenvalues\n" \
    "    info: 0 = success, <0 = invalid param, 1 = QZ failed, 2 = ordering failed"

#define DOC_TG01QD "Three-domain spectral splitting of a descriptor system.\n" \
    "\n" \
    "Computes orthogonal transformation matrices Q and Z which reduce the\n" \
    "regular pole pencil A-lambda*E of the descriptor system (A-lambda*E,B,C)\n" \
    "to generalized real Schur form with ordered generalized eigenvalues.\n" \
    "The pair (A,E) is reduced to three-block form:\n" \
    "\n" \
    "           ( A1  *   *  )             ( E1  *   *  )\n" \
    "  Q'*A*Z = ( 0   A2  *  ) ,  Q'*E*Z = ( 0   E2  *  )\n" \
    "           ( 0   0   A3 )             ( 0   0   E3 )\n" \
    "\n" \
    "If JOBFI='F': (A1,E1) = finite in domain, (A2,E2) = finite outside, (A3,E3) = infinite\n" \
    "If JOBFI='I': (A1,E1) = infinite, (A2,E2) = finite in domain, (A3,E3) = finite outside\n" \
    "\n" \
    "For continuous-time (DICO='C'):\n" \
    "  Domain of interest: Re(lambda) < ALPHA (STDOM='S') or Re(lambda) > ALPHA (STDOM='U')\n" \
    "For discrete-time (DICO='D'):\n" \
    "  Domain of interest: |lambda| < ALPHA (STDOM='S') or |lambda| > ALPHA (STDOM='U')\n" \
    "For STDOM='N': No spectral ordering of finite part.\n" \
    "\n" \
    "Parameters:\n" \
    "  dico (str): 'C' = continuous-time, 'D' = discrete-time\n" \
    "  stdom (str): 'S' = stability domain, 'U' = instability domain, 'N' = no ordering\n" \
    "  jobfi (str): 'F' = finite first, 'I' = infinite first\n" \
    "  n (int): Order of matrices A and E\n" \
    "  m (int): Number of columns of B\n" \
    "  p (int): Number of rows of C\n" \
    "  alpha (float): Domain boundary (>= 0 for discrete-time)\n" \
    "  a (ndarray): N-by-N state matrix (F-order)\n" \
    "  e (ndarray): N-by-N descriptor matrix (F-order)\n" \
    "  b (ndarray): N-by-M input matrix (F-order)\n" \
    "  c (ndarray): P-by-N output matrix (F-order)\n" \
    "  tol (float): Tolerance for rank decisions (<= 0 for default)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, b, c, n1, n2, n3, nd, niblck, iblck, q, z, alphar, alphai, beta, info):\n" \
    "    a: Transformed Q'*A*Z in Schur form\n" \
    "    e: Transformed Q'*E*Z upper triangular\n" \
    "    b: Transformed Q'*B\n" \
    "    c: Transformed C*Z\n" \
    "    n1, n2, n3: Eigenvalue counts in each block\n" \
    "    nd: Number of non-dynamic infinite eigenvalues\n" \
    "    niblck: Number of infinite blocks minus one\n" \
    "    iblck: Block dimensions for infinite part\n" \
    "    q, z: Orthogonal transformation matrices\n" \
    "    alphar, alphai, beta: Generalized eigenvalues\n" \
    "    info: 0 = success, <0 = invalid param, 1 = not regular, 2 = QZ failed, 3 = ordering failed"

#define DOC_TG01WD "Reduce descriptor system to generalized real Schur form.\n" \
    "\n" \
    "Reduces the pair (A,E) to a real generalized Schur form using an\n" \
    "orthogonal equivalence transformation (A,E) <-- (Q'*A*Z, Q'*E*Z)\n" \
    "and applies the transformation to matrices B and C:\n" \
    "B <-- Q'*B and C <-- C*Z.\n" \
    "\n" \
    "After reduction:\n" \
    "- A is in upper quasi-triangular form (elements below first subdiagonal = 0)\n" \
    "- E is in upper triangular form\n" \
    "\n" \
    "The generalized eigenvalues of (A,E) are returned as\n" \
    "(ALPHAR(j) + i*ALPHAI(j)) / BETA(j), j=1,...,N.\n" \
    "\n" \
    "Parameters:\n" \
    "  n (int): Order of matrices A and E (N >= 0)\n" \
    "  m (int): Number of columns of B (M >= 0)\n" \
    "  p (int): Number of rows of C (P >= 0)\n" \
    "  a (ndarray): N-by-N state dynamics matrix (F-order)\n" \
    "  e (ndarray): N-by-N descriptor matrix (F-order)\n" \
    "  b (ndarray): N-by-M input matrix (F-order)\n" \
    "  c (ndarray): P-by-N output matrix (F-order)\n" \
    "\n" \
    "Returns:\n" \
    "  (a, e, b, c, q, z, alphar, alphai, beta, info):\n" \
    "    a: Transformed Q'*A*Z in upper quasi-triangular form\n" \
    "    e: Transformed Q'*E*Z in upper triangular form\n" \
    "    b: Transformed Q'*B\n" \
    "    c: Transformed C*Z\n" \
    "    q: N-by-N left orthogonal transformation matrix\n" \
    "    z: N-by-N right orthogonal transformation matrix\n" \
    "    alphar: Real parts of generalized eigenvalues (size N)\n" \
    "    alphai: Imaginary parts of generalized eigenvalues (size N)\n" \
    "    beta: Denominators of generalized eigenvalues (size N)\n" \
    "    info: 0 = success, <0 = invalid param -i, >0 = QZ algorithm failed"

#define DOC_UD01BD "Read/copy coefficients of a matrix polynomial.\n" \
    "\n" \
    "Copies the coefficients of a matrix polynomial P(s) from input data array\n" \
    "to a 3D output array. The polynomial is:\n" \
    "    P(s) = P(0) + P(1)*s + ... + P(dp-1)*s^(dp-1) + P(dp)*s^dp\n" \
    "\n" \
    "Each coefficient P(k) is an MP-by-NP matrix. The input data contains\n" \
    "(DP+1) matrices stored row by row (matching Fortran file read order).\n" \
    "\n" \
    "Parameters:\n" \
    "  mp (int): Number of rows of each coefficient matrix (mp >= 1)\n" \
    "  np (int): Number of columns of each coefficient matrix (np >= 1)\n" \
    "  dp (int): Degree of the polynomial (dp >= 0)\n" \
    "  data (ndarray): Input data array of length mp*np*(dp+1) containing\n" \
    "                  coefficient matrices P(0), P(1), ..., P(dp) stored\n" \
    "                  row by row\n" \
    "\n" \
    "Returns:\n" \
    "  (p, info):\n" \
    "    p: 3D array (mp, np, dp+1) where p[:,:,k] = P(k), coefficient of s^k\n" \
    "    info: Exit code (0=success, -1=mp<1, -2=np<1, -3=dp<0)"

#define DOC_UD01CD "Read sparse matrix polynomial coefficients.\n" \
    "\n" \
    "Constructs a matrix polynomial from sparse element specifications:\n" \
    "    P(s) = P(0) + P(1)*s + ... + P(dp-1)*s^(dp-1) + P(dp)*s^dp\n" \
    "\n" \
    "Each nonzero polynomial element P_{i,j}(s) is specified by its row index,\n" \
    "column index, degree, and coefficients. All unspecified elements are zero.\n" \
    "\n" \
    "Parameters:\n" \
    "  mp (int): Number of rows of each coefficient matrix (mp >= 1)\n" \
    "  np (int): Number of columns of each coefficient matrix (np >= 1)\n" \
    "  dp (int): Maximum degree of the matrix polynomial (dp >= 0)\n" \
    "  rows (ndarray): Array of row indices (1-based), shape (nelem,)\n" \
    "  cols (ndarray): Array of column indices (1-based), shape (nelem,)\n" \
    "  degrees (ndarray): Array of polynomial degrees for each element, shape (nelem,)\n" \
    "  coeffs (ndarray): Concatenated coefficients for all elements.\n" \
    "                    Element e has (degrees[e]+1) coefficients: c_0, c_1, ..., c_d\n" \
    "                    representing P_{i,j}(s) = c_0 + c_1*s + ... + c_d*s^d\n" \
    "\n" \
    "Returns:\n" \
    "  (p, info):\n" \
    "    p: 3D array (mp, np, dp+1) where p[:,:,k] = P(k), coefficient of s^k\n" \
    "    info: Exit code:\n" \
    "          0 = success\n" \
    "          -1 = mp < 1\n" \
    "          -2 = np < 1\n" \
    "          -3 = dp < 0\n" \
    "          1 = warning: some indices/degrees out of bounds (skipped)"

#define DOC_UD01DD "Read/construct a sparse matrix from COO format.\n" \
    "\n" \
    "Initializes an M-by-N matrix to zero, then assigns nonzero elements\n" \
    "from sparse COO (Coordinate) format input arrays.\n" \
    "\n" \
    "The original Fortran routine reads from a file; this C implementation\n" \
    "accepts sparse data directly as arrays of row indices, column indices,\n" \
    "and values.\n" \
    "\n" \
    "Parameters:\n" \
    "  m (int): Number of rows of the matrix A (m >= 0)\n" \
    "  n (int): Number of columns of the matrix A (n >= 0)\n" \
    "  rows (ndarray): Array of row indices (1-based, Fortran convention)\n" \
    "  cols (ndarray): Array of column indices (1-based, Fortran convention)\n" \
    "  vals (ndarray): Array of values to assign\n" \
    "\n" \
    "Returns:\n" \
    "  (a, info):\n" \
    "    a: Output matrix (m, n) with sparse entries assigned\n" \
    "    info: Exit code:\n" \
    "          0 = success\n" \
    "          -1 = m < 0\n" \
    "          -2 = n < 0\n" \
    "          1 = warning: some indices were out of bounds (skipped)"

#define DOC_UD01MD "Print an M-by-N real matrix row by row.\n" \
    "\n" \
    "Prints the elements of A to 7 significant figures with column headers.\n" \
    "The matrix is printed in blocks of L columns.\n" \
    "\n" \
    "Parameters:\n" \
    "  m (int): Number of rows of matrix A (m >= 1)\n" \
    "  n (int): Number of columns of matrix A (n >= 1)\n" \
    "  l (int): Number of elements per line (1 <= l <= 5)\n" \
    "  a (ndarray): Array of dimension (m, n), the matrix to print\n" \
    "  text (str): Title caption (up to 72 characters)\n" \
    "\n" \
    "Returns:\n" \
    "  (result, info):\n" \
    "    result: Formatted string representation of the matrix\n" \
    "    info: Exit code:\n" \
    "          0 = success\n" \
    "          -1 = m < 1\n" \
    "          -2 = n < 1\n" \
    "          -3 = l < 1 or l > 5\n" \
    "          -6 = lda < m\n" \
    "          -100 = output buffer too small"

#define DOC_UD01MZ "Print an M-by-N complex matrix row by row.\n" \
    "\n" \
    "Prints the elements of A to 7 significant figures with column headers.\n" \
    "Complex numbers are displayed as real+imaginary pairs. The matrix is\n" \
    "printed in blocks of L columns (max 3 for complex due to wider format).\n" \
    "\n" \
    "Parameters:\n" \
    "  a (ndarray): Complex matrix of dimension (m, n) to print\n" \
    "  text (str): Title caption (up to 72 characters)\n" \
    "  l (int, optional): Number of elements per line (1 <= l <= 3, default=3)\n" \
    "\n" \
    "Returns:\n" \
    "  (result, info):\n" \
    "    result: Formatted string representation of the matrix\n" \
    "    info: Exit code:\n" \
    "          0 = success\n" \
    "          -1 = m < 1\n" \
    "          -2 = n < 1\n" \
    "          -3 = l < 1 or l > 3\n" \
    "          -5 = lda < m\n" \
    "          -6 = output buffer too small"

#define DOC_UD01ND "Print the coefficient matrices of a matrix polynomial.\n" \
    "\n" \
    "Prints the MP-by-NP coefficient matrices of a matrix polynomial:\n" \
    "    P(s) = P(0) + P(1)*s + ... + P(dp-1)*s^(dp-1) + P(dp)*s^dp\n" \
    "\n" \
    "The elements are output to 7 significant figures. Each coefficient matrix\n" \
    "is printed with a title showing the polynomial degree.\n" \
    "\n" \
    "Parameters:\n" \
    "  mp (int): Number of rows of each coefficient matrix (mp >= 1)\n" \
    "  np (int): Number of columns of each coefficient matrix (np >= 1)\n" \
    "  dp (int): Degree of the matrix polynomial (dp >= 0)\n" \
    "  l (int): Number of elements per line (1 <= l <= 5)\n" \
    "  p (ndarray): 3D array (mp, np, dp+1) containing polynomial coefficients.\n" \
    "               p[:,:,k] = P(k), the coefficient matrix of s^k\n" \
    "  text (str): Title caption (up to 72 characters).\n" \
    "              If blank, coefficient matrices are separated by empty lines.\n" \
    "\n" \
    "Returns:\n" \
    "  (result, info):\n" \
    "    result: Formatted string representation of the polynomial\n" \
    "    info: Exit code:\n" \
    "          0 = success\n" \
    "          -1 = mp < 1\n" \
    "          -2 = np < 1\n" \
    "          -3 = dp < 0\n" \
    "          -4 = l < 1 or l > 5\n" \
    "          -6 = ldp1 < mp\n" \
    "          -7 = ldp2 < np\n" \
    "          -100 = output buffer too small"

#define DOC_UE01MD "Get machine-specific parameters for SLICOT routines.\n" \
    "\n" \
    "Provides an extension of the LAPACK routine ILAENV to return\n" \
    "machine-specific parameters for SLICOT routines. The default values\n" \
    "aim to give good performance on a wide range of computers.\n" \
    "\n" \
    "Parameters:\n" \
    "  ispec (int): Specifies the parameter to be returned:\n" \
    "               1 = optimal blocksize\n" \
    "               2 = minimum block size for block routine\n" \
    "               3 = crossover point (use unblocked for N < this)\n" \
    "               4 = number of shifts (product eigenvalue routine)\n" \
    "               8 = crossover point for multishift QR\n" \
    "  name (str): Name of the calling subroutine (upper or lower case)\n" \
    "  opts (str): Character options to the subroutine, concatenated\n" \
    "  n1 (int): First problem dimension\n" \
    "  n2 (int): Second problem dimension\n" \
    "  n3 (int): Third problem dimension\n" \
    "\n" \
    "Returns:\n" \
    "  result (int): The parameter value according to ISPEC, or -1 for invalid ISPEC"

#define DOC_ZGEGS "Compute generalized Schur form for complex matrix pair.\n" \
    "\n" \
    "This routine is deprecated and replaced by ZGGES.\n" \
    "\n" \
    "Computes the generalized Schur factorization of a complex matrix pair (A,B):\n" \
    "    A = Q * S * Z^H\n" \
    "    B = Q * T * Z^H\n" \
    "\n" \
    "where Q and Z are unitary and S and T are upper triangular.\n" \
    "The generalized eigenvalues are alpha(j)/beta(j) where\n" \
    "alpha = diag(S) and beta = diag(T).\n" \
    "\n" \
    "Parameters:\n" \
    "  jobvsl (str): 'N' = no left Schur vectors, 'V' = compute left Schur vectors Q\n" \
    "  jobvsr (str): 'N' = no right Schur vectors, 'V' = compute right Schur vectors Z\n" \
    "  a (ndarray): Complex matrix A (n x n, F-order), overwritten with S on exit\n" \
    "  b (ndarray): Complex matrix B (n x n, F-order), overwritten with T on exit\n" \
    "  lwork (int, optional): Workspace size (-1 for query, default auto)\n" \
    "\n" \
    "Returns:\n" \
    "  (s, t, alpha, beta, vsl, vsr, info):\n" \
    "    s: Upper triangular Schur form of A (n x n)\n" \
    "    t: Upper triangular Schur form of B (n x n)\n" \
    "    alpha: Eigenvalue numerators from diag(S) (n,)\n" \
    "    beta: Eigenvalue denominators from diag(T) (n,)\n" \
    "    vsl: Left Schur vectors Q (n x n) if jobvsl='V', else empty\n" \
    "    vsr: Right Schur vectors Z (n x n) if jobvsr='V', else empty\n" \
    "    info: Exit code (0=success, <0=param error, 1..N=QZ failed, >N=LAPACK error)"

#define DOC_ZGEGV "Compute generalized eigenvalues and eigenvectors for complex matrix pair.\n" \
    "\n" \
    "This routine is deprecated and replaced by ZGGEV.\n" \
    "\n" \
    "Computes the eigenvalues and, optionally, the left and/or right eigenvectors\n" \
    "of a complex matrix pair (A,B). Given A*x = lambda*B*x, returns alpha and beta\n" \
    "such that lambda = alpha/beta.\n" \
    "\n" \
    "The eigenvalues are returned as alpha(j)/beta(j) to handle cases where\n" \
    "lambda may be infinite (when beta(j) = 0).\n" \
    "\n" \
    "Parameters:\n" \
    "  jobvl (str): 'N' = no left eigenvectors, 'V' = compute left eigenvectors\n" \
    "  jobvr (str): 'N' = no right eigenvectors, 'V' = compute right eigenvectors\n" \
    "  a (ndarray): Complex matrix A (n x n, F-order), overwritten on exit\n" \
    "  b (ndarray): Complex matrix B (n x n, F-order), overwritten on exit\n" \
    "  lwork (int, optional): Workspace size (-1 for query, default auto)\n" \
    "\n" \
    "Returns:\n" \
    "  (alpha, beta, vl, vr, info):\n" \
    "    alpha: Complex eigenvalue numerators (n,)\n" \
    "    beta: Complex eigenvalue denominators (n,)\n" \
    "    vl: Left eigenvectors (n x n) if jobvl='V', else empty\n" \
    "    vr: Right eigenvectors (n x n) if jobvr='V', else empty\n" \
    "    info: Exit code (0=success, <0=param error, 1..N=QZ failed, >N=LAPACK error)"

#endif /* SLICOT_DOCSTRINGS_H */
