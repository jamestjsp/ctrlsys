"""
Tests for MB04ED: Eigenvalues and orthogonal decomposition of a real
skew-Hamiltonian/skew-Hamiltonian pencil in factored form.

Computes eigenvalues of aS - bT where:
  S = J*Z'*J'*Z
  T = [[B, F], [G, B']]
  J = [[0, I], [-I, 0]]

Tests:
1. Basic case from HTML docs (N=8, JOB='T', COMPQ='I', COMPU='I')
2. Edge case: N=0 (quick return)
3. Mathematical property: orthogonality of Q and U matrices
4. Eigenvalues only mode (JOB='E')
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from fortran_reference import run_fortran_driver


def _mb04ed_data_file():
    return "SLICOT-Reference/examples/data/MB04ED.dat"


def _mb04ed_tokens():
    tokens = []
    with open(_mb04ed_data_file(), encoding="ascii") as data_file:
        next(data_file)
        for line in data_file:
            tokens.extend(line.split())
    return tokens


def _read_mb04ed_example_data():
    tokens = _mb04ed_tokens()
    offset = 0
    job, compq, compu = tokens[:3]
    n = int(tokens[3])
    offset = 4
    m = n // 2

    def matrix(rows, cols):
        nonlocal offset
        data = np.array([float(x) for x in tokens[offset:offset + rows * cols]])
        offset += rows * cols
        return np.asfortranarray(data.reshape(rows, cols))

    z = matrix(n, n)
    b = matrix(m, m)
    fg = matrix(m, m + 1)
    return job, compq, compu, z, b, fg


def _mb04ed_example_fortran_source():
    data_path = _mb04ed_data_file()
    return f"""
program mb04ed_reference
  implicit none
  integer, parameter :: nmax = 60
  integer, parameter :: ldb = nmax/2, ldfg = nmax/2, ldq = nmax
  integer, parameter :: ldu1 = nmax/2, ldu2 = nmax/2, ldwork = 3*nmax**2/2 + max(nmax,24) + 3
  integer, parameter :: ldz = nmax, liwork = nmax + 9
  character :: job, compq, compu
  integer :: i, info, j, n
  integer :: iwork(liwork)
  double precision :: alphai(nmax/2), alphar(nmax/2), b(ldb,nmax/2), beta(nmax/2)
  double precision :: dwork(ldwork), fg(ldfg,nmax/2+1), q(ldq,nmax)
  double precision :: u1(ldu1,nmax/2), u2(ldu2,nmax/2), z(ldz,nmax)

  open(unit=5, file='{data_path}', status='old')
  read(5, *)
  read(5, *) job, compq, compu, n
  read(5, *) ((z(i,j), j = 1,n), i = 1,n)
  read(5, *) ((b(i,j), j = 1,n/2), i = 1,n/2)
  read(5, *) ((fg(i,j), j = 1,n/2+1), i = 1,n/2)
  close(5)

  call MB04ED(job, compq, compu, n, z, ldz, b, ldb, fg, ldfg, q, ldq, &
       u1, ldu1, u2, ldu2, alphar, alphai, beta, iwork, liwork, dwork, ldwork, info)

  write(*,'(I0)') info
  do i = 1, n
     write(*,'(*(1X,ES24.16))') (z(i,j), j = 1,n)
  end do
  do i = 1, n/2
     write(*,'(*(1X,ES24.16))') (b(i,j), j = 1,n/2)
  end do
  do i = 1, n/2
     write(*,'(*(1X,ES24.16))') (fg(i,j), j = 1,n/2+1)
  end do
  write(*,'(*(1X,ES24.16))') (alphar(i), i = 1,n/2)
  write(*,'(*(1X,ES24.16))') (alphai(i), i = 1,n/2)
  write(*,'(*(1X,ES24.16))') (beta(i), i = 1,n/2)
  do i = 1, n
     write(*,'(*(1X,ES24.16))') (q(i,j), j = 1,n)
  end do
  do i = 1, n/2
     write(*,'(*(1X,ES24.16))') (u1(i,j), j = 1,n/2)
  end do
  do i = 1, n/2
     write(*,'(*(1X,ES24.16))') (u2(i,j), j = 1,n/2)
  end do
end program
"""


def _parse_matrix(lines, offset, rows, cols):
    values = [[float(x) for x in line.split()] for line in lines[offset:offset + rows]]
    return np.asfortranarray(np.array(values)), offset + rows


def _full_u(u1, u2):
    return np.block([[u1, u2], [-u2, u1]])


def _skew_parts_from_fg(fg):
    m = fg.shape[0]
    f = np.zeros((m, m), dtype=float, order="F")
    g = np.zeros((m, m), dtype=float, order="F")
    for i in range(m):
        for j in range(m):
            if i > j:
                g[i, j] = fg[i, j]
                g[j, i] = -fg[i, j]
            if i < j:
                f[i, j] = fg[i, j + 1]
                f[j, i] = -fg[i, j + 1]
    return f, g


def _t_matrix(b, fg):
    f, g = _skew_parts_from_fg(fg)
    return np.block([[b, f], [g, b.T]])


def _t_schur_matrix(b, fg):
    f, _ = _skew_parts_from_fg(fg)
    zero = np.zeros_like(b)
    return np.block([[b, f], [zero, b.T]])


def _assert_mb04ed_contract(z0, b0, fg0, z, b, fg, q, u1, u2):
    n = z.shape[0]
    m = n // 2
    u = _full_u(u1, u2)

    assert_allclose(q.T @ q, np.eye(n), rtol=1e-13, atol=1e-13)
    assert_allclose(u.T @ u, np.eye(n), rtol=1e-13, atol=1e-13)
    assert_allclose(np.tril(z[:m, :m], -1), 0.0, rtol=1e-13, atol=1e-13)
    assert_allclose(np.triu(z[m:, m:], 1), 0.0, rtol=1e-13, atol=1e-13)
    assert_allclose(np.tril(b, -2), 0.0, rtol=1e-13, atol=1e-13)

    z_residual = u.T @ z0 @ q - z
    assert_allclose(z_residual[:m, :], 0.0, rtol=1e-12, atol=1e-12)
    assert_allclose(z_residual[m:, m:], 0.0, rtol=1e-12, atol=1e-12)

    jmat = np.block([
        [np.zeros((m, m)), np.eye(m)],
        [-np.eye(m), np.zeros((m, m))],
    ])
    t_residual = jmat @ q.T @ jmat.T @ _t_matrix(b0, fg0) @ q - _t_schur_matrix(b, fg)
    assert_allclose(t_residual, 0.0, rtol=1e-12, atol=1e-12)


def test_mb04ed_html_example_matches_fortran_reference(tmp_path):
    from ctrlsys import mb04ed

    job, compq, compu, z, b, fg = _read_mb04ed_example_data()
    n = z.shape[0]
    m = n // 2

    z_out, b_out, fg_out, q_out, u1_out, u2_out, alphar, alphai, beta, info = mb04ed(
        job, compq, compu, z.copy(order="F"), b.copy(order="F"), fg.copy(order="F")
    )
    output = run_fortran_driver(_mb04ed_example_fortran_source(), tmp_path)
    lines = output.splitlines()
    ref_info = int(lines[0])
    offset = 1
    ref_z, offset = _parse_matrix(lines, offset, n, n)
    ref_b, offset = _parse_matrix(lines, offset, m, m)
    ref_fg, offset = _parse_matrix(lines, offset, m, m + 1)
    ref_alphar = np.array([float(x) for x in lines[offset].split()])
    ref_alphai = np.array([float(x) for x in lines[offset + 1].split()])
    ref_beta = np.array([float(x) for x in lines[offset + 2].split()])
    offset += 3
    ref_q, offset = _parse_matrix(lines, offset, n, n)
    ref_u1, offset = _parse_matrix(lines, offset, m, m)
    ref_u2, offset = _parse_matrix(lines, offset, m, m)

    assert info == ref_info == 0
    assert_allclose(alphar, ref_alphar, rtol=1e-11, atol=1e-11)
    assert_allclose(alphai, ref_alphai, rtol=1e-11, atol=1e-11)
    assert_allclose(beta, ref_beta, rtol=1e-11, atol=1e-11)
    assert_allclose(np.linalg.norm(z_out[:m, :]), np.linalg.norm(ref_z[:m, :]), rtol=1e-13, atol=1e-13)
    assert_allclose(np.linalg.norm(z_out[m:, m:]), np.linalg.norm(ref_z[m:, m:]), rtol=1e-13, atol=1e-13)
    assert_allclose(np.linalg.norm(b_out), np.linalg.norm(ref_b), rtol=1e-13, atol=1e-13)
    assert_allclose(np.linalg.norm(np.triu(fg_out[:, 1:], 1)), np.linalg.norm(np.triu(ref_fg[:, 1:], 1)), rtol=1e-13, atol=1e-13)
    assert_allclose(np.linalg.norm(q_out), np.linalg.norm(ref_q), rtol=1e-13, atol=1e-13)
    assert_allclose(np.linalg.norm(u1_out), np.linalg.norm(ref_u1), rtol=1e-13, atol=1e-13)
    assert_allclose(np.linalg.norm(u2_out), np.linalg.norm(ref_u2), rtol=1e-13, atol=1e-13)
    _assert_mb04ed_contract(z, b, fg, z_out, b_out, fg_out, q_out, u1_out, u2_out)
    _assert_mb04ed_contract(z, b, fg, ref_z, ref_b, ref_fg, ref_q, ref_u1, ref_u2)


def test_mb04ed_html_example():
    """
    Validate basic functionality using SLICOT HTML doc example.

    Input: N=8 (even), JOB='T', COMPQ='I', COMPU='I'
    Matrices: Z (8x8), B (4x4), FG (4x5) containing skew-symmetric F and G

    Note: Due to complex internal transformations, we verify structural properties
    rather than exact eigenvalue matches.
    """
    from ctrlsys import mb04ed

    n = 8
    m = n // 2  # m = 4

    # Matrix Z (8x8, read row-by-row from HTML)
    z = np.array([
        [0.0949, 3.3613, -4.7663, -0.5534, 0.6408, -3.2793, 3.4253, 2.9654],
        [0.1138, -1.5903, 2.1837, -4.1648, -4.3775, -1.7454, 0.1744, 2.3262],
        [2.7505, 4.4048, 4.4183, 3.0478, 2.7728, 2.3048, -0.6451, -1.2045],
        [3.6091, -4.1716, 3.4461, 3.6880, -0.0985, 3.8458, 0.2528, -1.3859],
        [0.4352, -3.2829, 3.7246, 0.4794, -0.3690, -1.5562, -3.4817, -2.2902],
        [1.3080, -3.9881, -3.5497, 3.5020, 2.2582, 4.4764, -4.4080, -1.6818],
        [1.1308, -1.5087, 2.4730, 2.1553, -1.7129, -4.8669, -2.4102, 4.2274],
        [4.7933, -4.3671, -0.0473, -2.0092, 1.2439, -4.7385, 3.4242, -0.2764]
    ], order='F', dtype=float)

    # Matrix B (4x4, read row-by-row from HTML)
    b = np.array([
        [2.0936, 1.5510, 4.5974, 2.5127],
        [2.5469, -3.3739, -1.5961, -2.4490],
        [-2.2397, -3.8100, 0.8527, 0.0596],
        [1.7970, -0.0164, -2.7619, 1.9908]
    ], order='F', dtype=float)

    # Matrix FG (4x5): strictly lower triangular part = G,
    # columns 2:5 strictly upper triangular part = F
    # Diagonals not referenced (assumed zero)
    fg = np.array([
        [1.0000, 2.0000, -4.0500, 1.3353, 0.2899],
        [-0.4318, 2.0000, 2.0000, -2.9860, -0.0160],
        [1.0241, 0.9469, 2.0000, 2.0000, 1.3303],
        [0.0946, -0.1272, -4.4003, 2.0000, 2.0000]
    ], order='F', dtype=float)

    # Call mb04ed with JOB='T', COMPQ='I', COMPU='I'
    z_out, b_out, fg_out, q_out, u1_out, u2_out, alphar, alphai, beta, info = mb04ed(
        'T', 'I', 'I', z, b, fg
    )

    assert info == 0, f"MB04ED returned info={info}"

    # Verify Q is orthogonal
    qtq = q_out.T @ q_out
    assert_allclose(qtq, np.eye(n), rtol=1e-13, atol=1e-14)

    # Verify U is orthogonal symplectic
    u = np.zeros((n, n), order='F', dtype=float)
    u[:m, :m] = u1_out
    u[:m, m:] = u2_out
    u[m:, :m] = -u2_out
    u[m:, m:] = u1_out
    utu = u.T @ u
    assert_allclose(utu, np.eye(n), rtol=1e-13, atol=1e-14)

    # Check Z11 is upper triangular (elements below diagonal should be ~0)
    for i in range(1, m):
        for j in range(i):
            assert abs(z_out[i, j]) < 1e-10, f"Z11[{i},{j}]={z_out[i,j]} should be zero"

    # Verify eigenvalue output has correct dimensions
    assert len(alphar) == m
    assert len(alphai) == m
    assert len(beta) == m

    # Verify all betas are positive (eigenvalues are finite)
    assert np.all(beta > 0), "All beta values should be positive"


def test_mb04ed_q_orthogonality():
    """
    Validate mathematical property: Q should be orthogonal (Q'*Q = I).

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb04ed

    np.random.seed(42)
    n = 8
    m = n // 2

    # Random Z matrix
    z = np.random.randn(n, n).astype(float, order='F')

    # Random B matrix
    b = np.random.randn(m, m).astype(float, order='F')

    # FG with skew-symmetric structure
    fg = np.zeros((m, m + 1), order='F', dtype=float)
    for i in range(1, m):
        for j in range(i):
            fg[i, j] = np.random.randn()  # G (strictly lower)
    for i in range(m - 1):
        for j in range(i + 2, m + 1):
            fg[i, j] = np.random.randn()  # F (strictly upper starting col 2)

    z_out, b_out, fg_out, q_out, u1_out, u2_out, alphar, alphai, beta, info = mb04ed(
        'T', 'I', 'I', z.copy(), b.copy(), fg.copy()
    )

    assert info == 0, f"MB04ED returned info={info}"

    # Q should be orthogonal: Q' * Q = I
    qtq = q_out.T @ q_out
    assert_allclose(qtq, np.eye(n), rtol=1e-13, atol=1e-14)

    # Q * Q' = I
    qqt = q_out @ q_out.T
    assert_allclose(qqt, np.eye(n), rtol=1e-13, atol=1e-14)


def test_mb04ed_u_orthogonal_symplectic():
    """
    Validate mathematical property: U should be orthogonal symplectic.

    U = [[U1, U2], [-U2, U1]] where U'*U = I

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb04ed

    np.random.seed(123)
    n = 8
    m = n // 2

    # Random Z matrix
    z = np.random.randn(n, n).astype(float, order='F')

    # Random B matrix
    b = np.random.randn(m, m).astype(float, order='F')

    # FG with skew-symmetric structure
    fg = np.zeros((m, m + 1), order='F', dtype=float)
    for i in range(1, m):
        for j in range(i):
            fg[i, j] = np.random.randn()
    for i in range(m - 1):
        for j in range(i + 2, m + 1):
            fg[i, j] = np.random.randn()

    z_out, b_out, fg_out, q_out, u1_out, u2_out, alphar, alphai, beta, info = mb04ed(
        'T', 'I', 'I', z.copy(), b.copy(), fg.copy()
    )

    assert info == 0, f"MB04ED returned info={info}"

    # Build full U matrix
    u = np.zeros((n, n), order='F', dtype=float)
    u[:m, :m] = u1_out
    u[:m, m:] = u2_out
    u[m:, :m] = -u2_out
    u[m:, m:] = u1_out

    # U should be orthogonal: U' * U = I
    utu = u.T @ u
    assert_allclose(utu, np.eye(n), rtol=1e-13, atol=1e-14)


def test_mb04ed_eigenvalues_only():
    """
    Test JOB='E' mode: compute eigenvalues only.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb04ed

    np.random.seed(456)
    n = 6
    m = n // 2

    # Random matrices
    z = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(m, m).astype(float, order='F')

    fg = np.zeros((m, m + 1), order='F', dtype=float)
    for i in range(1, m):
        for j in range(i):
            fg[i, j] = np.random.randn()
    for i in range(m - 1):
        for j in range(i + 2, m + 1):
            fg[i, j] = np.random.randn()

    z_out, b_out, fg_out, q_out, u1_out, u2_out, alphar, alphai, beta, info = mb04ed(
        'E', 'N', 'N', z.copy(), b.copy(), fg.copy()
    )

    assert info == 0, f"MB04ED returned info={info}"
    assert len(alphar) == m
    assert len(alphai) == m
    assert len(beta) == m


def test_mb04ed_n_zero():
    """
    Edge case: N=0 should return immediately with info=0.
    """
    from ctrlsys import mb04ed

    n = 0
    z = np.array([], order='F', dtype=float).reshape(0, 0)
    b = np.array([], order='F', dtype=float).reshape(0, 0)
    fg = np.array([], order='F', dtype=float).reshape(0, 1)

    z_out, b_out, fg_out, q_out, u1_out, u2_out, alphar, alphai, beta, info = mb04ed(
        'T', 'I', 'I', z, b, fg
    )

    assert info == 0, f"MB04ED returned info={info}"
    assert len(alphar) == 0
    assert len(alphai) == 0
    assert len(beta) == 0


def test_mb04ed_invalid_n_odd():
    """
    Test error handling: N must be even. N=5 should return info=-4.
    """
    from ctrlsys import mb04ed

    n = 5
    m = n // 2
    z = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(m, m).astype(float, order='F')
    fg = np.zeros((m, m + 1), order='F', dtype=float)

    z_out, b_out, fg_out, q_out, u1_out, u2_out, alphar, alphai, beta, info = mb04ed(
        'T', 'I', 'I', z, b, fg
    )

    assert info == -4, f"Expected info=-4 for odd N, got info={info}"


def test_mb04ed_compu_update():
    """
    Test COMPU='U' mode: update existing U0 matrix.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb04ed

    np.random.seed(789)
    n = 6
    m = n // 2

    # Random matrices
    z = np.random.randn(n, n).astype(float, order='F')
    b = np.random.randn(m, m).astype(float, order='F')

    fg = np.zeros((m, m + 1), order='F', dtype=float)
    for i in range(1, m):
        for j in range(i):
            fg[i, j] = np.random.randn()
    for i in range(m - 1):
        for j in range(i + 2, m + 1):
            fg[i, j] = np.random.randn()

    # Create initial orthogonal symplectic U0
    # Start with identity
    u1_in = np.eye(m, order='F', dtype=float)
    u2_in = np.zeros((m, m), order='F', dtype=float)

    z_out, b_out, fg_out, q_out, u1_out, u2_out, alphar, alphai, beta, info = mb04ed(
        'T', 'I', 'U', z.copy(), b.copy(), fg.copy(), u1_in.copy(), u2_in.copy()
    )

    assert info == 0, f"MB04ED returned info={info}"

    # Build full U matrix and check orthogonality
    u = np.zeros((n, n), order='F', dtype=float)
    u[:m, :m] = u1_out
    u[:m, m:] = u2_out
    u[m:, :m] = -u2_out
    u[m:, m:] = u1_out

    utu = u.T @ u
    assert_allclose(utu, np.eye(n), rtol=1e-13, atol=1e-14)
