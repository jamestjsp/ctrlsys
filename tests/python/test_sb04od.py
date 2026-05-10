"""
Tests for SB04OD - Generalized Sylvester equations with separation estimation.

Solves:
  A * R - L * B = scale * C
  D * R - L * E = scale * F
or:
  A' * R + D' * L = scale * C
  R * B' + L * E' = scale * (-F)

where (A,D) are M-by-M, (B,E) are N-by-N, and C, F, R, L are M-by-N.
"""

import numpy as np
import pytest

from ctrlsys import sb04od
from fortran_reference import run_fortran_driver


def _sb04od_data_file():
    return "SLICOT-Reference/examples/data/SB04OD.dat"


def _sb04od_tokens():
    tokens = []
    with open(_sb04od_data_file(), encoding="ascii") as data_file:
        next(data_file)
        for line in data_file:
            tokens.extend(line.split())
    return tokens


def _read_sb04od_example_data():
    tokens = _sb04od_tokens()
    offset = 0
    m = int(tokens[offset])
    n = int(tokens[offset + 1])
    reduce, trans, jobd = tokens[offset + 2:offset + 5]
    offset += 5

    def matrix(rows, cols):
        nonlocal offset
        data = np.array([float(x) for x in tokens[offset:offset + rows * cols]])
        offset += rows * cols
        return np.asfortranarray(data.reshape(rows, cols))

    a = matrix(m, m)
    b = matrix(n, n)
    c = matrix(m, n)
    d = matrix(m, m)
    e = matrix(n, n)
    f = matrix(m, n)
    return reduce, trans, jobd, a, b, c, d, e, f


def _sb04od_example_fortran_source():
    data_path = _sb04od_data_file()
    return f"""
program sb04od_reference
  implicit none
  integer, parameter :: mmax = 10, nmax = 10
  integer, parameter :: lda = mmax, ldb = nmax, ldc = mmax, ldd = mmax
  integer, parameter :: lde = nmax, ldf = mmax, ldp = mmax, ldq = mmax
  integer, parameter :: ldu = nmax, ldv = nmax
  integer, parameter :: ldwork = max(11*max(mmax,nmax), 10*max(mmax,nmax)+23, 2*mmax*nmax)
  integer, parameter :: liwork = mmax+nmax+6
  character :: jobd, reduce, trans
  integer :: i, info, j, m, n
  integer :: iwork(liwork)
  double precision :: dif, scale
  double precision :: a(lda,mmax), b(ldb,nmax), c(ldc,nmax), d(ldd,mmax)
  double precision :: dwork(ldwork), e(lde,nmax), f(ldf,nmax)
  double precision :: p(ldp,mmax), q(ldq,mmax), u(ldu,nmax), v(ldv,nmax)

  open(unit=5, file='{data_path}', status='old')
  read(5, *)
  read(5, *) m, n, reduce, trans, jobd
  read(5, *) ((a(i,j), j = 1,m), i = 1,m)
  read(5, *) ((b(i,j), j = 1,n), i = 1,n)
  read(5, *) ((c(i,j), j = 1,n), i = 1,m)
  read(5, *) ((d(i,j), j = 1,m), i = 1,m)
  read(5, *) ((e(i,j), j = 1,n), i = 1,n)
  read(5, *) ((f(i,j), j = 1,n), i = 1,m)
  close(5)

  call SB04OD(reduce, trans, jobd, m, n, a, lda, b, ldb, c, ldc, d, ldd, &
       e, lde, f, ldf, scale, dif, p, ldp, q, ldq, u, ldu, v, ldv, &
       iwork, dwork, ldwork, info)

  write(*,'(I0,2(1X,ES24.16))') info, scale, dif
  do i = 1, m
     write(*,'(*(1X,ES24.16))') (a(i,j), j = 1,m)
  end do
  do i = 1, n
     write(*,'(*(1X,ES24.16))') (b(i,j), j = 1,n)
  end do
  do i = 1, m
     write(*,'(*(1X,ES24.16))') (c(i,j), j = 1,n)
  end do
  do i = 1, m
     write(*,'(*(1X,ES24.16))') (d(i,j), j = 1,m)
  end do
  do i = 1, n
     write(*,'(*(1X,ES24.16))') (e(i,j), j = 1,n)
  end do
  do i = 1, m
     write(*,'(*(1X,ES24.16))') (f(i,j), j = 1,n)
  end do
  do i = 1, m
     write(*,'(*(1X,ES24.16))') (p(i,j), j = 1,m)
  end do
  do i = 1, m
     write(*,'(*(1X,ES24.16))') (q(i,j), j = 1,m)
  end do
  do i = 1, n
     write(*,'(*(1X,ES24.16))') (u(i,j), j = 1,n)
  end do
  do i = 1, n
     write(*,'(*(1X,ES24.16))') (v(i,j), j = 1,n)
  end do
end program
"""


def _parse_matrix(lines, offset, rows, cols):
    values = [[float(x) for x in line.split()] for line in lines[offset:offset + rows]]
    return np.asfortranarray(np.array(values)), offset + rows


def test_sb04od_html_example_matches_fortran_reference(tmp_path):
    reduce, trans, jobd, a, b, c, d, e, f = _read_sb04od_example_data()
    m, n = c.shape

    (a_out, b_out, c_out, d_out, e_out, f_out,
     scale, dif, p, q, u, v, info) = sb04od(
        reduce, trans, jobd,
        a.copy(order="F"), b.copy(order="F"), c.copy(order="F"),
        d.copy(order="F"), e.copy(order="F"), f.copy(order="F")
    )
    output = run_fortran_driver(_sb04od_example_fortran_source(), tmp_path)
    lines = output.splitlines()
    header = lines[0].split()
    ref_info = int(header[0])
    ref_scale = float(header[1])
    ref_dif = float(header[2])
    offset = 1
    ref_a, offset = _parse_matrix(lines, offset, m, m)
    ref_b, offset = _parse_matrix(lines, offset, n, n)
    ref_c, offset = _parse_matrix(lines, offset, m, n)
    ref_d, offset = _parse_matrix(lines, offset, m, m)
    ref_e, offset = _parse_matrix(lines, offset, n, n)
    ref_f, offset = _parse_matrix(lines, offset, m, n)
    ref_p, offset = _parse_matrix(lines, offset, m, m)
    ref_q, offset = _parse_matrix(lines, offset, m, m)
    ref_u, offset = _parse_matrix(lines, offset, n, n)
    ref_v, offset = _parse_matrix(lines, offset, n, n)

    assert info == ref_info == 0
    np.testing.assert_allclose(scale, ref_scale, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(dif, ref_dif, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(a_out, ref_a, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(b_out, ref_b, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(c_out, ref_c, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(d_out, ref_d, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(e_out, ref_e, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(f_out, ref_f, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(p, ref_p, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(q, ref_q, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(u, ref_u, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(v, ref_v, rtol=1e-11, atol=1e-11)


class TestSB04ODBasic:
    """Test basic functionality using SLICOT HTML doc example."""

    def test_html_doc_example(self):
        """
        Validate against SLICOT HTML documentation example.

        M=3, N=2, REDUCE='R', TRANS='N', JOBD='D'
        Solves equation (1) with full reduction and Dif estimate.
        """
        m, n = 3, 2

        # Input A (3x3) - row-wise in doc
        a = np.array([
            [1.6, -3.1, 1.9],
            [-3.8, 4.2, 2.4],
            [0.5, 2.2, -4.5]
        ], order='F', dtype=float)

        # Input B (2x2) - row-wise in doc
        b = np.array([
            [1.1, 0.1],
            [-1.3, -3.1]
        ], order='F', dtype=float)

        # Input C (3x2) - row-wise in doc
        c = np.array([
            [-2.0, 28.9],
            [-5.7, -11.8],
            [12.9, -31.7]
        ], order='F', dtype=float)

        # Input D (3x3) - row-wise in doc
        d = np.array([
            [2.5, 0.1, 1.7],
            [-2.5, 0.0, 0.9],
            [0.1, 5.1, -7.3]
        ], order='F', dtype=float)

        # Input E (2x2) - row-wise in doc
        e = np.array([
            [6.0, 2.4],
            [-3.6, 2.5]
        ], order='F', dtype=float)

        # Input F (3x2) - row-wise in doc
        f = np.array([
            [0.5, 23.8],
            [-11.0, -10.4],
            [39.5, -74.8]
        ], order='F', dtype=float)

        # Expected outputs from HTML doc
        # L (solution matrix in F output)
        l_expected = np.array([
            [-0.7538, -1.6210],
            [2.1778, 1.7005],
            [-3.5029, 2.7961]
        ], order='F', dtype=float)

        # R (solution matrix in C output)
        r_expected = np.array([
            [1.3064, 2.7989],
            [0.3698, -5.3376],
            [-0.8767, 6.7500]
        ], order='F', dtype=float)

        # DIF = 0.1147
        dif_expected = 0.1147

        # Call the routine
        (a_out, b_out, c_out, d_out, e_out, f_out,
         scale, dif, p, q, u, v, info) = sb04od(
            'R', 'N', 'D', a, b, c, d, e, f)

        assert info == 0
        np.testing.assert_allclose(scale, 1.0, rtol=1e-14)

        # Check solution matrices (rtol=1e-3 matches 4-decimal HTML precision)
        np.testing.assert_allclose(c_out, r_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(f_out, l_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(dif, dif_expected, rtol=1e-2, atol=1e-3)


class TestSB04ODTransposed:
    """Test transposed equation (2)."""

    def test_transposed_equation(self):
        """
        Test solving transposed equation (2).

        A' * R + D' * L = scale * C
        R * B' + L * E' = scale * (-F)

        Random seed: 42
        """
        np.random.seed(42)
        m, n = 3, 2

        # Generate random matrices
        a = np.random.randn(m, m).astype(float, order='F')
        b = np.random.randn(n, n).astype(float, order='F')
        c = np.random.randn(m, n).astype(float, order='F')
        d = np.random.randn(m, m).astype(float, order='F')
        e = np.random.randn(n, n).astype(float, order='F')
        f = np.random.randn(m, n).astype(float, order='F')

        # Save originals for residual check
        a_orig = a.copy()
        b_orig = b.copy()
        c_orig = c.copy()
        d_orig = d.copy()
        e_orig = e.copy()
        f_orig = f.copy()

        # Call with TRANS='T'
        (a_out, b_out, c_out, d_out, e_out, f_out,
         scale, dif, p, q, u, v, info) = sb04od(
            'R', 'T', 'N', a, b, c, d, e, f)

        assert info == 0
        assert 0.0 < scale <= 1.0

        # R is in c_out, L is in f_out
        r = c_out
        lmat = f_out

        # For REDUCE='R', solution is transformed back
        # Verify equation (2) residual: A' * R + D' * L = scale * C_orig
        lhs1 = a_orig.T @ r + d_orig.T @ lmat
        rhs1 = scale * c_orig
        np.testing.assert_allclose(lhs1, rhs1, rtol=1e-10, atol=1e-10)

        # R * B' + L * E' = scale * (-F_orig)
        lhs2 = r @ b_orig.T + lmat @ e_orig.T
        rhs2 = -scale * f_orig
        np.testing.assert_allclose(lhs2, rhs2, rtol=1e-10, atol=1e-10)


class TestSB04ODReduceOptions:
    """Test different REDUCE options."""

    def test_reduce_n_already_schur(self):
        """
        Test REDUCE='N' when matrices already in generalized Schur form.

        Random seed: 123
        """
        np.random.seed(123)
        m, n = 3, 2

        # Create upper quasi-triangular A (Schur form)
        # Eigenvalues of (A,D): 2/1=2, 1/2=0.5, -1/1.5=-0.667
        a = np.array([
            [2.0, 1.5, -0.5],
            [0.0, 1.0, 0.3],
            [0.0, 0.0, -1.0]
        ], order='F', dtype=float)

        # Upper triangular D
        d = np.array([
            [1.0, 0.2, 0.1],
            [0.0, 2.0, -0.3],
            [0.0, 0.0, 1.5]
        ], order='F', dtype=float)

        # Upper quasi-triangular B (Schur form)
        # Eigenvalues of (B,E): 5/1=5, 4/1=4 (distinct from A,D)
        b = np.array([
            [5.0, -0.5],
            [0.0, 4.0]
        ], order='F', dtype=float)

        # Upper triangular E
        e = np.array([
            [1.0, 0.2],
            [0.0, 1.0]
        ], order='F', dtype=float)

        # Random RHS
        c = np.random.randn(m, n).astype(float, order='F')
        f = np.random.randn(m, n).astype(float, order='F')

        c_orig = c.copy()
        f_orig = f.copy()

        # Call with REDUCE='N' (no reduction needed)
        (a_out, b_out, c_out, d_out, e_out, f_out,
         scale, dif, p, q, u, v, info) = sb04od(
            'N', 'N', 'N', a, b, c, d, e, f)

        assert info == 0

        # Verify equation (1): A * R - L * B = scale * C
        r = c_out
        lmat = f_out
        lhs1 = a @ r - lmat @ b
        rhs1 = scale * c_orig
        np.testing.assert_allclose(lhs1, rhs1, rtol=1e-10, atol=1e-10)

        # D * R - L * E = scale * F
        lhs2 = d @ r - lmat @ e
        rhs2 = scale * f_orig
        np.testing.assert_allclose(lhs2, rhs2, rtol=1e-10, atol=1e-10)


class TestSB04ODDifEstimates:
    """Test Dif estimation options."""

    def test_dif_one_norm(self):
        """
        Test JOBD='1' (one-norm Dif estimate only).

        Random seed: 456
        """
        np.random.seed(456)
        m, n = 2, 2

        a = np.random.randn(m, m).astype(float, order='F')
        b = np.random.randn(n, n).astype(float, order='F')
        c = np.random.randn(m, n).astype(float, order='F')
        d = np.random.randn(m, m).astype(float, order='F')
        e = np.random.randn(n, n).astype(float, order='F')
        f = np.random.randn(m, n).astype(float, order='F')

        # JOBD='1' computes only Dif, not the solution
        (a_out, b_out, c_out, d_out, e_out, f_out,
         scale, dif, p, q, u, v, info) = sb04od(
            'R', 'N', '1', a, b, c, d, e, f)

        assert info == 0
        assert dif > 0.0  # Dif should be positive

    def test_dif_frobenius_norm(self):
        """
        Test JOBD='2' (Frobenius norm Dif estimate only).

        Random seed: 789
        """
        np.random.seed(789)
        m, n = 2, 2

        a = np.random.randn(m, m).astype(float, order='F')
        b = np.random.randn(n, n).astype(float, order='F')
        c = np.random.randn(m, n).astype(float, order='F')
        d = np.random.randn(m, m).astype(float, order='F')
        e = np.random.randn(n, n).astype(float, order='F')
        f = np.random.randn(m, n).astype(float, order='F')

        (a_out, b_out, c_out, d_out, e_out, f_out,
         scale, dif, p, q, u, v, info) = sb04od(
            'R', 'N', '2', a, b, c, d, e, f)

        assert info == 0
        assert dif > 0.0


class TestSB04ODEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_dimensions(self):
        """Test M=0 or N=0 quick return."""
        # M=0
        a = np.zeros((0, 0), order='F', dtype=float)
        b = np.array([[1.0]], order='F', dtype=float)
        c = np.zeros((0, 1), order='F', dtype=float)
        d = np.zeros((0, 0), order='F', dtype=float)
        e = np.array([[1.0]], order='F', dtype=float)
        f = np.zeros((0, 1), order='F', dtype=float)

        (a_out, b_out, c_out, d_out, e_out, f_out,
         scale, dif, p, q, u, v, info) = sb04od(
            'R', 'N', 'D', a, b, c, d, e, f)

        assert info == 0
        assert scale == 1.0
        assert dif == 1.0

    def test_n_zero(self):
        """Test N=0 quick return."""
        a = np.array([[1.0]], order='F', dtype=float)
        b = np.zeros((0, 0), order='F', dtype=float)
        c = np.zeros((1, 0), order='F', dtype=float)
        d = np.array([[1.0]], order='F', dtype=float)
        e = np.zeros((0, 0), order='F', dtype=float)
        f = np.zeros((1, 0), order='F', dtype=float)

        (a_out, b_out, c_out, d_out, e_out, f_out,
         scale, dif, p, q, u, v, info) = sb04od(
            'R', 'N', 'N', a, b, c, d, e, f)

        assert info == 0
        assert scale == 1.0


class TestSB04ODResidualProperty:
    """Test mathematical property: residual equations hold."""

    def test_equation1_residual(self):
        """
        Validate equation (1) residual for randomly generated system.

        A * R - L * B = scale * C
        D * R - L * E = scale * F

        Random seed: 888
        """
        np.random.seed(888)
        m, n = 4, 3

        a = np.random.randn(m, m).astype(float, order='F')
        b = np.random.randn(n, n).astype(float, order='F')
        c = np.random.randn(m, n).astype(float, order='F')
        d = np.random.randn(m, m).astype(float, order='F')
        e = np.random.randn(n, n).astype(float, order='F')
        f = np.random.randn(m, n).astype(float, order='F')

        c_orig = c.copy()
        f_orig = f.copy()

        (a_out, b_out, c_out, d_out, e_out, f_out,
         scale, dif, p, q, u, v, info) = sb04od(
            'R', 'N', 'N', a, b, c, d, e, f)

        assert info == 0

        # Extract solution: R is in c_out, L is in f_out
        r = c_out
        lmat = f_out

        # The solution satisfies equations in original coordinates
        # For REDUCE='R', the solution is back-transformed:
        # R = Q * R1 * V', L = P * L1 * U'
        # The equations hold for original A,B,C,D,E,F (before reduction)

        # Use original matrices (before Schur decomposition)
        a_orig = a.copy()
        b_orig = b.copy()
        d_orig = d.copy()
        e_orig = e.copy()

        # Unfortunately, a,b,c,d,e,f are modified in-place by the call
        # So we need to re-read them from the saved copies
        # Actually, we saved c_orig and f_orig but not a_orig etc. before the call
        # Let me fix by checking the actual semantics

        # Actually, we need the original input matrices before modification
        # Let me re-run with fresh copies
        np.random.seed(888)
        a = np.random.randn(m, m).astype(float, order='F')
        b = np.random.randn(n, n).astype(float, order='F')
        c = np.random.randn(m, n).astype(float, order='F')
        d = np.random.randn(m, m).astype(float, order='F')
        e = np.random.randn(n, n).astype(float, order='F')
        f = np.random.randn(m, n).astype(float, order='F')

        a_orig = a.copy()
        b_orig = b.copy()
        c_orig = c.copy()
        d_orig = d.copy()
        e_orig = e.copy()
        f_orig = f.copy()

        (a_out, b_out, c_out, d_out, e_out, f_out,
         scale, dif, p, q, u, v, info) = sb04od(
            'R', 'N', 'N', a, b, c, d, e, f)

        assert info == 0

        r = c_out
        lmat = f_out

        # Verify: A_orig * R - L * B_orig = scale * C_orig
        lhs1 = a_orig @ r - lmat @ b_orig
        rhs1 = scale * c_orig
        np.testing.assert_allclose(lhs1, rhs1, rtol=1e-10, atol=1e-10)

        # Verify: D_orig * R - L * E_orig = scale * F_orig
        lhs2 = d_orig @ r - lmat @ e_orig
        rhs2 = scale * f_orig
        np.testing.assert_allclose(lhs2, rhs2, rtol=1e-10, atol=1e-10)


class TestSB04ODErrors:
    """Test error handling."""

    def test_invalid_reduce(self):
        """Test invalid REDUCE parameter."""
        a = np.eye(2, order='F', dtype=float)
        b = np.eye(2, order='F', dtype=float)
        c = np.ones((2, 2), order='F', dtype=float)
        d = np.eye(2, order='F', dtype=float)
        e = np.eye(2, order='F', dtype=float)
        f = np.ones((2, 2), order='F', dtype=float)

        with pytest.raises(RuntimeError):
            sb04od('X', 'N', 'N', a, b, c, d, e, f)

    def test_invalid_trans(self):
        """Test invalid TRANS parameter."""
        a = np.eye(2, order='F', dtype=float)
        b = np.eye(2, order='F', dtype=float)
        c = np.ones((2, 2), order='F', dtype=float)
        d = np.eye(2, order='F', dtype=float)
        e = np.eye(2, order='F', dtype=float)
        f = np.ones((2, 2), order='F', dtype=float)

        with pytest.raises(RuntimeError):
            sb04od('R', 'X', 'N', a, b, c, d, e, f)
