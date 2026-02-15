"""
Tests for MB03XS: Eigenvalues and real skew-Hamiltonian Schur form.

MB03XS computes eigenvalues and real skew-Hamiltonian Schur form of
a skew-Hamiltonian matrix W = [[A, G], [Q, A^T]] where G, Q are
skew-symmetric, transforming it to [[Aout, Gout], [0, Aout^T]]
where Aout is in Schur canonical form.

Test strategy: Generate valid skew-Hamiltonian matrices, verify Schur
decomposition properties (eigenvalue preservation, orthogonality).
"""
import numpy as np
import pytest
from ctrlsys import mb03xs


def build_skew_hamiltonian(a, q_lower, g_upper):
    """
    Build full 2N x 2N skew-Hamiltonian matrix from components.

    W = [[A, G], [Q, A^T]]
    where Q and G are skew-symmetric (Q = -Q^T, G = -G^T).

    Parameters:
    - a: N x N matrix
    - q_lower: strictly lower triangular part of Q
    - g_upper: strictly upper triangular part of G

    Returns full 2N x 2N matrix W.
    """
    n = a.shape[0]
    w = np.zeros((2 * n, 2 * n), dtype=float, order='F')
    q = np.zeros((n, n), dtype=float, order='F')
    g = np.zeros((n, n), dtype=float, order='F')

    for i in range(n):
        for j in range(i):
            q[i, j] = q_lower[i, j]
            q[j, i] = -q_lower[i, j]
    for i in range(n):
        for j in range(i + 1, n):
            g[i, j] = g_upper[i, j]
            g[j, i] = -g_upper[i, j]

    w[:n, :n] = a
    w[:n, n:] = g
    w[n:, :n] = q
    w[n:, n:] = a.T
    return w


def pack_qg(n, q_lower, g_upper):
    """
    Pack Q and G into QG array format.

    QG has dimension N x (N+1):
    - Columns 0:N-1 (0-indexed) contain strictly lower triangular Q (Fortran cols 1:N)
    - Columns 1:N (0-indexed) contain strictly upper triangular G (Fortran cols 2:N+1)

    Note: Column 0 only has Q's strictly lower part (rows 1:N-1)
    Column N only has G's strictly upper part (rows 0:N-2)
    Columns 1:N-1 have both Q lower and G upper parts
    """
    qg = np.zeros((n, n + 1), dtype=float, order='F')
    for i in range(1, n):
        for j in range(i):
            qg[i, j] = q_lower[i, j]
    for i in range(n - 1):
        for j in range(i + 1, n):
            qg[i, j + 1] = g_upper[i, j]
    return qg


def unpack_g_from_qg(qg):
    """Extract G from QG array (strictly upper triangular in cols 1:N+1)."""
    n = qg.shape[0]
    g = np.zeros((n, n), dtype=float, order='F')
    for i in range(n - 1):
        for j in range(i + 1, n):
            g[i, j] = qg[i, j + 1]
            g[j, i] = -qg[i, j + 1]
    return g


class TestMB03XSBasic:
    """Basic functionality tests for MB03XS."""

    def test_small_2x2_system(self):
        """
        Test 2x2 system with known eigenvalues.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n = 2

        a = np.array([[1.0, 2.0], [0.5, 1.5]], dtype=float, order='F')
        q_lower = np.zeros((n, n), dtype=float, order='F')
        q_lower[1, 0] = 0.3
        g_upper = np.zeros((n, n), dtype=float, order='F')
        g_upper[0, 1] = 0.4

        qg = pack_qg(n, q_lower, g_upper)
        a_work = a.copy(order='F')
        qg_work = qg.copy(order='F')

        w_orig = build_skew_hamiltonian(a, q_lower, g_upper)
        eig_orig = np.linalg.eigvals(w_orig)

        a_out, qg_out, wr, wi, info = mb03xs(a_work, qg_work, jobu='N')

        assert info == 0, f"MB03XS failed with info={info}"
        assert wr.shape == (n,)
        assert wi.shape == (n,)

        eig_computed = wr + 1j * wi
        eig_full = np.concatenate([eig_computed, eig_computed])

        eig_orig_sorted = np.sort(np.abs(eig_orig))
        eig_full_sorted = np.sort(np.abs(eig_full))
        np.testing.assert_allclose(eig_orig_sorted, eig_full_sorted, rtol=1e-10)

    def test_4x4_with_u_matrices(self):
        """
        Test 4x4 system with U1, U2 computation.

        Validates orthogonality: U^T U = I where U = [[U1, U2], [-U2, U1]]

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n = 4

        a = np.random.randn(n, n).astype(float, order='F')
        q_lower = np.zeros((n, n), dtype=float, order='F')
        g_upper = np.zeros((n, n), dtype=float, order='F')
        for i in range(1, n):
            for j in range(i):
                q_lower[i, j] = np.random.randn()
        for i in range(n - 1):
            for j in range(i + 1, n):
                g_upper[i, j] = np.random.randn()

        qg = pack_qg(n, q_lower, g_upper)
        a_work = a.copy(order='F')
        qg_work = qg.copy(order='F')

        a_out, qg_out, u1, u2, wr, wi, info = mb03xs(a_work, qg_work, jobu='U')

        assert info == 0, f"MB03XS failed with info={info}"
        assert u1.shape == (n, n)
        assert u2.shape == (n, n)

        full_u = np.zeros((2 * n, 2 * n), dtype=float, order='F')
        full_u[:n, :n] = u1
        full_u[:n, n:] = u2
        full_u[n:, :n] = -u2
        full_u[n:, n:] = u1

        np.testing.assert_allclose(full_u.T @ full_u, np.eye(2 * n), rtol=1e-12, atol=1e-14)


class TestMB03XSEigenvaluePreservation:
    """Tests for eigenvalue preservation under skew-Hamiltonian Schur decomposition."""

    def test_eigenvalue_preservation_5x5(self):
        """
        Validate eigenvalues preserved under transformation.

        For skew-Hamiltonian W, eigenvalues come in pairs: if lambda is an
        eigenvalue, so is lambda (with algebraic multiplicity 2).

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n = 5

        a = np.random.randn(n, n).astype(float, order='F')
        q_lower = np.zeros((n, n), dtype=float, order='F')
        g_upper = np.zeros((n, n), dtype=float, order='F')
        for i in range(1, n):
            for j in range(i):
                q_lower[i, j] = np.random.randn() * 0.5
        for i in range(n - 1):
            for j in range(i + 1, n):
                g_upper[i, j] = np.random.randn() * 0.5

        w_orig = build_skew_hamiltonian(a, q_lower, g_upper)
        eig_orig = np.linalg.eigvals(w_orig)

        qg = pack_qg(n, q_lower, g_upper)
        a_work = a.copy(order='F')
        qg_work = qg.copy(order='F')

        a_out, qg_out, u1, u2, wr, wi, info = mb03xs(a_work, qg_work, jobu='U')

        assert info == 0
        eig_half = wr + 1j * wi
        eig_full = np.concatenate([eig_half, eig_half])

        eig_orig_sorted = np.sort(np.abs(eig_orig))
        eig_full_sorted = np.sort(np.abs(eig_full))

        np.testing.assert_allclose(
            eig_orig_sorted,
            eig_full_sorted,
            rtol=1e-10, atol=1e-12
        )

    def test_schur_form_structure(self):
        """
        Validate output is in Schur canonical form.

        Aout should be block upper triangular with 1x1 and 2x2 blocks.
        Each 2x2 block has equal diagonals and opposite off-diagonals.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n = 6

        a = np.random.randn(n, n).astype(float, order='F')
        q_lower = np.zeros((n, n), dtype=float, order='F')
        g_upper = np.zeros((n, n), dtype=float, order='F')
        for i in range(1, n):
            for j in range(i):
                q_lower[i, j] = np.random.randn() * 0.3
        for i in range(n - 1):
            for j in range(i + 1, n):
                g_upper[i, j] = np.random.randn() * 0.3

        qg = pack_qg(n, q_lower, g_upper)
        a_work = a.copy(order='F')
        qg_work = qg.copy(order='F')

        a_out, qg_out, wr, wi, info = mb03xs(a_work, qg_work, jobu='N')

        assert info == 0

        for i in range(n - 2):
            if abs(a_out[i + 1, i]) > 1e-14:
                assert abs(a_out[i + 2, i + 1]) < 1e-14, \
                    f"Invalid Schur structure: subdiagonal at ({i+2},{i+1})"


class TestMB03XSTransformation:
    """Tests for transformation matrix properties."""

    def test_transformation_reconstructs_schur_form(self):
        """
        Validate U^T W U = [[Aout, Gout], [0, Aout^T]] for Q=0 case.

        When Q=0, we can directly verify transformation properties by
        constructing W from A and G, transforming, and checking result.

        Random seed: 888 (for reproducibility)
        """
        np.random.seed(888)
        n = 4

        a = np.random.randn(n, n).astype(float, order='F')
        q_lower = np.zeros((n, n), dtype=float, order='F')
        g_upper = np.zeros((n, n), dtype=float, order='F')
        for i in range(n - 1):
            for j in range(i + 1, n):
                g_upper[i, j] = np.random.randn() * 0.4

        w_orig = build_skew_hamiltonian(a, q_lower, g_upper)
        qg = pack_qg(n, q_lower, g_upper)
        a_work = a.copy(order='F')
        qg_work = qg.copy(order='F')

        a_out, qg_out, u1, u2, wr, wi, info = mb03xs(a_work, qg_work, jobu='U')

        assert info == 0

        full_u = np.zeros((2 * n, 2 * n), dtype=float, order='F')
        full_u[:n, :n] = u1
        full_u[:n, n:] = u2
        full_u[n:, :n] = -u2
        full_u[n:, n:] = u1

        w_transformed = full_u.T @ w_orig @ full_u

        np.testing.assert_allclose(w_transformed[n:, :n], np.zeros((n, n)), atol=1e-11,
            err_msg="Lower-left block of U^T W U should be zero")

        eig_a_out = np.linalg.eigvals(a_out)
        eig_w_trans = np.linalg.eigvals(w_transformed[:n, :n])
        np.testing.assert_allclose(
            np.sort(np.abs(eig_a_out)),
            np.sort(np.abs(eig_w_trans)),
            rtol=1e-10,
            err_msg="Eigenvalues of Aout should match transformed matrix"
        )

    def test_transformation_with_nonzero_q(self):
        """
        Validate eigenvalue preservation when Q is non-zero.

        Even with Q≠0, the eigenvalues of the transformed Aout should match
        the correct eigenvalues of the original skew-Hamiltonian matrix W.

        Random seed: 999 (for reproducibility)
        """
        np.random.seed(999)
        n = 4

        a = np.random.randn(n, n).astype(float, order='F')
        q_lower = np.zeros((n, n), dtype=float, order='F')
        g_upper = np.zeros((n, n), dtype=float, order='F')
        for i in range(1, n):
            for j in range(i):
                q_lower[i, j] = np.random.randn() * 0.4
        for i in range(n - 1):
            for j in range(i + 1, n):
                g_upper[i, j] = np.random.randn() * 0.4

        w_orig = build_skew_hamiltonian(a, q_lower, g_upper)
        eig_orig = np.linalg.eigvals(w_orig)

        qg = pack_qg(n, q_lower, g_upper)
        a_work = a.copy(order='F')
        qg_work = qg.copy(order='F')

        a_out, qg_out, u1, u2, wr, wi, info = mb03xs(a_work, qg_work, jobu='U')

        assert info == 0

        eig_half = wr + 1j * wi
        eig_full = np.concatenate([eig_half, eig_half])

        eig_orig_sorted = np.sort(np.abs(eig_orig))
        eig_full_sorted = np.sort(np.abs(eig_full))

        np.testing.assert_allclose(
            eig_orig_sorted,
            eig_full_sorted,
            rtol=1e-10, atol=1e-12,
            err_msg="Eigenvalues should be preserved"
        )


class TestMB03XSEdgeCases:
    """Edge case tests for MB03XS."""

    def test_n_equals_1(self):
        """Test N=1 case (degenerate)."""
        n = 1
        a = np.array([[2.5]], dtype=float, order='F')
        qg = np.zeros((n, n + 1), dtype=float, order='F')

        a_out, qg_out, wr, wi, info = mb03xs(a.copy(order='F'), qg.copy(order='F'), jobu='N')

        assert info == 0
        np.testing.assert_allclose(a_out[0, 0], 2.5, rtol=1e-14)
        np.testing.assert_allclose(wr[0], 2.5, rtol=1e-14)
        np.testing.assert_allclose(wi[0], 0.0, atol=1e-14)

    def test_diagonal_a_matrix(self):
        """
        Test with diagonal A matrix and zero Q, G.

        Eigenvalues should be diagonal elements.
        """
        n = 3
        a = np.diag([1.0, 2.0, 3.0]).astype(float, order='F')
        qg = np.zeros((n, n + 1), dtype=float, order='F')

        a_out, qg_out, wr, wi, info = mb03xs(a.copy(order='F'), qg.copy(order='F'), jobu='N')

        assert info == 0
        np.testing.assert_allclose(sorted(wr), [1.0, 2.0, 3.0], rtol=1e-14)
        np.testing.assert_allclose(wi, np.zeros(n), atol=1e-14)

    def test_identity_a_matrix(self):
        """Test with A = I and zero Q, G."""
        n = 4
        a = np.eye(n, dtype=float, order='F')
        qg = np.zeros((n, n + 1), dtype=float, order='F')

        a_out, qg_out, u1, u2, wr, wi, info = mb03xs(
            a.copy(order='F'), qg.copy(order='F'), jobu='U'
        )

        assert info == 0
        np.testing.assert_allclose(wr, np.ones(n), rtol=1e-14)
        np.testing.assert_allclose(wi, np.zeros(n), atol=1e-14)
        np.testing.assert_allclose(u1.T @ u1 + u2.T @ u2, np.eye(n), rtol=1e-12, atol=1e-14)


class TestMB03XSScaling:
    """Tests for scaling robustness."""

    def test_large_elements(self):
        """
        Test with large matrix elements.

        Random seed: 111 (for reproducibility)
        """
        np.random.seed(111)
        n = 3
        scale = 1e6

        a = np.random.randn(n, n).astype(float, order='F') * scale
        q_lower = np.zeros((n, n), dtype=float, order='F')
        g_upper = np.zeros((n, n), dtype=float, order='F')
        for i in range(1, n):
            for j in range(i):
                q_lower[i, j] = np.random.randn() * scale
        for i in range(n - 1):
            for j in range(i + 1, n):
                g_upper[i, j] = np.random.randn() * scale

        w_orig = build_skew_hamiltonian(a, q_lower, g_upper)
        eig_orig = np.linalg.eigvals(w_orig)

        qg = pack_qg(n, q_lower, g_upper)
        a_work = a.copy(order='F')
        qg_work = qg.copy(order='F')

        a_out, qg_out, wr, wi, info = mb03xs(a_work, qg_work, jobu='N')

        assert info == 0
        eig_half = wr + 1j * wi
        eig_full = np.concatenate([eig_half, eig_half])

        eig_orig_sorted = np.sort(np.abs(eig_orig))
        eig_full_sorted = np.sort(np.abs(eig_full))
        np.testing.assert_allclose(eig_orig_sorted, eig_full_sorted, rtol=1e-8)

    def test_small_elements(self):
        """
        Test with small matrix elements.

        Random seed: 222 (for reproducibility)
        """
        np.random.seed(222)
        n = 3
        scale = 1e-6

        a = np.random.randn(n, n).astype(float, order='F') * scale
        q_lower = np.zeros((n, n), dtype=float, order='F')
        g_upper = np.zeros((n, n), dtype=float, order='F')
        for i in range(1, n):
            for j in range(i):
                q_lower[i, j] = np.random.randn() * scale
        for i in range(n - 1):
            for j in range(i + 1, n):
                g_upper[i, j] = np.random.randn() * scale

        w_orig = build_skew_hamiltonian(a, q_lower, g_upper)
        eig_orig = np.linalg.eigvals(w_orig)

        qg = pack_qg(n, q_lower, g_upper)
        a_work = a.copy(order='F')
        qg_work = qg.copy(order='F')

        a_out, qg_out, wr, wi, info = mb03xs(a_work, qg_work, jobu='N')

        assert info == 0
        eig_half = wr + 1j * wi
        eig_full = np.concatenate([eig_half, eig_half])

        eig_orig_sorted = np.sort(np.abs(eig_orig))
        eig_full_sorted = np.sort(np.abs(eig_full))
        np.testing.assert_allclose(eig_orig_sorted, eig_full_sorted, rtol=1e-8, atol=1e-18)
