"""
Tests for MB04PB: Computation of the Paige/Van Loan (PVL) form of a Hamiltonian matrix (block algorithm).

MB04PB reduces a Hamiltonian matrix H = [[A, G], [Q, -A^T]] where G and Q are symmetric,
to PVL form using an orthogonal symplectic transformation U such that U^T H U has
upper Hessenberg A and diagonal Q.
"""

import numpy as np
import pytest

from ctrlsys import mb04pb, mb04wp


def build_hamiltonian(a, qg, n):
    """Build full 2N x 2N Hamiltonian from A and packed QG storage.

    QG layout: columns 0..N-1 store lower triangle of Q,
               columns 1..N store upper triangle of G.
    """
    Q_lower = np.tril(qg[:, :n])
    Q = Q_lower + Q_lower.T - np.diag(np.diag(Q_lower))

    G_upper = np.triu(qg[:, 1:n+1])
    G = G_upper + G_upper.T - np.diag(np.diag(G_upper))

    H = np.zeros((2*n, 2*n), order='F', dtype=float)
    H[:n, :n] = a
    H[:n, n:] = G
    H[n:, :n] = Q
    H[n:, n:] = -a.T
    return H


def build_pvl_hamiltonian(a_out, qg_out, n):
    """Build Hamiltonian from PVL-reduced output.

    After reduction: Aout is upper Hessenberg, Qout is diagonal.
    QG storage: Q diagonal at qg[i,i], G upper triangle at qg[i,j+1] for j>=i.
    """
    q_diag_vals = np.array([qg_out[i, i] for i in range(n)])
    Q_diag = np.diag(q_diag_vals)

    G_upper = np.triu(qg_out[:, 1:n+1])
    G = G_upper + G_upper.T - np.diag(np.diag(G_upper))

    H = np.zeros((2*n, 2*n), order='F', dtype=float)
    H[:n, :n] = np.triu(a_out, -1)
    H[:n, n:] = G
    H[n:, :n] = Q_diag
    H[n:, n:] = -np.triu(a_out, -1).T
    return H


def reconstruct_u(a_out, qg_out, cs, tau, n, ilo):
    """Reconstruct orthogonal symplectic U from MB04PB output using MB04WP.

    U = [[U1, U2], [-U2, U1]] with U^T U = I_{2n}.
    """
    u1 = a_out.copy(order='F')
    u2 = qg_out[:, :n].copy(order='F')

    u1_out, u2_out, info = mb04wp(n=n, ilo=ilo, u1=u1, u2=u2, cs=cs, tau=tau)
    assert info == 0

    U = np.zeros((2*n, 2*n), order='F', dtype=float)
    U[:n, :n] = u1_out
    U[:n, n:] = u2_out
    U[n:, :n] = -u2_out
    U[n:, n:] = u1_out
    return U


class TestMB04PBBasic:
    """Basic functionality tests using HTML doc example data."""

    def test_html_doc_example(self):
        """
        Test MB04PB using the example from SLICOT HTML documentation.
        N=5, ILO=1 (full reduction from scratch).
        """
        n = 5
        ilo = 1

        a = np.array([
            [0.9501, 0.7621, 0.6154, 0.4057, 0.0579],
            [0.2311, 0.4565, 0.7919, 0.9355, 0.3529],
            [0.6068, 0.0185, 0.9218, 0.9169, 0.8132],
            [0.4860, 0.8214, 0.7382, 0.4103, 0.0099],
            [0.8913, 0.4447, 0.1763, 0.8936, 0.1389],
        ], order='F', dtype=float)

        qg = np.array([
            [0.3869, 0.4055, 0.2140, 1.0224, 1.1103, 0.7016],
            [1.3801, 0.7567, 1.4936, 1.2913, 0.9515, 1.1755],
            [0.7993, 1.7598, 1.6433, 1.0503, 0.8839, 1.1010],
            [1.2019, 1.1956, 0.9346, 0.6824, 0.7590, 1.1364],
            [0.8780, 0.9029, 1.6565, 1.1022, 0.7408, 0.3793],
        ], order='F', dtype=float)

        H_orig = build_hamiltonian(a.copy(), qg.copy(), n)

        a_out, qg_out, cs, tau, info = mb04pb(n, ilo, a, qg)

        assert info == 0
        assert a_out.shape == (n, n)
        assert qg_out.shape == (n, n + 1)
        assert cs.shape == (2 * n - 2,)
        assert tau.shape == (n - 1,)

        expected_a = np.array([
            [0.9501, -1.5494, 0.5268, 0.3187, -0.6890],
            [-2.4922, 2.0907, -1.3598, 0.5682, 0.5618],
            [0.0000, -1.7723, 0.3960, -0.2624, -0.3709],
            [0.0000, 0.0000, -0.2648, 0.2136, -0.3226],
            [0.0000, 0.0000, 0.0000, -0.2308, 0.2319],
        ], order='F', dtype=float)

        np.testing.assert_allclose(a_out[0, 0], expected_a[0, 0], rtol=1e-3)
        np.testing.assert_allclose(np.abs(a_out[1, 0]), np.abs(expected_a[1, 0]), rtol=1e-3)

        U = reconstruct_u(a_out, qg_out, cs, tau, n, ilo)
        H_pvl = build_pvl_hamiltonian(a_out, qg_out, n)
        H_check = U @ H_pvl @ U.T
        np.testing.assert_allclose(H_check, H_orig, atol=1e-12)

    def test_small_matrix_n2(self):
        """Test MB04PB with N=2 minimal non-trivial case."""
        n = 2
        ilo = 1

        a = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ], order='F', dtype=float)

        qg = np.array([
            [0.5, 1.0, 2.0],
            [1.5, 2.5, 3.0],
        ], order='F', dtype=float)

        H_orig = build_hamiltonian(a.copy(), qg.copy(), n)

        a_out, qg_out, cs, tau, info = mb04pb(n, ilo, a, qg)

        assert info == 0
        assert a_out.shape == (n, n)
        assert qg_out.shape == (n, n + 1)

        U = reconstruct_u(a_out, qg_out, cs, tau, n, ilo)
        H_pvl = build_pvl_hamiltonian(a_out, qg_out, n)
        H_check = U @ H_pvl @ U.T
        np.testing.assert_allclose(H_check, H_orig, atol=1e-13)


class TestMB04PBEdgeCases:
    """Edge case tests."""

    def test_n_equals_zero(self):
        """Test MB04PB with N=0 (quick return)."""
        n = 0
        ilo = 1
        a = np.zeros((0, 0), order='F', dtype=float)
        qg = np.zeros((0, 1), order='F', dtype=float)
        a_out, qg_out, cs, tau, info = mb04pb(n, ilo, a, qg)
        assert info == 0

    def test_n_equals_one(self):
        """Test MB04PB with N=1 (trivial case, no reduction needed)."""
        n = 1
        ilo = 1
        a = np.array([[2.5]], order='F', dtype=float)
        qg = np.array([[1.0, 3.0]], order='F', dtype=float)
        a_out, qg_out, cs, tau, info = mb04pb(n, ilo, a, qg)
        assert info == 0
        assert a_out.shape == (1, 1)
        np.testing.assert_allclose(a_out[0, 0], 2.5, rtol=1e-14)


class TestMB04PBILOParameter:
    """Tests for ILO parameter handling."""

    def test_ilo_greater_than_one(self):
        """Test MB04PB with ILO > 1 (partial reduction)."""
        n = 4
        ilo = 2

        a = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 5.0, 6.0, 7.0],
            [0.0, 8.0, 9.0, 10.0],
            [0.0, 11.0, 12.0, 13.0],
        ], order='F', dtype=float)

        qg = np.array([
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 0.5, 1.5, 2.5, 3.5],
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 1.5, 2.5, 3.5, 4.5],
        ], order='F', dtype=float)

        a_out, qg_out, cs, tau, info = mb04pb(n, ilo, a, qg)

        assert info == 0
        np.testing.assert_allclose(a_out[0, 0], 1.0, rtol=1e-14)


class TestMB04PBMathProperties:
    """Mathematical property validation tests."""

    def test_eigenvalue_preservation(self):
        """
        Validate Hamiltonian eigenvalue preservation under PVL transformation.

        The transformation U^T H U should preserve all eigenvalues of H.
        Random seed: 42
        """
        np.random.seed(42)
        n = 6
        ilo = 1

        a = np.random.randn(n, n).astype(float, order='F')
        qg_raw = np.random.randn(n, n + 1).astype(float, order='F')

        Q_lower = np.tril(qg_raw[:, :n])
        Q_sym = Q_lower + Q_lower.T - np.diag(np.diag(Q_lower))
        G_upper = np.triu(qg_raw[:, 1:n+1])
        G_sym = G_upper + G_upper.T - np.diag(np.diag(G_upper))

        qg = np.zeros((n, n + 1), order='F', dtype=float)
        for i in range(n):
            for j in range(i + 1):
                qg[i, j] = Q_sym[i, j]
        for i in range(n):
            for j in range(i, n):
                qg[i, j + 1] = G_sym[i, j]

        H_orig = build_hamiltonian(a.copy(), qg.copy(), n)
        eig_before = np.sort(np.linalg.eigvals(H_orig))

        a_out, qg_out, cs, tau, info = mb04pb(n, ilo, a, qg)
        assert info == 0

        U = reconstruct_u(a_out, qg_out, cs, tau, n, ilo)
        H_pvl = build_pvl_hamiltonian(a_out, qg_out, n)
        eig_after = np.sort(np.linalg.eigvals(H_pvl))

        idx_before = np.argsort(eig_before.real + 1j * eig_before.imag)
        idx_after = np.argsort(eig_after.real + 1j * eig_after.imag)
        np.testing.assert_allclose(
            np.sort(np.abs(eig_before[idx_before])),
            np.sort(np.abs(eig_after[idx_after])),
            rtol=1e-10
        )

    def test_orthogonal_symplectic_u(self):
        """
        Validate that U is orthogonal symplectic: U^T U = I and U^T J U = J.

        Random seed: 123
        """
        np.random.seed(123)
        n = 5
        ilo = 1

        a = np.random.randn(n, n).astype(float, order='F')
        qg_raw = np.random.randn(n, n + 1).astype(float, order='F')
        Q_lower = np.tril(qg_raw[:, :n])
        Q_sym = Q_lower + Q_lower.T - np.diag(np.diag(Q_lower))
        G_upper = np.triu(qg_raw[:, 1:n+1])
        G_sym = G_upper + G_upper.T - np.diag(np.diag(G_upper))

        qg = np.zeros((n, n + 1), order='F', dtype=float)
        for i in range(n):
            for j in range(i + 1):
                qg[i, j] = Q_sym[i, j]
        for i in range(n):
            for j in range(i, n):
                qg[i, j + 1] = G_sym[i, j]

        a_out, qg_out, cs, tau, info = mb04pb(n, ilo, a, qg)
        assert info == 0

        U = reconstruct_u(a_out, qg_out, cs, tau, n, ilo)

        np.testing.assert_allclose(U.T @ U, np.eye(2*n), atol=1e-13)

        J = np.zeros((2*n, 2*n), dtype=float)
        J[:n, n:] = np.eye(n)
        J[n:, :n] = -np.eye(n)
        np.testing.assert_allclose(U.T @ J @ U, J, atol=1e-13)

    def test_similarity_transformation_residual(self):
        """
        Validate U^T H U = H_pvl by checking residual directly.

        Random seed: 456
        """
        np.random.seed(456)
        n = 4
        ilo = 1

        a = np.random.randn(n, n).astype(float, order='F')
        qg_raw = np.random.randn(n, n + 1).astype(float, order='F')
        Q_lower = np.tril(qg_raw[:, :n])
        Q_sym = Q_lower + Q_lower.T - np.diag(np.diag(Q_lower))
        G_upper = np.triu(qg_raw[:, 1:n+1])
        G_sym = G_upper + G_upper.T - np.diag(np.diag(G_upper))

        qg = np.zeros((n, n + 1), order='F', dtype=float)
        for i in range(n):
            for j in range(i + 1):
                qg[i, j] = Q_sym[i, j]
        for i in range(n):
            for j in range(i, n):
                qg[i, j + 1] = G_sym[i, j]

        H_orig = build_hamiltonian(a.copy(), qg.copy(), n)

        a_out, qg_out, cs, tau, info = mb04pb(n, ilo, a, qg)
        assert info == 0

        U = reconstruct_u(a_out, qg_out, cs, tau, n, ilo)
        H_pvl = build_pvl_hamiltonian(a_out, qg_out, n)

        residual = U.T @ H_orig @ U - H_pvl
        np.testing.assert_allclose(residual, np.zeros((2*n, 2*n)), atol=1e-12)

    def test_hessenberg_structure(self):
        """
        Validate that U^T H U has upper Hessenberg A block.

        A_out stores reflector data below subdiagonal, so check
        the transformed Hamiltonian instead.

        Random seed: 789
        """
        np.random.seed(789)
        n = 7
        ilo = 1

        a = np.random.randn(n, n).astype(float, order='F')
        qg_raw = np.random.randn(n, n + 1).astype(float, order='F')
        Q_lower = np.tril(qg_raw[:, :n])
        Q_sym = Q_lower + Q_lower.T - np.diag(np.diag(Q_lower))
        G_upper = np.triu(qg_raw[:, 1:n+1])
        G_sym = G_upper + G_upper.T - np.diag(np.diag(G_upper))

        qg = np.zeros((n, n + 1), order='F', dtype=float)
        for i in range(n):
            for j in range(i + 1):
                qg[i, j] = Q_sym[i, j]
        for i in range(n):
            for j in range(i, n):
                qg[i, j + 1] = G_sym[i, j]

        H_orig = build_hamiltonian(a.copy(), qg.copy(), n)

        a_out, qg_out, cs, tau, info = mb04pb(n, ilo, a, qg)
        assert info == 0

        U = reconstruct_u(a_out, qg_out, cs, tau, n, ilo)
        H_trans = U.T @ H_orig @ U

        A_block = H_trans[:n, :n]
        for j in range(n):
            for i in range(j + 2, n):
                assert abs(A_block[i, j]) < 1e-12, \
                    f"A[{i},{j}] = {A_block[i,j]} should be zero (upper Hessenberg)"

    def test_diagonal_q(self):
        """
        Validate that Qout is diagonal after PVL reduction.

        Random seed: 888
        """
        np.random.seed(888)
        n = 6
        ilo = 1

        a = np.random.randn(n, n).astype(float, order='F')
        qg_raw = np.random.randn(n, n + 1).astype(float, order='F')
        Q_lower = np.tril(qg_raw[:, :n])
        Q_sym = Q_lower + Q_lower.T - np.diag(np.diag(Q_lower))
        G_upper = np.triu(qg_raw[:, 1:n+1])
        G_sym = G_upper + G_upper.T - np.diag(np.diag(G_upper))

        qg = np.zeros((n, n + 1), order='F', dtype=float)
        for i in range(n):
            for j in range(i + 1):
                qg[i, j] = Q_sym[i, j]
        for i in range(n):
            for j in range(i, n):
                qg[i, j + 1] = G_sym[i, j]

        H_orig = build_hamiltonian(a.copy(), qg.copy(), n)

        a_out, qg_out, cs, tau, info = mb04pb(n, ilo, a, qg)
        assert info == 0

        U = reconstruct_u(a_out, qg_out, cs, tau, n, ilo)
        H_pvl = build_pvl_hamiltonian(a_out, qg_out, n)
        H_check = U.T @ H_orig @ U

        Q_transformed = H_check[n:, :n]
        for i in range(n):
            for j in range(n):
                if i != j:
                    assert abs(Q_transformed[i, j]) < 1e-12, \
                        f"Q[{i},{j}] = {Q_transformed[i,j]} should be zero (diagonal Q)"

    def test_deterministic_results(self):
        """
        Verify that results are deterministic with same input.

        Random seed: 999
        """
        np.random.seed(999)
        n = 3
        ilo = 1

        a = np.random.randn(n, n).astype(float, order='F')
        qg = np.random.randn(n, n + 1).astype(float, order='F')

        a1, qg1, cs1, tau1, info1 = mb04pb(n, ilo, a.copy(), qg.copy())
        a2, qg2, cs2, tau2, info2 = mb04pb(n, ilo, a.copy(), qg.copy())

        assert info1 == 0
        assert info2 == 0
        np.testing.assert_allclose(a1, a2, rtol=1e-14)
        np.testing.assert_allclose(qg1, qg2, rtol=1e-14)
        np.testing.assert_allclose(cs1, cs2, rtol=1e-14)
        np.testing.assert_allclose(tau1, tau2, rtol=1e-14)

    def test_larger_matrix_n10(self):
        """
        Test with larger matrix N=10 to exercise blocked code path.

        Random seed: 314
        """
        np.random.seed(314)
        n = 10
        ilo = 1

        a = np.random.randn(n, n).astype(float, order='F')
        qg_raw = np.random.randn(n, n + 1).astype(float, order='F')
        Q_lower = np.tril(qg_raw[:, :n])
        Q_sym = Q_lower + Q_lower.T - np.diag(np.diag(Q_lower))
        G_upper = np.triu(qg_raw[:, 1:n+1])
        G_sym = G_upper + G_upper.T - np.diag(np.diag(G_upper))

        qg = np.zeros((n, n + 1), order='F', dtype=float)
        for i in range(n):
            for j in range(i + 1):
                qg[i, j] = Q_sym[i, j]
        for i in range(n):
            for j in range(i, n):
                qg[i, j + 1] = G_sym[i, j]

        H_orig = build_hamiltonian(a.copy(), qg.copy(), n)

        a_out, qg_out, cs, tau, info = mb04pb(n, ilo, a, qg)
        assert info == 0

        U = reconstruct_u(a_out, qg_out, cs, tau, n, ilo)
        H_pvl = build_pvl_hamiltonian(a_out, qg_out, n)
        residual = U.T @ H_orig @ U - H_pvl
        np.testing.assert_allclose(residual, np.zeros((2*n, 2*n)), atol=1e-11)


class TestMB04PBErrorHandling:
    """Error handling tests."""

    def test_invalid_n_negative(self):
        n = -1
        ilo = 1
        a = np.zeros((1, 1), order='F', dtype=float)
        qg = np.zeros((1, 2), order='F', dtype=float)
        with pytest.raises((ValueError, RuntimeError)):
            mb04pb(n, ilo, a, qg)

    def test_invalid_ilo_zero(self):
        n = 3
        ilo = 0
        a = np.zeros((n, n), order='F', dtype=float)
        qg = np.zeros((n, n + 1), order='F', dtype=float)
        with pytest.raises((ValueError, RuntimeError)):
            mb04pb(n, ilo, a, qg)

    def test_invalid_ilo_greater_than_n(self):
        n = 3
        ilo = 5
        a = np.zeros((n, n), order='F', dtype=float)
        qg = np.zeros((n, n + 1), order='F', dtype=float)
        with pytest.raises((ValueError, RuntimeError)):
            mb04pb(n, ilo, a, qg)
