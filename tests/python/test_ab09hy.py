"""
Tests for AB09HY - Cholesky factors of controllability and observability Grammians

AB09HY computes the Cholesky factors Su and Ru of the controllability
Grammian P = Su*Su' and observability Grammian Q = Ru'*Ru, respectively,
satisfying:

    A*P + P*A' + scalec^2*B*B' = 0             (1) continuous Lyapunov
    A'*Q + Q*A + scaleo^2*Cw'*Cw = 0           (2) continuous Lyapunov

where:
    Cw = Hw - Bw'*X
    Hw = inv(Dw)*C
    Bw = (B*D' + P*C')*inv(Dw')
    D*D' = Dw*Dw' (Dw upper triangular from RQ factorization)

and X is the stabilizing solution of the Riccati equation:
    Aw'*X + X*Aw + Hw'*Hw + X*Bw*Bw'*X = 0    (3)

with Aw = A - Bw*Hw.

Requirements:
- Matrix A must be stable (eigenvalues with negative real parts)
- Matrix A must be in real Schur form (quasi-upper triangular)
- Matrix D (P-by-M) must have full row rank (P <= M required)

Outputs:
- S: Upper triangular Cholesky factor Su (N-by-N)
- R: Upper triangular Cholesky factor Ru (N-by-N)
- SCALEC: Scaling factor for controllability Grammian
- SCALEO: Scaling factor for observability Grammian

Error codes:
- INFO = 0: success
- INFO = 1: A is not stable or not in real Schur form
- INFO = 2: Hamiltonian reduction failed
- INFO = 3: Hamiltonian reordering failed
- INFO = 4: Hamiltonian has < N stable eigenvalues
- INFO = 5: U11 singular in Riccati solver
- INFO = 6: D does not have full row rank
"""

import numpy as np
import pytest
from ctrlsys import ab09hy


class TestAB09HYBasic:
    """Basic functionality tests."""

    def test_simple_stable_system(self):
        """
        Test with a simple stable diagonal system in Schur form.

        For diagonal A (which is trivially in Schur form), the system is stable
        if all diagonal elements are negative. D must have full row rank (P <= M).

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n, m, p = 3, 2, 1

        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')

        b = np.array([
            [1.0, 0.5],
            [0.5, 1.0],
            [0.3, 0.7]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.5, 0.3]
        ], order='F', dtype=float)

        d = np.array([
            [1.0, 0.5]
        ], order='F', dtype=float)

        s, r, scalec, scaleo, rcond, info = ab09hy(n, m, p, a, b, c, d)

        assert info == 0, f"Expected info=0, got {info}"
        assert scalec > 0, "SCALEC must be positive"
        assert scaleo > 0, "SCALEO must be positive"
        assert 0 < rcond <= 1, "RCOND must be in (0, 1]"

        assert s.shape == (n, n)
        assert r.shape == (n, n)

        # Extract upper triangular parts (SLICOT only defines upper triangle)
        s_upper = np.triu(s)
        r_upper = np.triu(r)

        # Verify controllability Grammian: P = Su*Su'
        p_computed = s_upper @ s_upper.T

        # Verify observability Grammian: Q = Ru'*Ru
        q_computed = r_upper.T @ r_upper

        # Both Grammians should be positive semi-definite
        p_eigvals = np.linalg.eigvalsh(p_computed)
        q_eigvals = np.linalg.eigvalsh(q_computed)
        assert np.all(p_eigvals >= -1e-10), "P must be positive semi-definite"
        assert np.all(q_eigvals >= -1e-10), "Q must be positive semi-definite"

    def test_2x2_schur_block_system(self):
        """
        Test with a 2x2 Schur block (complex conjugate eigenvalues).

        A 2x2 block with form [[a, b], [-b, a]] has eigenvalues a +/- bi.
        For stability, a < 0.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n, m, p = 2, 2, 1

        a = np.array([
            [-1.0,  0.5],
            [-0.5, -1.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 1.0]
        ], order='F', dtype=float)

        d = np.array([
            [1.0, 0.5]
        ], order='F', dtype=float)

        s, r, scalec, scaleo, rcond, info = ab09hy(n, m, p, a, b, c, d)

        assert info == 0, f"Expected info=0, got {info}"
        assert scalec > 0
        assert scaleo > 0


class TestAB09HYGrammianProperties:
    """Tests for Grammian mathematical properties."""

    def test_lyapunov_residual_controllability(self):
        """
        Verify the controllability Grammian P = S*S' satisfies Lyapunov equation:
            A*P + P*A' + scalec^2*B*B' = 0

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n, m, p = 3, 2, 1

        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')
        b = np.array([
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0]
        ], order='F', dtype=float)
        c = np.array([[1.0, 1.0, 1.0]], order='F', dtype=float)
        d = np.array([[1.0, 0.5]], order='F', dtype=float)

        s, r, scalec, scaleo, rcond, info = ab09hy(n, m, p, a, b, c, d)
        assert info == 0

        p_gram = s @ s.T
        residual = a @ p_gram + p_gram @ a.T + (scalec ** 2) * (b @ b.T)

        np.testing.assert_allclose(residual, 0.0, atol=1e-10,
            err_msg="Controllability Lyapunov equation not satisfied")

    def test_grammians_positive_semidefinite(self):
        """
        Verify Grammians P = S*S' and Q = R'*R are positive semi-definite.

        P and Q should have non-negative eigenvalues.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n, m, p = 4, 3, 2

        a = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        d = np.zeros((p, m), order='F', dtype=float)
        d[0, 0] = 1.0
        d[1, 1] = 1.0

        s, r, scalec, scaleo, rcond, info = ab09hy(n, m, p, a, b, c, d)
        assert info == 0

        p_gram = s @ s.T
        q_gram = r.T @ r

        eig_p = np.linalg.eigvalsh(p_gram)
        eig_q = np.linalg.eigvalsh(q_gram)

        assert np.all(eig_p >= -1e-14), "P must be positive semi-definite"
        assert np.all(eig_q >= -1e-14), "Q must be positive semi-definite"

    def test_cholesky_factor_symmetry(self):
        """
        Verify P = S*S' and Q = R'*R are symmetric.

        Random seed: 111 (for reproducibility)
        """
        np.random.seed(111)
        n, m, p = 3, 2, 1

        a = np.diag([-0.5, -1.5, -2.5]).astype(float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')
        d = np.array([[1.0, 0.0]], order='F', dtype=float)

        s, r, scalec, scaleo, rcond, info = ab09hy(n, m, p, a, b, c, d)
        assert info == 0

        p_gram = s @ s.T
        q_gram = r.T @ r

        np.testing.assert_allclose(p_gram, p_gram.T, rtol=1e-14,
            err_msg="P must be symmetric")
        np.testing.assert_allclose(q_gram, q_gram.T, rtol=1e-14,
            err_msg="Q must be symmetric")


class TestAB09HYEdgeCases:
    """Edge case tests."""

    def test_square_d_matrix(self):
        """
        Test with square D matrix (P = M).

        Random seed: 222 (for reproducibility)
        """
        np.random.seed(222)
        n, m, p = 3, 2, 2

        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')
        b = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5]
        ], order='F', dtype=float)
        c = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], order='F', dtype=float)
        d = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        s, r, scalec, scaleo, rcond, info = ab09hy(n, m, p, a, b, c, d)

        assert info == 0
        assert scalec > 0
        assert scaleo > 0

    def test_siso_system(self):
        """
        Test with SISO system (M = P = 1).

        A must be in Schur form (upper triangular for real eigenvalues).

        Random seed: 333 (for reproducibility)
        """
        np.random.seed(333)
        n, m, p = 2, 1, 1

        a = np.array([
            [-1.0,  0.2],
            [ 0.0, -2.0]
        ], order='F', dtype=float)
        b = np.array([[1.0], [0.5]], order='F', dtype=float)
        c = np.array([[1.0, 0.5]], order='F', dtype=float)
        d = np.array([[1.0]], order='F', dtype=float)

        s, r, scalec, scaleo, rcond, info = ab09hy(n, m, p, a, b, c, d)

        assert info == 0
        assert s.shape == (n, n)
        assert r.shape == (n, n)

    def test_minimal_system_n1(self):
        """
        Test with minimal 1st order system.

        Random seed: 444 (for reproducibility)
        """
        np.random.seed(444)
        n, m, p = 1, 2, 1

        a = np.array([[-1.0]], order='F', dtype=float)
        b = np.array([[1.0, 0.5]], order='F', dtype=float)
        c = np.array([[1.0]], order='F', dtype=float)
        d = np.array([[1.0, 0.0]], order='F', dtype=float)

        s, r, scalec, scaleo, rcond, info = ab09hy(n, m, p, a, b, c, d)

        assert info == 0
        assert s.shape == (1, 1)
        assert r.shape == (1, 1)
        assert s[0, 0] > 0


class TestAB09HYErrorHandling:
    """Error handling tests."""

    def test_unstable_system(self):
        """
        Test INFO=1 for unstable system (positive eigenvalue).

        A matrix with positive real eigenvalue is not stable.
        """
        n, m, p = 2, 2, 1

        a = np.array([
            [1.0, 0.0],
            [0.0, -1.0]
        ], order='F', dtype=float)
        b = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)
        c = np.array([[1.0, 1.0]], order='F', dtype=float)
        d = np.array([[1.0, 0.5]], order='F', dtype=float)

        s, r, scalec, scaleo, rcond, info = ab09hy(n, m, p, a, b, c, d)

        assert info == 1, f"Expected info=1 for unstable system, got {info}"

    def test_rank_deficient_d(self):
        """
        Test INFO=6 when D does not have full row rank.

        A P-by-M matrix D with P > M cannot have full row rank.
        Also, D with zero rows doesn't have full row rank.
        """
        n, m, p = 2, 2, 2

        a = np.array([
            [-1.0, 0.0],
            [0.0, -2.0]
        ], order='F', dtype=float)
        b = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)
        c = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        d = np.array([
            [1.0, 0.0],
            [2.0, 0.0]
        ], order='F', dtype=float)

        s, r, scalec, scaleo, rcond, info = ab09hy(n, m, p, a, b, c, d)

        assert info == 6, f"Expected info=6 for rank-deficient D, got {info}"

    def test_p_greater_than_m_error(self):
        """
        Test parameter validation when P > M.

        The routine requires M >= P >= 0.
        """
        n, m, p = 2, 1, 2

        a = np.array([
            [-1.0, 0.0],
            [0.0, -2.0]
        ], order='F', dtype=float)
        b = np.array([[1.0], [0.5]], order='F', dtype=float)
        c = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)
        d = np.array([
            [1.0],
            [0.5]
        ], order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09hy(n, m, p, a, b, c, d)

    def test_negative_n_error(self):
        """Test error for negative n."""
        a = np.array([[-1.0]], order='F', dtype=float)
        b = np.array([[1.0]], order='F', dtype=float)
        c = np.array([[1.0]], order='F', dtype=float)
        d = np.array([[1.0]], order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09hy(-1, 1, 1, a, b, c, d)

    def test_negative_m_error(self):
        """Test error for negative m."""
        a = np.array([[-1.0]], order='F', dtype=float)
        b = np.array([[1.0]], order='F', dtype=float)
        c = np.array([[1.0]], order='F', dtype=float)
        d = np.array([[1.0]], order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09hy(1, -1, 1, a, b, c, d)

    def test_negative_p_error(self):
        """Test error for negative p."""
        a = np.array([[-1.0]], order='F', dtype=float)
        b = np.array([[1.0]], order='F', dtype=float)
        c = np.array([[1.0]], order='F', dtype=float)
        d = np.array([[1.0]], order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09hy(1, 1, -1, a, b, c, d)


class TestAB09HYQuickReturn:
    """Quick return tests for zero dimensions."""

    def test_n_zero(self):
        """
        Test quick return when n=0.

        With n=0, routine should return immediately with success.
        """
        n, m, p = 0, 2, 1

        a = np.zeros((0, 0), order='F', dtype=float)
        b = np.zeros((0, m), order='F', dtype=float)
        c = np.zeros((p, 0), order='F', dtype=float)
        d = np.array([[1.0, 0.5]], order='F', dtype=float)

        s, r, scalec, scaleo, rcond, info = ab09hy(n, m, p, a, b, c, d)

        assert info == 0
        assert scalec == 1.0
        assert scaleo == 1.0

    def test_m_zero(self):
        """
        Test quick return when m=0.

        With m=0, routine should return immediately with success.
        """
        n, m, p = 2, 0, 0

        a = np.array([
            [-1.0, 0.0],
            [0.0, -2.0]
        ], order='F', dtype=float)
        b = np.zeros((n, 0), order='F', dtype=float)
        c = np.zeros((0, n), order='F', dtype=float)
        d = np.zeros((0, 0), order='F', dtype=float)

        s, r, scalec, scaleo, rcond, info = ab09hy(n, m, p, a, b, c, d)

        assert info == 0
        assert scalec == 1.0
        assert scaleo == 1.0

    def test_p_zero(self):
        """
        Test quick return when p=0.

        With p=0, routine should return immediately with success.
        """
        n, m, p = 2, 2, 0

        a = np.array([
            [-1.0, 0.0],
            [0.0, -2.0]
        ], order='F', dtype=float)
        b = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)
        c = np.zeros((0, n), order='F', dtype=float)
        d = np.zeros((0, m), order='F', dtype=float)

        s, r, scalec, scaleo, rcond, info = ab09hy(n, m, p, a, b, c, d)

        assert info == 0
        assert scalec == 1.0
        assert scaleo == 1.0
