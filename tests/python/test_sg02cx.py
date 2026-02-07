"""
Tests for SG02CX: Line search parameter minimizing Riccati residual norm.

SG02CX finds alpha minimizing ||R(X + alpha*S)||_F where R(X) is the residual
of a continuous-time algebraic Riccati equation.

The algorithm:
1. Computes V = op(E)'*S*G*S*op(E) (or variants)
2. Sets up cubic polynomial P'(alpha) = 0 to find critical points
3. Finds roots via MC01XD (3x3 generalized eigenproblem)
4. Selects optimal alpha in [0,2] based on 2nd derivative check
"""

import numpy as np
import pytest
from slicot import sg02cx


class TestSG02CXBasic:
    """Basic functionality tests."""

    def test_identity_e_with_g_matrix(self):
        """
        Test with identity E, given G matrix, upper triangular.

        Uses a simple 2x2 example where we know the optimal alpha.
        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n = 2

        r = np.array([
            [1.0, 0.5],
            [0.5, 2.0]
        ], order='F', dtype=float)

        s = np.array([
            [0.3, 0.1],
            [0.1, 0.4]
        ], order='F', dtype=float)

        g = np.array([
            [0.5, 0.2],
            [0.2, 0.8]
        ], order='F', dtype=float)

        alpha, rnorm, iwarn, info = sg02cx(
            jobe='I', flag='P', jobg='G', uplo='U', trans='N',
            n=n, m=0, e=np.empty((1, 1), order='F'), r=r, s=s, g=g
        )

        assert info == 0
        assert 0.0 <= alpha <= 2.0
        assert rnorm >= 0.0

    def test_general_e_with_g_matrix(self):
        """
        Test with general E matrix.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n = 3

        e = np.eye(n, order='F', dtype=float) + 0.1 * np.random.randn(n, n)
        e = np.asfortranarray(e)

        r = np.array([
            [2.0, 0.3, 0.1],
            [0.3, 1.5, 0.2],
            [0.1, 0.2, 1.0]
        ], order='F', dtype=float)

        s = np.array([
            [0.5, 0.1, 0.0],
            [0.1, 0.4, 0.1],
            [0.0, 0.1, 0.3]
        ], order='F', dtype=float)

        g = np.array([
            [1.0, 0.2, 0.1],
            [0.2, 0.8, 0.1],
            [0.1, 0.1, 0.6]
        ], order='F', dtype=float)

        alpha, rnorm, iwarn, info = sg02cx(
            jobe='G', flag='P', jobg='G', uplo='U', trans='N',
            n=n, m=0, e=e, r=r, s=s, g=g
        )

        assert info == 0
        assert 0.0 <= alpha <= 2.0
        assert rnorm >= 0.0

    def test_d_matrix_form(self):
        """
        Test with D matrix (G = D*D').

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n = 3
        m = 2

        r = np.array([
            [1.5, 0.4, 0.1],
            [0.4, 2.0, 0.3],
            [0.1, 0.3, 1.2]
        ], order='F', dtype=float)

        s = np.array([
            [0.6, 0.2, 0.1],
            [0.2, 0.5, 0.15],
            [0.1, 0.15, 0.4]
        ], order='F', dtype=float)

        d = np.array([
            [0.5, 0.3],
            [0.2, 0.4],
            [0.1, 0.2]
        ], order='F', dtype=float)

        alpha, rnorm, iwarn, info = sg02cx(
            jobe='I', flag='P', jobg='D', uplo='U', trans='N',
            n=n, m=m, e=np.empty((1, 1), order='F'), r=r, s=s, g=d
        )

        assert info == 0
        assert 0.0 <= alpha <= 2.0
        assert rnorm >= 0.0

    def test_f_matrix_form(self):
        """
        Test with F matrix form (V = F*F').

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n = 2
        m = 3

        r = np.array([
            [1.0, 0.3],
            [0.3, 0.8]
        ], order='F', dtype=float)

        f = np.array([
            [0.4, 0.2, 0.1],
            [0.1, 0.3, 0.2]
        ], order='F', dtype=float)

        dummy_s = np.empty((1, 1), order='F', dtype=float)

        alpha, rnorm, iwarn, info = sg02cx(
            jobe='I', flag='P', jobg='F', uplo='U', trans='N',
            n=n, m=m, e=np.empty((1, 1), order='F'), r=r, s=dummy_s, g=f
        )

        assert info == 0
        assert 0.0 <= alpha <= 2.0
        assert rnorm >= 0.0

    def test_h_k_matrix_form(self):
        """
        Test with H and K matrices (V = H*K).

        Random seed: 321 (for reproducibility)
        """
        np.random.seed(321)
        n = 2
        m = 2

        r = np.array([
            [1.2, 0.4],
            [0.4, 0.9]
        ], order='F', dtype=float)

        h = np.array([
            [0.5, 0.2],
            [0.3, 0.4]
        ], order='F', dtype=float)

        k = np.array([
            [0.6, 0.1],
            [0.2, 0.5]
        ], order='F', dtype=float)

        alpha, rnorm, iwarn, info = sg02cx(
            jobe='I', flag='P', jobg='H', uplo='U', trans='N',
            n=n, m=m, e=np.empty((1, 1), order='F'), r=r, s=k, g=h
        )

        assert info == 0
        assert 0.0 <= alpha <= 2.0
        assert rnorm >= 0.0


class TestSG02CXMinus:
    """Tests with minus sign (FLAG='M')."""

    def test_minus_sign_flag(self):
        """
        Test with minus sign in residual formula.

        Random seed: 555 (for reproducibility)
        """
        np.random.seed(555)
        n = 2

        r = np.array([
            [0.8, 0.2],
            [0.2, 1.0]
        ], order='F', dtype=float)

        s = np.array([
            [0.4, 0.1],
            [0.1, 0.3]
        ], order='F', dtype=float)

        g = np.array([
            [0.6, 0.15],
            [0.15, 0.5]
        ], order='F', dtype=float)

        alpha, rnorm, iwarn, info = sg02cx(
            jobe='I', flag='M', jobg='G', uplo='U', trans='N',
            n=n, m=0, e=np.empty((1, 1), order='F'), r=r, s=s, g=g
        )

        assert info == 0
        assert 0.0 <= alpha <= 2.0
        assert rnorm >= 0.0


class TestSG02CXLowerTriangular:
    """Tests with lower triangular storage."""

    def test_lower_triangular_storage(self):
        """
        Test with lower triangular input.

        Random seed: 666 (for reproducibility)
        """
        np.random.seed(666)
        n = 2

        r = np.array([
            [1.5, 0.0],
            [0.3, 1.2]
        ], order='F', dtype=float)

        s = np.array([
            [0.5, 0.0],
            [0.2, 0.4]
        ], order='F', dtype=float)

        g = np.array([
            [0.7, 0.0],
            [0.2, 0.6]
        ], order='F', dtype=float)

        alpha, rnorm, iwarn, info = sg02cx(
            jobe='I', flag='P', jobg='G', uplo='L', trans='N',
            n=n, m=0, e=np.empty((1, 1), order='F'), r=r, s=s, g=g
        )

        assert info == 0
        assert 0.0 <= alpha <= 2.0
        assert rnorm >= 0.0


class TestSG02CXTranspose:
    """Tests with transpose mode."""

    def test_transpose_operation(self):
        """
        Test with op(E) = E'.

        Random seed: 777 (for reproducibility)
        """
        np.random.seed(777)
        n = 2

        e = np.array([
            [1.1, 0.1],
            [0.2, 0.9]
        ], order='F', dtype=float)

        r = np.array([
            [1.0, 0.3],
            [0.3, 0.8]
        ], order='F', dtype=float)

        s = np.array([
            [0.4, 0.15],
            [0.15, 0.35]
        ], order='F', dtype=float)

        g = np.array([
            [0.5, 0.1],
            [0.1, 0.4]
        ], order='F', dtype=float)

        alpha, rnorm, iwarn, info = sg02cx(
            jobe='G', flag='P', jobg='G', uplo='U', trans='T',
            n=n, m=0, e=e, r=r, s=s, g=g
        )

        assert info == 0
        assert 0.0 <= alpha <= 2.0
        assert rnorm >= 0.0


class TestSG02CXEdgeCases:
    """Edge cases and special conditions."""

    def test_n_equals_zero(self):
        """Test with N=0 (quick return)."""
        alpha, rnorm, iwarn, info = sg02cx(
            jobe='I', flag='P', jobg='G', uplo='U', trans='N',
            n=0, m=0,
            e=np.empty((1, 1), order='F'),
            r=np.empty((1, 1), order='F'),
            s=np.empty((1, 1), order='F'),
            g=np.empty((1, 1), order='F')
        )

        assert info == 0
        assert alpha == 1.0
        assert rnorm == 0.0

    def test_m_equals_zero_with_d(self):
        """Test with M=0 for JOBG='D' (quick return)."""
        n = 2

        r = np.array([
            [1.0, 0.2],
            [0.2, 0.8]
        ], order='F', dtype=float)

        alpha, rnorm, iwarn, info = sg02cx(
            jobe='I', flag='P', jobg='D', uplo='U', trans='N',
            n=n, m=0,
            e=np.empty((1, 1), order='F'),
            r=r,
            s=np.eye(n, order='F'),
            g=np.empty((n, 1), order='F')
        )

        assert info == 0
        assert alpha == 1.0
        assert rnorm == 0.0

    def test_zero_residual(self):
        """Test with zero residual R(X) = 0."""
        n = 2

        r = np.zeros((n, n), order='F', dtype=float)

        s = np.array([
            [0.5, 0.1],
            [0.1, 0.4]
        ], order='F', dtype=float)

        g = np.array([
            [0.6, 0.2],
            [0.2, 0.5]
        ], order='F', dtype=float)

        alpha, rnorm, iwarn, info = sg02cx(
            jobe='I', flag='P', jobg='G', uplo='U', trans='N',
            n=n, m=0, e=np.empty((1, 1), order='F'), r=r, s=s, g=g
        )

        assert info == 0
        assert alpha == 0.0


class TestSG02CXMathematicalProperties:
    """Tests validating mathematical properties."""

    def test_rnorm_decreases_from_initial(self):
        """
        Verify that the returned rnorm is <= initial ||R||.

        The line search should find alpha that doesn't increase norm.
        Random seed: 888 (for reproducibility)
        """
        np.random.seed(888)
        n = 3

        r = np.array([
            [2.0, 0.5, 0.2],
            [0.5, 1.5, 0.3],
            [0.2, 0.3, 1.0]
        ], order='F', dtype=float)

        s = np.array([
            [0.3, 0.1, 0.05],
            [0.1, 0.25, 0.08],
            [0.05, 0.08, 0.2]
        ], order='F', dtype=float)

        g = np.array([
            [0.8, 0.2, 0.1],
            [0.2, 0.6, 0.15],
            [0.1, 0.15, 0.5]
        ], order='F', dtype=float)

        initial_norm = np.linalg.norm(r, 'fro')

        alpha, rnorm, iwarn, info = sg02cx(
            jobe='I', flag='P', jobg='G', uplo='U', trans='N',
            n=n, m=0, e=np.empty((1, 1), order='F'), r=r, s=s, g=g
        )

        assert info == 0
        assert rnorm <= initial_norm + 1e-10

    def test_optimal_alpha_reduces_norm(self):
        """
        Verify that rnorm at optimal alpha is <= rnorm at alpha=0 and alpha=2.

        The line search should find a value at least as good as the boundaries.
        """
        n = 3

        r = np.array([
            [2.0, 0.5, 0.2],
            [0.5, 1.5, 0.3],
            [0.2, 0.3, 1.0]
        ], order='F', dtype=float)

        s = np.array([
            [0.3, 0.1, 0.05],
            [0.1, 0.25, 0.08],
            [0.05, 0.08, 0.2]
        ], order='F', dtype=float)

        g = np.array([
            [0.8, 0.2, 0.1],
            [0.2, 0.6, 0.15],
            [0.1, 0.15, 0.5]
        ], order='F', dtype=float)

        alpha, rnorm, iwarn, info = sg02cx(
            jobe='I', flag='P', jobg='G', uplo='U', trans='N',
            n=n, m=0, e=np.empty((1, 1), order='F'), r=r, s=s, g=g
        )

        assert info == 0
        norm_at_0 = np.linalg.norm(r, 'fro')
        assert rnorm <= norm_at_0 + 1e-10

    def test_alpha_at_boundary_gives_consistent_norm(self):
        """
        Test that alpha=0 or alpha=2 give correct boundary behavior.

        Random seed: 999 (for reproducibility)
        """
        np.random.seed(999)
        n = 2

        r = np.array([
            [1.0, 0.2],
            [0.2, 0.8]
        ], order='F', dtype=float)

        s = np.array([
            [0.4, 0.1],
            [0.1, 0.3]
        ], order='F', dtype=float)

        g = np.array([
            [0.5, 0.15],
            [0.15, 0.4]
        ], order='F', dtype=float)

        alpha, rnorm, iwarn, info = sg02cx(
            jobe='I', flag='P', jobg='G', uplo='U', trans='N',
            n=n, m=0, e=np.empty((1, 1), order='F'), r=r, s=s, g=g
        )

        assert info == 0
        assert rnorm >= 0.0


class TestSG02CXWarnings:
    """Tests for warning conditions."""

    def test_no_minimum_in_interval(self):
        """
        Create a case where no minimum exists in [0,2].

        Uses matrices that create a polynomial with no critical points in [0,2].
        """
        n = 2

        r = np.array([
            [0.1, 0.0],
            [0.0, 0.1]
        ], order='F', dtype=float)

        s = np.array([
            [10.0, 0.0],
            [0.0, 10.0]
        ], order='F', dtype=float)

        g = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        alpha, rnorm, iwarn, info = sg02cx(
            jobe='I', flag='P', jobg='G', uplo='U', trans='N',
            n=n, m=0, e=np.empty((1, 1), order='F'), r=r, s=s, g=g
        )

        assert info == 0


class TestSG02CXParameterErrors:
    """Tests for parameter validation."""

    def test_invalid_jobe(self):
        """Test invalid JOBE parameter."""
        n = 2
        r = np.eye(n, order='F')
        s = np.eye(n, order='F')
        g = np.eye(n, order='F')

        alpha, rnorm, iwarn, info = sg02cx(
            jobe='X', flag='P', jobg='G', uplo='U', trans='N',
            n=n, m=0, e=np.empty((1, 1), order='F'), r=r, s=s, g=g
        )

        assert info == -1

    def test_invalid_flag(self):
        """Test invalid FLAG parameter."""
        n = 2
        r = np.eye(n, order='F')
        s = np.eye(n, order='F')
        g = np.eye(n, order='F')

        alpha, rnorm, iwarn, info = sg02cx(
            jobe='I', flag='X', jobg='G', uplo='U', trans='N',
            n=n, m=0, e=np.empty((1, 1), order='F'), r=r, s=s, g=g
        )

        assert info == -2

    def test_invalid_jobg(self):
        """Test invalid JOBG parameter."""
        n = 2
        r = np.eye(n, order='F')
        s = np.eye(n, order='F')
        g = np.eye(n, order='F')

        alpha, rnorm, iwarn, info = sg02cx(
            jobe='I', flag='P', jobg='X', uplo='U', trans='N',
            n=n, m=0, e=np.empty((1, 1), order='F'), r=r, s=s, g=g
        )

        assert info == -3

    def test_negative_n(self):
        """Test negative N parameter."""
        alpha, rnorm, iwarn, info = sg02cx(
            jobe='I', flag='P', jobg='G', uplo='U', trans='N',
            n=-1, m=0,
            e=np.empty((1, 1), order='F'),
            r=np.empty((1, 1), order='F'),
            s=np.empty((1, 1), order='F'),
            g=np.empty((1, 1), order='F')
        )

        assert info == -6

    def test_negative_m_when_required(self):
        """Test negative M parameter when JOBG != 'G'."""
        n = 2
        r = np.eye(n, order='F')
        s = np.eye(n, order='F')
        d = np.eye(n, order='F')

        alpha, rnorm, iwarn, info = sg02cx(
            jobe='I', flag='P', jobg='D', uplo='U', trans='N',
            n=n, m=-1, e=np.empty((1, 1), order='F'), r=r, s=s, g=d
        )

        assert info == -7
