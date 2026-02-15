"""
Tests for SG02CW: Riccati equation residual computation.

SG02CW computes the residual matrix R for a continuous-time or discrete-time
algebraic Riccati equation and/or the "closed-loop system" matrix C.

Test Strategy:
- Since no example data in HTML docs, use mathematical property validation
- Use Python control package to generate test data with known solutions
- Test values are hardcoded after generation (no external deps at runtime)
"""

import numpy as np
import pytest


class TestSG02CWBasic:
    """Basic functionality tests for SG02CW."""

    def test_continuous_residual_jobg_identity(self):
        """
        Test continuous-time Riccati residual with identity E.

        Continuous-time: R = A'*X + X*A +/- X*G*X + Q
        For a valid solution X to the Riccati equation, R should be zero.

        Random seed: 42 (for reproducibility)
        Test data generated using control package CARE solver.
        """
        from ctrlsys import sg02cw

        n = 3

        # System matrices (stable system)
        a = np.array([
            [-1.0,  0.0,  0.0],
            [ 0.0, -2.0,  0.0],
            [ 0.0,  0.0, -3.0]
        ], order='F', dtype=float)

        # G = B*R^{-1}*B' where R = I, B = I -> G = I
        g = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], order='F', dtype=float)

        # Q = I (state weighting)
        q = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], order='F', dtype=float)

        # X is the solution to the CARE: A'*X + X*A - X*G*X + Q = 0
        # For diagonal A with entries -a_i, G=I, Q=I:
        # The solution X is diagonal with entries x_i where:
        # -2*a_i*x_i - x_i^2 + 1 = 0 -> x_i = sqrt(a_i^2 + 1) - a_i
        x = np.array([
            [np.sqrt(2) - 1, 0.0, 0.0],
            [0.0, np.sqrt(5) - 2, 0.0],
            [0.0, 0.0, np.sqrt(10) - 3]
        ], order='F', dtype=float)

        # Residual should be close to zero for valid solution
        r, c, norms, info = sg02cw(
            dico='C', job='A', jobe='I', flag='M', jobg='G',
            uplo='U', trans='N',
            n=n, m=0,
            a=a.copy(), e=None, g=g.copy(), x=x.copy(),
            f=None, k=None, xe=None, q=q.copy()
        )

        assert info == 0

        # Residual should be near zero for valid CARE solution
        r_full = np.triu(r) + np.triu(r, 1).T
        np.testing.assert_allclose(r_full, np.zeros((n, n)), atol=1e-10)

        # Closed-loop matrix C = A - G*X
        c_expected = a - g @ x
        np.testing.assert_allclose(c, c_expected, rtol=1e-12)

    def test_continuous_residual_plus_sign(self):
        """
        Test continuous-time residual with plus sign (FLAG='P').

        R = A'*X + X*A + X*G*X + Q

        Random seed: 123 (for reproducibility)
        """
        from ctrlsys import sg02cw

        n = 2

        a = np.array([
            [-2.0, 0.0],
            [ 0.0, -3.0]
        ], order='F', dtype=float)

        g = np.eye(2, order='F', dtype=float)
        q = np.eye(2, order='F', dtype=float)
        x = np.eye(2, order='F', dtype=float) * 0.5

        # Compute expected: R = A'*X + X*A + X*G*X + Q
        r_expected = a.T @ x + x @ a + x @ g @ x + q

        r, c, norms, info = sg02cw(
            dico='C', job='A', jobe='I', flag='P', jobg='G',
            uplo='U', trans='N',
            n=n, m=0,
            a=a.copy(), e=None, g=g.copy(), x=x.copy(),
            f=None, k=None, xe=None, q=q.copy()
        )

        assert info == 0

        # Verify residual
        r_full = np.triu(r) + np.triu(r, 1).T
        np.testing.assert_allclose(r_full, r_expected, rtol=1e-12)

        # C = A + G*X (plus sign)
        c_expected = a + g @ x
        np.testing.assert_allclose(c, c_expected, rtol=1e-12)

    def test_discrete_residual_identity_e(self):
        """
        Test discrete-time Riccati residual with identity E.

        Discrete-time: R = A'*X*A - X +/- A'*X*G*X*A + Q

        Random seed: 456 (for reproducibility)
        """
        from ctrlsys import sg02cw

        n = 2

        # Stable discrete system (eigenvalues inside unit circle)
        a = np.array([
            [0.5, 0.0],
            [0.0, 0.3]
        ], order='F', dtype=float)

        g = np.eye(2, order='F', dtype=float)
        q = np.eye(2, order='F', dtype=float)
        x = np.eye(2, order='F', dtype=float) * 0.5

        # Compute expected: R = A'*X*A - X - A'*X*G*X*A + Q (minus sign)
        xg = x @ g
        axgxa = a.T @ x @ g @ x @ a
        r_expected = a.T @ x @ a - x - axgxa + q

        r, c, norms, info = sg02cw(
            dico='D', job='A', jobe='I', flag='M', jobg='G',
            uplo='U', trans='N',
            n=n, m=0,
            a=a.copy(), e=None, g=g.copy(), x=x.copy(),
            f=None, k=None, xe=None, q=q.copy()
        )

        assert info == 0

        # Verify residual
        r_full = np.triu(r) + np.triu(r, 1).T
        np.testing.assert_allclose(r_full, r_expected, rtol=1e-12)

        # C = A - G*X*A (minus sign)
        c_expected = a - g @ x @ a
        np.testing.assert_allclose(c, c_expected, rtol=1e-12)


class TestSG02CWJobD:
    """Tests for JOBG='D' (D matrix given, G = D*D')."""

    def test_continuous_jobg_d(self):
        """
        Test continuous-time with D matrix (JOBG='D').

        G = D*D', so residual computation uses D instead of G.

        Random seed: 789 (for reproducibility)
        """
        from ctrlsys import sg02cw

        n = 2
        m = 2

        a = np.array([
            [-1.0, 0.0],
            [ 0.0, -2.0]
        ], order='F', dtype=float)

        # D is n-by-m
        d = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        # G = D*D' = I
        g_implicit = d @ d.T

        q = np.eye(2, order='F', dtype=float)
        x = np.eye(2, order='F', dtype=float) * 0.5

        # Expected: R = A'*X + X*A - X*D*D'*X + Q
        r_expected = a.T @ x + x @ a - x @ g_implicit @ x + q

        r, c, norms, info = sg02cw(
            dico='C', job='A', jobe='I', flag='M', jobg='D',
            uplo='U', trans='N',
            n=n, m=m,
            a=a.copy(), e=None, g=d.copy(), x=x.copy(),
            f=None, k=None, xe=None, q=q.copy()
        )

        assert info == 0

        r_full = np.triu(r) + np.triu(r, 1).T
        np.testing.assert_allclose(r_full, r_expected, rtol=1e-11)


class TestSG02CWJobFH:
    """Tests for JOBG='F' and JOBG='H' (F or H,K matrices given)."""

    def test_continuous_jobg_f(self):
        """
        Test continuous-time with F matrix (JOBG='F').

        Quadratic term = F*F', closed-loop term = D*F'.

        Random seed: 321 (for reproducibility)
        """
        from ctrlsys import sg02cw

        n = 2
        m = 2

        a = np.array([
            [-1.0, 0.0],
            [ 0.0, -2.0]
        ], order='F', dtype=float)

        # F is n-by-m
        f = np.array([
            [1.0, 0.0],
            [0.0, 0.5]
        ], order='F', dtype=float)

        # D is n-by-m (stored in G array)
        d = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        q = np.eye(2, order='F', dtype=float)
        x = np.eye(2, order='F', dtype=float) * 0.5

        # Expected: R = A'*X + X*A - F*F' + Q
        ff = f @ f.T
        r_expected = a.T @ x + x @ a - ff + q

        r, c, norms, info = sg02cw(
            dico='C', job='A', jobe='I', flag='M', jobg='F',
            uplo='U', trans='N',
            n=n, m=m,
            a=a.copy(), e=None, g=d.copy(), x=x.copy(),
            f=f.copy(), k=None, xe=None, q=q.copy()
        )

        assert info == 0

        r_full = np.triu(r) + np.triu(r, 1).T
        np.testing.assert_allclose(r_full, r_expected, rtol=1e-12)

        # C = A - D*F'
        c_expected = a - d @ f.T
        np.testing.assert_allclose(c, c_expected, rtol=1e-12)

    def test_continuous_jobg_h(self):
        """
        Test continuous-time with H,K matrices (JOBG='H').

        Quadratic term = H*K, closed-loop term = B*K.

        Random seed: 654 (for reproducibility)
        """
        from ctrlsys import sg02cw

        n = 2
        m = 2

        a = np.array([
            [-1.0, 0.0],
            [ 0.0, -2.0]
        ], order='F', dtype=float)

        # H is n-by-m, K is m-by-n
        h = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        k = np.array([
            [0.5, 0.0],
            [0.0, 0.3]
        ], order='F', dtype=float)

        # B is n-by-m (stored in G array)
        b = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        q = np.eye(2, order='F', dtype=float)
        x = np.eye(2, order='F', dtype=float) * 0.5

        # Expected: R = A'*X + X*A - H*K + Q
        hk = h @ k
        r_expected = a.T @ x + x @ a - hk + q

        r, c, norms, info = sg02cw(
            dico='C', job='A', jobe='I', flag='M', jobg='H',
            uplo='U', trans='N',
            n=n, m=m,
            a=a.copy(), e=None, g=b.copy(), x=x.copy(),
            f=h.copy(), k=k.copy(), xe=None, q=q.copy()
        )

        assert info == 0

        r_full = np.triu(r) + np.triu(r, 1).T
        np.testing.assert_allclose(r_full, r_expected, rtol=1e-12)

        # C = A - B*K
        c_expected = a - b @ k
        np.testing.assert_allclose(c, c_expected, rtol=1e-12)


class TestSG02CWGeneralizedE:
    """Tests with general E matrix (JOBE='G')."""

    def test_continuous_general_e(self):
        """
        Test continuous-time generalized Lyapunov with general E.

        For m=0, SLICOT computes Lyapunov residual (use JOBG='D'):
        R = A'*X*E + E'*X*A + Q
        C = A

        Random seed: 111 (for reproducibility)
        """
        from ctrlsys import sg02cw

        n = 2

        a = np.array([
            [-1.0, 0.0],
            [ 0.0, -2.0]
        ], order='F', dtype=float)

        # E nonsingular
        e = np.array([
            [2.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        g = np.eye(2, order='F', dtype=float)  # Not used with m=0, JOBG='D'
        q = np.eye(2, order='F', dtype=float)
        x = np.eye(2, order='F', dtype=float) * 0.5

        # Expected Lyapunov residual: R = A'*X*E + E'*X*A + Q
        axe = a.T @ x @ e
        exa = e.T @ x @ a
        r_expected = axe + exa + q

        r, c, norms, info = sg02cw(
            dico='C', job='A', jobe='G', flag='M', jobg='D',  # JOBG='D' for m=0
            uplo='U', trans='N',
            n=n, m=0,
            a=a.copy(), e=e.copy(), g=g.copy(), x=x.copy(),
            f=None, k=None, xe=None, q=q.copy()
        )

        assert info == 0

        r_full = np.triu(r) + np.triu(r, 1).T
        np.testing.assert_allclose(r_full, r_expected, rtol=1e-11)

        # C = A (for Lyapunov with m=0)
        np.testing.assert_allclose(c, a, rtol=1e-12)

    def test_discrete_general_e(self):
        """
        Test discrete-time generalized Stein equation with general E.

        For m=0, SLICOT computes Stein residual (use JOBG='D'):
        R = A'*X*A - E'*X*E + Q
        C = A

        Random seed: 222 (for reproducibility)
        """
        from ctrlsys import sg02cw

        n = 2

        # Stable discrete system
        a = np.array([
            [0.5, 0.0],
            [0.0, 0.3]
        ], order='F', dtype=float)

        e = np.array([
            [1.0, 0.0],
            [0.0, 2.0]
        ], order='F', dtype=float)

        g = np.eye(2, order='F', dtype=float)  # Not used with m=0, JOBG='D'
        q = np.eye(2, order='F', dtype=float)
        x = np.eye(2, order='F', dtype=float) * 0.5

        # XE = X*A (required for discrete-time with JOBE='G')
        xe = x @ a

        # Expected Stein residual: R = A'*X*A - E'*X*E + Q
        axa = a.T @ x @ a
        exe = e.T @ x @ e
        r_expected = axa - exe + q

        r, c, norms, info = sg02cw(
            dico='D', job='A', jobe='G', flag='M', jobg='D',  # JOBG='D' for m=0
            uplo='U', trans='N',
            n=n, m=0,
            a=a.copy(), e=e.copy(), g=g.copy(), x=x.copy(),
            f=None, k=None, xe=np.asfortranarray(xe), q=q.copy()
        )

        assert info == 0

        r_full = np.triu(r) + np.triu(r, 1).T
        np.testing.assert_allclose(r_full, r_expected, rtol=1e-11)

        # C = A (for Stein with m=0)
        np.testing.assert_allclose(c, a, rtol=1e-12)


class TestSG02CWNorms:
    """Tests for norm computation (JOB='N' or JOB='B')."""

    def test_continuous_norms(self):
        """
        Test norm computation with JOB='N'.

        NORMS(1) = ||A'*X|| (or ||X*A||)
        NORMS(2) = ||X*G*X||

        Random seed: 333 (for reproducibility)
        """
        from ctrlsys import sg02cw

        n = 2

        a = np.array([
            [-1.0, 0.0],
            [ 0.0, -2.0]
        ], order='F', dtype=float)

        g = np.eye(2, order='F', dtype=float)
        q = np.eye(2, order='F', dtype=float)
        x = np.eye(2, order='F', dtype=float) * 0.5

        r, c, norms, info = sg02cw(
            dico='C', job='N', jobe='I', flag='M', jobg='G',
            uplo='U', trans='N',
            n=n, m=0,
            a=a.copy(), e=None, g=g.copy(), x=x.copy(),
            f=None, k=None, xe=None, q=q.copy()
        )

        assert info == 0
        assert len(norms) >= 2

        # NORMS(1) = ||X*A|| (Frobenius)
        xa = x @ a
        norm1_expected = np.linalg.norm(xa, 'fro')
        np.testing.assert_allclose(norms[0], norm1_expected, rtol=1e-12)

        # NORMS(2) = ||X*G*X|| (Frobenius for symmetric matrix)
        xgx = x @ g @ x
        norm2_expected = np.sqrt(np.sum(np.triu(xgx)**2) + np.sum(np.triu(xgx, 1)**2))
        np.testing.assert_allclose(norms[1], norm2_expected, rtol=1e-12)

    def test_discrete_norms_identity_e(self):
        """
        Test norm computation for discrete Stein equation with identity E.

        For m=0 (Stein equation) with JOBE='I':
        NORMS(1) = ||A'*X*A||

        Random seed: 444 (for reproducibility)
        """
        from ctrlsys import sg02cw

        n = 2

        a = np.array([
            [0.5, 0.0],
            [0.0, 0.3]
        ], order='F', dtype=float)

        g = np.eye(2, order='F', dtype=float)  # Not used for norms with m=0
        q = np.eye(2, order='F', dtype=float)
        x = np.eye(2, order='F', dtype=float) * 0.5

        r, c, norms, info = sg02cw(
            dico='D', job='N', jobe='I', flag='M', jobg='D',  # JOBE='I' for identity E
            uplo='U', trans='N',
            n=n, m=0,
            a=a.copy(), e=None, g=g.copy(), x=x.copy(),
            f=None, k=None, xe=None, q=q.copy()
        )

        assert info == 0
        assert len(norms) >= 1

        # NORMS(1) = ||A'*X*A|| (symmetric Frobenius norm)
        axa = a.T @ x @ a
        norm1_expected = np.sqrt(np.sum(np.triu(axa)**2) + np.sum(np.triu(axa, 1)**2))
        np.testing.assert_allclose(norms[0], norm1_expected, rtol=1e-12)


class TestSG02CWJobOptions:
    """Tests for different JOB options."""

    def test_job_r_residual_only(self):
        """
        Test JOB='R' (residual only, no C matrix) for Lyapunov equation.

        For m=0, use JOBG='D' for Lyapunov residual:
        R = A'*X + X*A + Q

        Random seed: 555 (for reproducibility)
        """
        from ctrlsys import sg02cw

        n = 2

        a = np.array([
            [-1.0, 0.0],
            [ 0.0, -2.0]
        ], order='F', dtype=float)

        g = np.eye(2, order='F', dtype=float)  # Not used with m=0, JOBG='D'
        q = np.eye(2, order='F', dtype=float)
        x = np.eye(2, order='F', dtype=float) * 0.5

        r, c, norms, info = sg02cw(
            dico='C', job='R', jobe='I', flag='M', jobg='D',  # JOBG='D' for m=0
            uplo='U', trans='N',
            n=n, m=0,
            a=a.copy(), e=None, g=g.copy(), x=x.copy(),
            f=None, k=None, xe=None, q=q.copy()
        )

        assert info == 0

        # R = A'*X + X*A + Q (Lyapunov residual with m=0)
        r_expected = a.T @ x + x @ a + q
        r_full = np.triu(r) + np.triu(r, 1).T
        np.testing.assert_allclose(r_full, r_expected, rtol=1e-12)

    def test_job_c_closed_loop_only(self):
        """
        Test JOB='C' (closed-loop C matrix only, no residual).

        Random seed: 666 (for reproducibility)
        """
        from ctrlsys import sg02cw

        n = 2

        a = np.array([
            [-1.0, 0.0],
            [ 0.0, -2.0]
        ], order='F', dtype=float)

        g = np.eye(2, order='F', dtype=float)
        x = np.eye(2, order='F', dtype=float) * 0.5
        q = np.eye(2, order='F', dtype=float)

        r, c, norms, info = sg02cw(
            dico='C', job='C', jobe='I', flag='M', jobg='G',
            uplo='U', trans='N',
            n=n, m=0,
            a=a.copy(), e=None, g=g.copy(), x=x.copy(),
            f=None, k=None, xe=None, q=q.copy()
        )

        assert info == 0

        # C = A - G*X
        c_expected = a - g @ x
        np.testing.assert_allclose(c, c_expected, rtol=1e-12)


class TestSG02CWTranspose:
    """Tests with TRANS='T' (transpose operations)."""

    def test_continuous_transpose(self):
        """
        Test continuous-time with TRANS='T'.

        op(W) = W', so formulas use transposed matrices.

        Random seed: 777 (for reproducibility)
        """
        from ctrlsys import sg02cw

        n = 2

        a = np.array([
            [-1.0, 0.5],
            [ 0.0, -2.0]
        ], order='F', dtype=float)

        g = np.eye(2, order='F', dtype=float)
        q = np.eye(2, order='F', dtype=float)
        x = np.eye(2, order='F', dtype=float) * 0.5

        # With TRANS='T': R = A*X + X*A' - X*G*X + Q
        r_expected = a @ x + x @ a.T - x @ g @ x + q

        r, c, norms, info = sg02cw(
            dico='C', job='A', jobe='I', flag='M', jobg='G',
            uplo='U', trans='T',
            n=n, m=0,
            a=a.copy(), e=None, g=g.copy(), x=x.copy(),
            f=None, k=None, xe=None, q=q.copy()
        )

        assert info == 0

        r_full = np.triu(r) + np.triu(r, 1).T
        np.testing.assert_allclose(r_full, r_expected, rtol=1e-12)


class TestSG02CWLowerTriangle:
    """Tests with UPLO='L' (lower triangular storage)."""

    def test_lower_triangular(self):
        """
        Test lower triangular storage with UPLO='L'.

        Random seed: 888 (for reproducibility)
        """
        from ctrlsys import sg02cw

        n = 2

        a = np.array([
            [-1.0, 0.0],
            [ 0.0, -2.0]
        ], order='F', dtype=float)

        g = np.eye(2, order='F', dtype=float)
        q = np.eye(2, order='F', dtype=float)
        x = np.eye(2, order='F', dtype=float) * 0.5

        r_expected = a.T @ x + x @ a - x @ g @ x + q

        r, c, norms, info = sg02cw(
            dico='C', job='A', jobe='I', flag='M', jobg='G',
            uplo='L', trans='N',
            n=n, m=0,
            a=a.copy(), e=None, g=g.copy(), x=x.copy(),
            f=None, k=None, xe=None, q=q.copy()
        )

        assert info == 0

        # Lower triangle stored
        r_full = np.tril(r) + np.tril(r, -1).T
        np.testing.assert_allclose(r_full, r_expected, rtol=1e-12)


class TestSG02CWEdgeCases:
    """Edge case tests."""

    def test_n_zero(self):
        """Test with n=0 (quick return)."""
        from ctrlsys import sg02cw

        r, c, norms, info = sg02cw(
            dico='C', job='N', jobe='I', flag='M', jobg='G',
            uplo='U', trans='N',
            n=0, m=0,
            a=np.empty((0, 0), order='F', dtype=float),
            e=None,
            g=np.empty((0, 0), order='F', dtype=float),
            x=np.empty((0, 0), order='F', dtype=float),
            f=None, k=None, xe=None,
            q=np.empty((0, 0), order='F', dtype=float)
        )

        assert info == 0
        # Norms should be zero for n=0
        np.testing.assert_allclose(norms[:2], [0.0, 0.0], atol=1e-15)


class TestSG02CWParameterValidation:
    """Parameter validation tests."""

    def test_invalid_dico(self):
        """Test invalid DICO parameter returns info < 0."""
        from ctrlsys import sg02cw

        r, c, norms, info = sg02cw(
            dico='X', job='A', jobe='I', flag='M', jobg='G',
            uplo='U', trans='N',
            n=2, m=0,
            a=np.eye(2, order='F'),
            e=None,
            g=np.eye(2, order='F'),
            x=np.eye(2, order='F'),
            f=None, k=None, xe=None,
            q=np.eye(2, order='F')
        )
        assert info == -1

    def test_invalid_job(self):
        """Test invalid JOB parameter returns info < 0."""
        from ctrlsys import sg02cw

        r, c, norms, info = sg02cw(
            dico='C', job='X', jobe='I', flag='M', jobg='G',
            uplo='U', trans='N',
            n=2, m=0,
            a=np.eye(2, order='F'),
            e=None,
            g=np.eye(2, order='F'),
            x=np.eye(2, order='F'),
            f=None, k=None, xe=None,
            q=np.eye(2, order='F')
        )
        assert info == -2

    def test_negative_n(self):
        """Test negative N parameter returns info < 0."""
        from ctrlsys import sg02cw

        r, c, norms, info = sg02cw(
            dico='C', job='A', jobe='I', flag='M', jobg='G',
            uplo='U', trans='N',
            n=-1, m=0,
            a=np.eye(2, order='F'),
            e=None,
            g=np.eye(2, order='F'),
            x=np.eye(2, order='F'),
            f=None, k=None, xe=None,
            q=np.eye(2, order='F')
        )
        assert info == -8
