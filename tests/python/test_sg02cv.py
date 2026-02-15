"""
Tests for SG02CV: Lyapunov residual matrix computation.

SG02CV computes residual matrix R for continuous/discrete Lyapunov equations:
  Continuous: R = op(A)'*X + X*op(A) + Q  (JOBE='I')
              R = op(A)'*X*op(E) + op(E)'*X*op(A) + Q  (JOBE='G')
  Discrete:   R = op(A)'*X*op(A) - X + Q  (JOBE='I')
              R = op(A)'*X*op(A) - op(E)'*X*op(E) + Q  (JOBE='G')
"""

import numpy as np
import pytest
from ctrlsys import sg02cv


def make_schur_form(n, seed):
    """Create an upper real Schur matrix (upper triangular for simplicity)."""
    np.random.seed(seed)
    a = np.triu(np.random.randn(n, n).astype(float, order='F'))
    return np.asfortranarray(a)


def make_upper_triangular(n, seed):
    """Create an upper triangular matrix."""
    np.random.seed(seed)
    e = np.triu(np.random.randn(n, n).astype(float, order='F'))
    np.fill_diagonal(e, np.abs(np.diag(e)) + 0.5)
    return np.asfortranarray(e)


def make_symmetric(n, seed):
    """Create a symmetric positive definite matrix."""
    np.random.seed(seed)
    m = np.random.randn(n, n)
    s = m @ m.T + np.eye(n)
    return np.asfortranarray(s)


class TestSG02CVContinuousIdentity:
    """Test continuous-time case with JOBE='I' (identity E)."""

    def test_residual_only_notrans(self):
        """
        Test R = A'*X + X*A + Q (TRANS='N', UPLO='U').

        Random seed: 42
        """
        np.random.seed(42)
        n = 3
        a = make_schur_form(n, 42)
        x = make_symmetric(n, 43)
        q = make_symmetric(n, 44)

        r_in = np.triu(q.copy()).astype(float, order='F')
        x_in = np.asfortranarray(x.copy())

        r_out, norms, info = sg02cv('C', 'R', 'I', 'U', 'N', a, None, x_in, r_in)

        assert info == 0

        r_expected_upper = np.triu(a.T @ x + x @ a + q)
        np.testing.assert_allclose(np.triu(r_out), r_expected_upper, rtol=1e-13)

    def test_residual_with_norms_trans(self):
        """
        Test R = A*X + X*A' + Q (TRANS='T', UPLO='U') with norms.

        With TRANS='T', op(A)=A', so norm is ||op(A)'*X|| = ||A*X||_F.
        Random seed: 100
        """
        np.random.seed(100)
        n = 4
        a = make_schur_form(n, 100)
        x = make_symmetric(n, 101)
        q = make_symmetric(n, 102)

        r_in = np.triu(q.copy()).astype(float, order='F')
        x_in = np.asfortranarray(x.copy())

        r_out, norms, info = sg02cv('C', 'N', 'I', 'U', 'T', a, None, x_in, r_in)

        assert info == 0

        r_expected_upper = np.triu(a @ x + x @ a.T + q)
        np.testing.assert_allclose(np.triu(r_out), r_expected_upper, rtol=1e-13)

        ax_norm = np.linalg.norm(a @ x, 'fro')
        np.testing.assert_allclose(norms[0], ax_norm, rtol=1e-13)

    def test_lower_triangular_storage(self):
        """
        Test with UPLO='L' (lower triangular storage).

        Random seed: 200
        """
        np.random.seed(200)
        n = 3
        a = make_schur_form(n, 200)
        x = make_symmetric(n, 201)
        q = make_symmetric(n, 202)

        r_in = np.tril(q.copy()).astype(float, order='F')
        x_in = np.asfortranarray(x.copy())

        r_out, norms, info = sg02cv('C', 'R', 'I', 'L', 'N', a, None, x_in, r_in)

        assert info == 0

        r_expected_lower = np.tril(a.T @ x + x @ a + q)
        np.testing.assert_allclose(np.tril(r_out), r_expected_lower, rtol=1e-13)


class TestSG02CVContinuousGeneral:
    """Test continuous-time case with JOBE='G' (general E)."""

    def test_residual_with_e_notrans(self):
        """
        Test R = A'*X*E + E'*X*A + Q (TRANS='N').

        Random seed: 300
        """
        np.random.seed(300)
        n = 3
        a = make_schur_form(n, 300)
        e = make_upper_triangular(n, 301)
        x = make_symmetric(n, 302)
        q = make_symmetric(n, 303)

        r_in = np.triu(q.copy()).astype(float, order='F')
        x_in = np.asfortranarray(x.copy())

        r_out, norms, info = sg02cv('C', 'R', 'G', 'U', 'N', a, e, x_in, r_in)

        assert info == 0

        r_expected_upper = np.triu(a.T @ x @ e + e.T @ x @ a + q)
        np.testing.assert_allclose(np.triu(r_out), r_expected_upper, rtol=1e-13)

    def test_residual_with_e_trans_norms(self):
        """
        Test R = A*X*E' + E*X*A' + Q (TRANS='T') with norms.

        Random seed: 400
        """
        np.random.seed(400)
        n = 4
        a = make_schur_form(n, 400)
        e = make_upper_triangular(n, 401)
        x = make_symmetric(n, 402)
        q = make_symmetric(n, 403)

        r_in = np.triu(q.copy()).astype(float, order='F')
        x_in = np.asfortranarray(x.copy())

        r_out, norms, info = sg02cv('C', 'N', 'G', 'U', 'T', a, e, x_in, r_in)

        assert info == 0

        r_expected_upper = np.triu(a @ x @ e.T + e @ x @ a.T + q)
        np.testing.assert_allclose(np.triu(r_out), r_expected_upper, rtol=1e-13)

        axe_t = a @ x @ e.T
        norm_expected = np.linalg.norm(axe_t, 'fro')
        np.testing.assert_allclose(norms[0], norm_expected, rtol=1e-13)


class TestSG02CVDiscreteIdentity:
    """Test discrete-time case with JOBE='I' (identity E)."""

    def test_residual_notrans(self):
        """
        Test R = A'*X*A - X + Q (TRANS='N').

        Random seed: 500
        """
        np.random.seed(500)
        n = 3
        a = make_schur_form(n, 500)
        a = 0.5 * a / (np.linalg.norm(a) + 1)
        a = np.asfortranarray(a)
        x = make_symmetric(n, 501)
        q = make_symmetric(n, 502)

        r_in = np.triu(q.copy()).astype(float, order='F')
        x_in = np.asfortranarray(x.copy())

        r_out, norms, info = sg02cv('D', 'R', 'I', 'U', 'N', a, None, x_in, r_in)

        assert info == 0

        r_expected_upper = np.triu(a.T @ x @ a - x + q)
        np.testing.assert_allclose(np.triu(r_out), r_expected_upper, rtol=1e-13)

    def test_residual_trans_norms(self):
        """
        Test R = A*X*A' - X + Q (TRANS='T') with norms.

        Random seed: 600
        """
        np.random.seed(600)
        n = 4
        a = make_schur_form(n, 600)
        a = 0.5 * a / (np.linalg.norm(a) + 1)
        a = np.asfortranarray(a)
        x = make_symmetric(n, 601)
        q = make_symmetric(n, 602)

        r_in = np.triu(q.copy()).astype(float, order='F')
        x_in = np.asfortranarray(x.copy())

        r_out, norms, info = sg02cv('D', 'N', 'I', 'U', 'T', a, None, x_in, r_in)

        assert info == 0

        r_expected_upper = np.triu(a @ x @ a.T - x + q)
        np.testing.assert_allclose(np.triu(r_out), r_expected_upper, rtol=1e-13)

        axa_t = a @ x @ a.T
        norm_expected = np.linalg.norm(np.triu(axa_t) + np.triu(axa_t, 1).T, 'fro')
        np.testing.assert_allclose(norms[0], norm_expected, rtol=1e-13)


class TestSG02CVDiscreteGeneral:
    """Test discrete-time case with JOBE='G' (general E)."""

    def test_residual_notrans(self):
        """
        Test R = A'*X*A - E'*X*E + Q (TRANS='N').

        Random seed: 700
        """
        np.random.seed(700)
        n = 3
        a = make_schur_form(n, 700)
        e = make_upper_triangular(n, 701)
        x = make_symmetric(n, 702)
        q = make_symmetric(n, 703)

        r_in = np.triu(q.copy()).astype(float, order='F')
        x_in = np.asfortranarray(x.copy())

        r_out, norms, info = sg02cv('D', 'R', 'G', 'U', 'N', a, e, x_in, r_in)

        assert info == 0

        r_expected_upper = np.triu(a.T @ x @ a - e.T @ x @ e + q)
        np.testing.assert_allclose(np.triu(r_out), r_expected_upper, rtol=1e-13)

    def test_residual_trans_norms(self):
        """
        Test R = A*X*A' - E*X*E' + Q (TRANS='T') with norms.

        Random seed: 800
        """
        np.random.seed(800)
        n = 4
        a = make_schur_form(n, 800)
        e = make_upper_triangular(n, 801)
        x = make_symmetric(n, 802)
        q = make_symmetric(n, 803)

        r_in = np.triu(q.copy()).astype(float, order='F')
        x_in = np.asfortranarray(x.copy())

        r_out, norms, info = sg02cv('D', 'N', 'G', 'U', 'T', a, e, x_in, r_in)

        assert info == 0

        r_expected_upper = np.triu(a @ x @ a.T - e @ x @ e.T + q)
        np.testing.assert_allclose(np.triu(r_out), r_expected_upper, rtol=1e-13)

        axa_t = a @ x @ a.T
        exe_t = e @ x @ e.T
        norm1_expected = np.linalg.norm(np.triu(axa_t) + np.triu(axa_t, 1).T, 'fro')
        norm2_expected = np.linalg.norm(np.triu(exe_t) + np.triu(exe_t, 1).T, 'fro')
        np.testing.assert_allclose(norms[0], norm1_expected, rtol=1e-13)
        np.testing.assert_allclose(norms[1], norm2_expected, rtol=1e-13)


class TestSG02CVEdgeCases:
    """Edge cases and error handling."""

    def test_n_zero(self):
        """Test quick return for n=0."""
        a = np.array([], dtype=float, order='F').reshape(0, 0)
        x = np.array([], dtype=float, order='F').reshape(0, 0)
        r = np.array([], dtype=float, order='F').reshape(0, 0)

        r_out, norms, info = sg02cv('C', 'N', 'I', 'U', 'N', a, None, x, r)

        assert info == 0
        assert norms[0] == 0.0

    def test_n_one(self):
        """
        Test n=1 case.

        Random seed: 900
        """
        np.random.seed(900)
        n = 1
        a = np.array([[2.0]], order='F', dtype=float)
        x = np.array([[3.0]], order='F', dtype=float)
        q = np.array([[1.0]], order='F', dtype=float)

        r_in = q.copy()

        r_out, norms, info = sg02cv('C', 'R', 'I', 'U', 'N', a, None, x.copy(), r_in)

        assert info == 0
        r_expected = a.T @ x + x @ a + q
        np.testing.assert_allclose(r_out, r_expected, rtol=1e-14)

    def test_invalid_dico(self):
        """Test invalid DICO parameter."""
        a = np.array([[1.0]], order='F', dtype=float)
        x = np.array([[1.0]], order='F', dtype=float)
        r = np.array([[1.0]], order='F', dtype=float)

        r_out, norms, info = sg02cv('X', 'R', 'I', 'U', 'N', a, None, x, r)

        assert info == -1

    def test_invalid_job(self):
        """Test invalid JOB parameter."""
        a = np.array([[1.0]], order='F', dtype=float)
        x = np.array([[1.0]], order='F', dtype=float)
        r = np.array([[1.0]], order='F', dtype=float)

        r_out, norms, info = sg02cv('C', 'X', 'I', 'U', 'N', a, None, x, r)

        assert info == -2


class TestSG02CVMathProperties:
    """Mathematical property validation."""

    def test_residual_zero_for_solution(self):
        """
        Validate that if X solves A'X + XA + Q = 0, then residual R = 0.

        Uses stable A and computes Q = -(A'X + XA) for given X.
        Random seed: 1000
        """
        np.random.seed(1000)
        n = 3
        a = make_schur_form(n, 1000)
        a = a - 2 * np.eye(n)
        a = np.asfortranarray(a)

        x = make_symmetric(n, 1001)

        q = -(a.T @ x + x @ a)
        q = np.asfortranarray(q)

        r_in = np.triu(q.copy()).astype(float, order='F')
        x_in = np.asfortranarray(x.copy())

        r_out, norms, info = sg02cv('C', 'R', 'I', 'U', 'N', a, None, x_in, r_in)

        assert info == 0
        np.testing.assert_allclose(np.triu(r_out), np.zeros((n, n)), atol=1e-14)

    def test_residual_symmetry_discrete(self):
        """
        Validate residual is symmetric for discrete Lyapunov.

        R = A'XA - X + Q should be symmetric when X and Q are symmetric.
        Random seed: 1100
        """
        np.random.seed(1100)
        n = 4
        a = make_schur_form(n, 1100)
        a = 0.5 * a / (np.linalg.norm(a) + 1)
        a = np.asfortranarray(a)
        x = make_symmetric(n, 1101)
        q = make_symmetric(n, 1102)

        r_in = np.triu(q.copy()).astype(float, order='F')
        x_in = np.asfortranarray(x.copy())

        r_out, norms, info = sg02cv('D', 'R', 'I', 'U', 'N', a, None, x_in, r_in)

        assert info == 0

        r_full = np.triu(r_out) + np.triu(r_out, 1).T
        np.testing.assert_allclose(r_full, r_full.T, rtol=1e-14)
