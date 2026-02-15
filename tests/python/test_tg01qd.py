"""
Tests for tg01qd - Three-domain spectral splitting of descriptor system.

TG01QD computes orthogonal transformation matrices Q and Z which reduce
the regular pole pencil A-lambda*E of the descriptor system (A-lambda*E,B,C)
to generalized real Schur form with ordered generalized eigenvalues.

The pair (A,E) is reduced to a three-block form:
    Q'*A*Z = [A1 * *; 0 A2 *; 0 0 A3]
    Q'*E*Z = [E1 * *; 0 E2 *; 0 0 E3]

Block meaning depends on JOBFI and STDOM parameters.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


class TestTg01qdBasic:
    """Test basic functionality from HTML doc example."""

    def test_html_doc_example(self):
        """
        Validate basic functionality using SLICOT HTML doc example.

        N=4, M=2, P=2, DICO='C', STDOM='S', JOBFI='F', ALPHA=-1e-7, TOL=0.0
        Expected: N1=1, N2=2, N3=1, ND=1, NIBLCK=1
        """
        from ctrlsys import tg01qd

        n, m, p = 4, 2, 2
        dico = 'C'
        stdom = 'S'
        jobfi = 'F'
        alpha = -1.0e-7
        tol = 0.0

        a = np.array([
            [-1.0, 0.0, 0.0, 3.0],
            [0.0, 0.0, 1.0, 2.0],
            [1.0, 1.0, 0.0, 4.0],
            [0.0, 0.0, 0.0, 0.0]
        ], order='F', dtype=float)

        e = np.array([
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [3.0, 9.0, 6.0, 3.0],
            [0.0, 0.0, 2.0, 0.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ], order='F', dtype=float)

        c = np.array([
            [-1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, -1.0, 1.0]
        ], order='F', dtype=float)

        result = tg01qd(dico, stdom, jobfi, n, m, p, alpha, a, e, b, c, tol)

        a_out, e_out, b_out, c_out, n1, n2, n3, nd, niblck, iblck, \
            q, z, alphar, alphai, beta, info = result

        assert info == 0, f"tg01qd failed with info={info}"
        assert n1 == 1, f"Expected n1=1, got {n1}"
        assert n2 == 2, f"Expected n2=2, got {n2}"
        assert n3 == 1, f"Expected n3=1, got {n3}"
        assert nd == 1, f"Expected nd=1, got {nd}"

        # Verify orthogonality of Q and Z (platform-independent)
        assert_allclose(q.T @ q, np.eye(n), rtol=1e-13, atol=1e-14)
        assert_allclose(z.T @ z, np.eye(n), rtol=1e-13, atol=1e-14)

        # Verify A is quasi-upper triangular (block Schur form)
        for i in range(2, n):
            for j in range(i - 1):
                assert abs(a_out[i, j]) < 1e-10, f"A not quasi-upper triangular at [{i},{j}]"

        # Verify E is upper triangular
        for i in range(1, n):
            for j in range(i):
                assert abs(e_out[i, j]) < 1e-10, f"E not upper triangular at [{i},{j}]"

        # Verify diagonal absolute values (sign-independent)
        a_diag_expected = np.array([1.6311, 0.4550, 0.0000, 2.2913])
        e_diag_expected = np.array([0.4484, 3.3099, 2.3524, 0.0000])
        assert_allclose(np.abs(np.diag(a_out)), a_diag_expected, rtol=5e-4, atol=1e-4)
        assert_allclose(np.abs(np.diag(e_out)), e_diag_expected, rtol=5e-4, atol=1e-4)

        eig_finite_expected = [-3.6375, 0.1375, 0.0]
        nf = n1 + n2
        eig_computed = []
        for j in range(n):
            if beta[j] != 0:
                eig_computed.append(complex(alphar[j], alphai[j]) / beta[j])
        eig_computed_real = sorted([e.real for e in eig_computed if abs(e.imag) < 1e-6])
        eig_expected_sorted = sorted(eig_finite_expected)
        assert_allclose(eig_computed_real, eig_expected_sorted, rtol=5e-3, atol=1e-4)


class TestTg01qdOrthogonality:
    """Test orthogonality properties of Q and Z matrices."""

    def test_orthogonality_preservation(self):
        """
        Validate Q'*Q = I and Z'*Z = I (orthogonal transformations).

        Random seed: 42 (for reproducibility)
        """
        from ctrlsys import tg01qd

        np.random.seed(42)
        n, m, p = 5, 2, 3

        a = np.random.randn(n, n).astype(float, order='F')
        e = np.random.randn(n, n).astype(float, order='F')
        e = e + 3.0 * np.eye(n, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        result = tg01qd('C', 'S', 'F', n, m, p, 0.0, a, e, b, c, 0.0)
        a_out, e_out, b_out, c_out, n1, n2, n3, nd, niblck, iblck, \
            q, z, alphar, alphai, beta, info = result

        assert info == 0, f"tg01qd failed with info={info}"

        q_orth = q.T @ q
        z_orth = z.T @ z
        assert_allclose(q_orth, np.eye(n), rtol=1e-13, atol=1e-14)
        assert_allclose(z_orth, np.eye(n), rtol=1e-13, atol=1e-14)


class TestTg01qdTransformation:
    """Test transformation properties."""

    def test_transformation_consistency(self):
        """
        Validate Q'*A*Z, Q'*E*Z, Q'*B, C*Z transformations.

        Random seed: 123 (for reproducibility)
        """
        from ctrlsys import tg01qd

        np.random.seed(123)
        n, m, p = 4, 2, 2

        a_orig = np.random.randn(n, n).astype(float, order='F')
        e_orig = np.random.randn(n, n).astype(float, order='F')
        e_orig = e_orig + 3.0 * np.eye(n, order='F')
        b_orig = np.random.randn(n, m).astype(float, order='F')
        c_orig = np.random.randn(p, n).astype(float, order='F')

        a = a_orig.copy(order='F')
        e = e_orig.copy(order='F')
        b = b_orig.copy(order='F')
        c = c_orig.copy(order='F')

        result = tg01qd('C', 'S', 'F', n, m, p, 0.0, a, e, b, c, 0.0)
        a_out, e_out, b_out, c_out, n1, n2, n3, nd, niblck, iblck, \
            q, z, alphar, alphai, beta, info = result

        assert info == 0, f"tg01qd failed with info={info}"

        a_transformed = q.T @ a_orig @ z
        e_transformed = q.T @ e_orig @ z
        b_transformed = q.T @ b_orig
        c_transformed = c_orig @ z

        assert_allclose(a_out, a_transformed, rtol=1e-13, atol=1e-14)
        assert_allclose(e_out, e_transformed, rtol=1e-13, atol=1e-14)
        assert_allclose(b_out, b_transformed, rtol=1e-13, atol=1e-14)
        assert_allclose(c_out, c_transformed, rtol=1e-13, atol=1e-14)


class TestTg01qdModeParameters:
    """Test different mode parameter combinations."""

    def test_discrete_time_stability(self):
        """
        Test DICO='D' (discrete-time), STDOM='S' (inside circle).

        Random seed: 456 (for reproducibility)
        """
        from ctrlsys import tg01qd

        np.random.seed(456)
        n, m, p = 4, 2, 2

        a = 0.5 * np.random.randn(n, n).astype(float, order='F')
        e = np.eye(n, order='F', dtype=float) + 0.1 * np.random.randn(n, n).astype(float, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        result = tg01qd('D', 'S', 'F', n, m, p, 1.0, a, e, b, c, 0.0)
        a_out, e_out, b_out, c_out, n1, n2, n3, nd, niblck, iblck, \
            q, z, alphar, alphai, beta, info = result

        assert info == 0, f"tg01qd failed with info={info}"

        q_orth = q.T @ q
        z_orth = z.T @ z
        assert_allclose(q_orth, np.eye(n), rtol=1e-13, atol=1e-14)
        assert_allclose(z_orth, np.eye(n), rtol=1e-13, atol=1e-14)

    def test_infinite_first_separation(self):
        """
        Test JOBFI='I' (infinite eigenvalues first).

        Random seed: 789 (for reproducibility)
        """
        from ctrlsys import tg01qd

        np.random.seed(789)
        n, m, p = 4, 2, 2

        a = np.random.randn(n, n).astype(float, order='F')
        e = np.random.randn(n, n).astype(float, order='F')
        e[3, :] = 0.0
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        result = tg01qd('C', 'S', 'I', n, m, p, 0.0, a, e, b, c, 0.0)
        a_out, e_out, b_out, c_out, n1, n2, n3, nd, niblck, iblck, \
            q, z, alphar, alphai, beta, info = result

        assert info == 0, f"tg01qd failed with info={info}"

        q_orth = q.T @ q
        z_orth = z.T @ z
        assert_allclose(q_orth, np.eye(n), rtol=1e-13, atol=1e-14)
        assert_allclose(z_orth, np.eye(n), rtol=1e-13, atol=1e-14)

    def test_stdom_unstable(self):
        """
        Test STDOM='U' (unstable eigenvalues first).

        Random seed: 111 (for reproducibility)
        """
        from ctrlsys import tg01qd

        np.random.seed(111)
        n, m, p = 4, 2, 2

        a = np.random.randn(n, n).astype(float, order='F')
        e = np.random.randn(n, n).astype(float, order='F')
        e = e + 3.0 * np.eye(n, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        result = tg01qd('C', 'U', 'F', n, m, p, 0.0, a, e, b, c, 0.0)
        a_out, e_out, b_out, c_out, n1, n2, n3, nd, niblck, iblck, \
            q, z, alphar, alphai, beta, info = result

        assert info == 0, f"tg01qd failed with info={info}"

        q_orth = q.T @ q
        z_orth = z.T @ z
        assert_allclose(q_orth, np.eye(n), rtol=1e-13, atol=1e-14)
        assert_allclose(z_orth, np.eye(n), rtol=1e-13, atol=1e-14)

    def test_stdom_none(self):
        """
        Test STDOM='N' (no spectral ordering of finite part).

        Random seed: 222 (for reproducibility)
        """
        from ctrlsys import tg01qd

        np.random.seed(222)
        n, m, p = 4, 2, 2

        a = np.random.randn(n, n).astype(float, order='F')
        e = np.random.randn(n, n).astype(float, order='F')
        e = e + 3.0 * np.eye(n, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        result = tg01qd('C', 'N', 'F', n, m, p, 0.0, a, e, b, c, 0.0)
        a_out, e_out, b_out, c_out, n1, n2, n3, nd, niblck, iblck, \
            q, z, alphar, alphai, beta, info = result

        assert info == 0, f"tg01qd failed with info={info}"
        assert n2 == 0, f"Expected n2=0 for STDOM='N', got {n2}"


class TestTg01qdEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_dimension(self):
        """Test with N=0 (quick return)."""
        from ctrlsys import tg01qd

        n, m, p = 0, 2, 2

        a = np.array([], order='F', dtype=float).reshape(0, 0)
        e = np.array([], order='F', dtype=float).reshape(0, 0)
        b = np.array([], order='F', dtype=float).reshape(0, m)
        c = np.array([], order='F', dtype=float).reshape(p, 0)

        result = tg01qd('C', 'S', 'F', n, m, p, 0.0, a, e, b, c, 0.0)
        a_out, e_out, b_out, c_out, n1, n2, n3, nd, niblck, iblck, \
            q, z, alphar, alphai, beta, info = result

        assert info == 0
        assert n1 == 0
        assert n2 == 0
        assert n3 == 0
        assert nd == 0
        assert niblck == 0

    def test_invalid_dico(self):
        """Test invalid DICO parameter returns error."""
        from ctrlsys import tg01qd

        n, m, p = 3, 1, 1

        a = np.eye(n, order='F', dtype=float)
        e = np.eye(n, order='F', dtype=float)
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)

        result = tg01qd('X', 'S', 'F', n, m, p, 0.0, a, e, b, c, 0.0)
        info = result[-1]
        assert info == -1, f"Expected info=-1 for invalid DICO, got {info}"

    def test_invalid_stdom(self):
        """Test invalid STDOM parameter returns error."""
        from ctrlsys import tg01qd

        n, m, p = 3, 1, 1

        a = np.eye(n, order='F', dtype=float)
        e = np.eye(n, order='F', dtype=float)
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)

        result = tg01qd('C', 'X', 'F', n, m, p, 0.0, a, e, b, c, 0.0)
        info = result[-1]
        assert info == -2, f"Expected info=-2 for invalid STDOM, got {info}"

    def test_invalid_jobfi(self):
        """Test invalid JOBFI parameter returns error."""
        from ctrlsys import tg01qd

        n, m, p = 3, 1, 1

        a = np.eye(n, order='F', dtype=float)
        e = np.eye(n, order='F', dtype=float)
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)

        result = tg01qd('C', 'S', 'X', n, m, p, 0.0, a, e, b, c, 0.0)
        info = result[-1]
        assert info == -3, f"Expected info=-3 for invalid JOBFI, got {info}"

    def test_invalid_alpha_discrete(self):
        """Test negative ALPHA for discrete-time returns error."""
        from ctrlsys import tg01qd

        n, m, p = 3, 1, 1

        a = np.eye(n, order='F', dtype=float)
        e = np.eye(n, order='F', dtype=float)
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)

        result = tg01qd('D', 'S', 'F', n, m, p, -1.0, a, e, b, c, 0.0)
        info = result[-1]
        assert info == -7, f"Expected info=-7 for negative ALPHA in discrete-time, got {info}"


class TestTg01qdSchurStructure:
    """Test Schur form structure of output."""

    def test_finite_part_schur_structure(self):
        """
        Validate output is in quasi-upper triangular (real Schur) form.

        Random seed: 333 (for reproducibility)
        """
        from ctrlsys import tg01qd

        np.random.seed(333)
        n, m, p = 5, 2, 2

        a = np.random.randn(n, n).astype(float, order='F')
        e = np.random.randn(n, n).astype(float, order='F')
        e = e + 3.0 * np.eye(n, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        result = tg01qd('C', 'S', 'F', n, m, p, 0.0, a, e, b, c, 0.0)
        a_out, e_out, b_out, c_out, n1, n2, n3, nd, niblck, iblck, \
            q, z, alphar, alphai, beta, info = result

        assert info == 0, f"tg01qd failed with info={info}"

        for i in range(n - 1):
            if i + 2 < n:
                assert abs(a_out[i + 2, i]) < 1e-10, \
                    f"A not quasi-upper triangular: A[{i+2},{i}]={a_out[i+2, i]}"

        for j in range(n):
            for i in range(j + 1, n):
                assert abs(e_out[i, j]) < 1e-10, \
                    f"E not upper triangular: E[{i},{j}]={e_out[i, j]}"


class TestTg01qdEigenvaluePreservation:
    """Test eigenvalue preservation property."""

    def test_eigenvalue_preservation(self):
        """
        Validate that eigenvalues are preserved under transformation.

        Computes generalized eigenvalues of (A, E) using inv(E)*A since E
        is made invertible by adding 3*I.

        Random seed: 444 (for reproducibility)
        """
        from ctrlsys import tg01qd

        np.random.seed(444)
        n, m, p = 4, 2, 2

        a = np.random.randn(n, n).astype(float, order='F')
        e = np.random.randn(n, n).astype(float, order='F')
        e = e + 3.0 * np.eye(n, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        eig_orig = np.linalg.eigvals(np.linalg.solve(e, a))
        eig_orig_sorted = np.sort(np.real(eig_orig))

        result = tg01qd('C', 'S', 'F', n, m, p, 0.0, a.copy(order='F'),
                        e.copy(order='F'), b.copy(order='F'), c.copy(order='F'), 0.0)
        a_out, e_out, b_out, c_out, n1, n2, n3, nd, niblck, iblck, \
            q, z, alphar, alphai, beta, info = result

        assert info == 0, f"tg01qd failed with info={info}"

        eig_computed = []
        for j in range(n):
            if abs(beta[j]) > 1e-14:
                eig_computed.append(complex(alphar[j], alphai[j]) / beta[j])
            else:
                eig_computed.append(np.inf)

        eig_finite = [ev for ev in eig_computed if np.isfinite(ev)]
        eig_finite_sorted = np.sort(np.real(eig_finite))

        assert_allclose(eig_finite_sorted, eig_orig_sorted, rtol=1e-10, atol=1e-12)
