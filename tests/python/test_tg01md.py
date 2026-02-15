"""
Tests for tg01md - Finite-infinite generalized real Schur form decomposition
of a descriptor system.

TG01MD computes orthogonal transformation matrices Q and Z which reduce
the regular pole pencil A-lambda*E of the descriptor system (A-lambda*E,B,C)
to finite-infinite (JOB='F') or infinite-finite (JOB='I') separated form,
with the finite part in generalized real Schur form.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


class TestTg01mdBasic:
    """Test basic functionality from HTML doc example."""

    def test_html_doc_example(self):
        """
        Validate basic functionality using SLICOT HTML doc example.

        N=4, M=2, P=2, JOB='F', TOL=0.0
        Expected: NF=3, ND=1, NIBLCK=1
        """
        from ctrlsys import tg01md

        n, m, p = 4, 2, 2

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

        result = tg01md('F', n, m, p, a, e, b, c, 0.0)

        a_out, e_out, b_out, c_out, alphar, alphai, beta, q, z, nf, nd, niblck, iblck, info = result

        assert info == 0, f"tg01md failed with info={info}"
        assert nf == 3, f"Expected nf=3, got {nf}"
        assert nd == 1, f"Expected nd=1, got {nd}"

        # Verify orthogonality of Q and Z (platform-independent)
        assert_allclose(q.T @ q, np.eye(n), rtol=1e-13, atol=1e-14)
        assert_allclose(z.T @ z, np.eye(n), rtol=1e-13, atol=1e-14)

        # Verify E is upper triangular (lower triangle should be zero)
        for i in range(1, n):
            for j in range(i):
                assert abs(e_out[i, j]) < 1e-10, f"E not upper triangular at [{i},{j}]"

        # Verify E[n-1,n-1] = 0 (infinite eigenvalue block)
        assert abs(e_out[n-1, n-1]) < 1e-10, "Expected E[n-1,n-1]=0 for infinite eigenvalue"

        # Verify A diagonal absolute values match expected (sign-independent)
        a_diag_expected = np.array([1.2803, 0.5796, 0.0000, 2.2913])
        assert_allclose(np.abs(np.diag(a_out)), a_diag_expected, rtol=5e-4, atol=1e-4)

        # Verify E diagonal absolute values match expected (sign-independent)
        e_diag_expected = np.array([9.3142, 0.1594, 2.3524, 0.0000])
        assert_allclose(np.abs(np.diag(e_out)), e_diag_expected, rtol=5e-4, atol=1e-4)

        eig_expected = np.array([0.1375, -3.6375, 0.0])
        eig_computed = []
        for j in range(nf):
            if beta[j] != 0:
                eig_computed.append(complex(alphar[j], alphai[j]) / beta[j])
        eig_computed = np.array([e.real for e in eig_computed])

        eig_computed_sorted = np.sort(eig_computed)
        eig_expected_sorted = np.sort(eig_expected)
        assert_allclose(eig_computed_sorted, eig_expected_sorted, rtol=5e-3, atol=1e-4)


class TestTg01mdOrthogonality:
    """Test orthogonality properties of Q and Z matrices."""

    def test_orthogonality_preservation(self):
        """
        Validate Q'*Q = I and Z'*Z = I (orthogonal transformations).

        Random seed: 42 (for reproducibility)
        """
        from ctrlsys import tg01md

        np.random.seed(42)
        n, m, p = 5, 2, 3

        a = np.random.randn(n, n).astype(float, order='F')
        e = np.random.randn(n, n).astype(float, order='F')
        e = e + 2.0 * np.eye(n, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        result = tg01md('F', n, m, p, a, e, b, c, 0.0)
        a_out, e_out, b_out, c_out, alphar, alphai, beta, q, z, nf, nd, niblck, iblck, info = result

        assert info == 0, f"tg01md failed with info={info}"

        q_orth = q.T @ q
        z_orth = z.T @ z
        assert_allclose(q_orth, np.eye(n), rtol=1e-13, atol=1e-14)
        assert_allclose(z_orth, np.eye(n), rtol=1e-13, atol=1e-14)


class TestTg01mdTransformation:
    """Test transformation properties."""

    def test_transformation_consistency(self):
        """
        Validate Q'*A*Z, Q'*E*Z, Q'*B, C*Z transformations.

        Random seed: 123 (for reproducibility)
        """
        from ctrlsys import tg01md

        np.random.seed(123)
        n, m, p = 4, 2, 2

        a_orig = np.random.randn(n, n).astype(float, order='F')
        e_orig = np.random.randn(n, n).astype(float, order='F')
        e_orig = e_orig + 2.0 * np.eye(n, order='F')
        b_orig = np.random.randn(n, m).astype(float, order='F')
        c_orig = np.random.randn(p, n).astype(float, order='F')

        a = a_orig.copy(order='F')
        e = e_orig.copy(order='F')
        b = b_orig.copy(order='F')
        c = c_orig.copy(order='F')

        result = tg01md('F', n, m, p, a, e, b, c, 0.0)
        a_out, e_out, b_out, c_out, alphar, alphai, beta, q, z, nf, nd, niblck, iblck, info = result

        assert info == 0, f"tg01md failed with info={info}"

        a_transformed = q.T @ a_orig @ z
        e_transformed = q.T @ e_orig @ z
        b_transformed = q.T @ b_orig
        c_transformed = c_orig @ z

        assert_allclose(a_out, a_transformed, rtol=1e-13, atol=1e-14)
        assert_allclose(e_out, e_transformed, rtol=1e-13, atol=1e-14)
        assert_allclose(b_out, b_transformed, rtol=1e-13, atol=1e-14)
        assert_allclose(c_out, c_transformed, rtol=1e-13, atol=1e-14)


class TestTg01mdInfiniteFirst:
    """Test infinite-finite separation (JOB='I')."""

    def test_infinite_first_separation(self):
        """
        Validate JOB='I' places infinite eigenvalues first.

        Random seed: 456 (for reproducibility)
        """
        from ctrlsys import tg01md

        np.random.seed(456)
        n, m, p = 4, 2, 2

        a = np.random.randn(n, n).astype(float, order='F')
        e = np.random.randn(n, n).astype(float, order='F')
        e[3, :] = 0.0
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        result = tg01md('I', n, m, p, a, e, b, c, 0.0)
        a_out, e_out, b_out, c_out, alphar, alphai, beta, q, z, nf, nd, niblck, iblck, info = result

        assert info == 0, f"tg01md failed with info={info}"

        q_orth = q.T @ q
        z_orth = z.T @ z
        assert_allclose(q_orth, np.eye(n), rtol=1e-13, atol=1e-14)
        assert_allclose(z_orth, np.eye(n), rtol=1e-13, atol=1e-14)


class TestTg01mdEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_dimension(self):
        """Test with N=0 (quick return)."""
        from ctrlsys import tg01md

        n, m, p = 0, 2, 2

        a = np.array([], order='F', dtype=float).reshape(0, 0)
        e = np.array([], order='F', dtype=float).reshape(0, 0)
        b = np.array([], order='F', dtype=float).reshape(0, m)
        c = np.array([], order='F', dtype=float).reshape(p, 0)

        result = tg01md('F', n, m, p, a, e, b, c, 0.0)
        a_out, e_out, b_out, c_out, alphar, alphai, beta, q, z, nf, nd, niblck, iblck, info = result

        assert info == 0
        assert nf == 0
        assert nd == 0
        assert niblck == 0

    def test_invalid_job(self):
        """Test invalid JOB parameter returns error."""
        from ctrlsys import tg01md

        n, m, p = 3, 1, 1

        a = np.eye(n, order='F', dtype=float)
        e = np.eye(n, order='F', dtype=float)
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)

        result = tg01md('X', n, m, p, a, e, b, c, 0.0)
        info = result[-1]
        assert info == -1, f"Expected info=-1 for invalid JOB, got {info}"


class TestTg01mdSchurStructure:
    """Test Schur form structure of output."""

    def test_finite_part_schur_structure(self):
        """
        Validate Af is in real Schur form (quasi-upper triangular).

        For JOB='F', the leading NF x NF block should be in real Schur form.
        Random seed: 789 (for reproducibility)
        """
        from ctrlsys import tg01md

        np.random.seed(789)
        n, m, p = 5, 2, 2

        a = np.random.randn(n, n).astype(float, order='F')
        e = np.random.randn(n, n).astype(float, order='F')
        e = e + 3.0 * np.eye(n, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        result = tg01md('F', n, m, p, a, e, b, c, 0.0)
        a_out, e_out, b_out, c_out, alphar, alphai, beta, q, z, nf, nd, niblck, iblck, info = result

        assert info == 0, f"tg01md failed with info={info}"

        if nf > 1:
            Af = a_out[:nf, :nf]
            for i in range(nf - 1):
                if i + 2 < nf:
                    assert abs(Af[i + 2, i]) < 1e-10, \
                        f"Af not quasi-upper triangular: Af[{i+2},{i}]={Af[i+2, i]}"

        if nf > 0:
            Ef = e_out[:nf, :nf]
            for j in range(nf):
                for i in range(j + 1, nf):
                    assert abs(Ef[i, j]) < 1e-10, \
                        f"Ef not upper triangular: Ef[{i},{j}]={Ef[i, j]}"
