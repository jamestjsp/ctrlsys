"""
Tests for MB04VX: Separate epsilon and infinite pencils from staircase form.

MB04VX separates s*E(eps)-A(eps) and s*E(inf)-A(inf) from s*E(eps,inf)-A(eps,inf).
This is similar to MB04TX but with NBLCKS as input-only and MNEI having 3 elements.
"""

import numpy as np
import pytest
from ctrlsys import mb04vx


class TestMB04VXBasic:
    """Basic functionality tests for MB04VX."""

    def test_simple_2x3_pencil(self):
        """
        Test with simple 2x3 pencil with 2 blocks.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)

        m, n = 3, 4
        nblcks = 2

        # Create staircase form pencil (already processed by MB04UD/MB04TX-like)
        # Block structure: nu = [2, 1], mu = [2, 2]
        inuk = np.array([2, 1], dtype=np.int32)
        imuk = np.array([2, 2], dtype=np.int32)

        # A matrix in staircase form
        a = np.array([
            [1.0, 2.0, 0.5, 0.3],
            [0.0, 3.0, 1.0, 0.2],
            [0.0, 0.0, 2.0, 0.1]
        ], order='F', dtype=float)

        # E matrix in staircase form (upper triangular structure in blocks)
        e = np.array([
            [1.0, 0.5, 0.3, 0.2],
            [0.0, 1.0, 0.4, 0.1],
            [0.0, 0.0, 1.0, 0.5]
        ], order='F', dtype=float)

        # Transformation matrices
        q = np.eye(m, order='F', dtype=float)
        z = np.eye(n, order='F', dtype=float)

        # Call routine
        a_out, e_out, q_out, z_out, inuk_out, imuk_out, mnei = mb04vx(
            a, e, q, z, nblcks, inuk, imuk, updatq=True, updatz=True
        )

        # Validate MNEI output dimensions
        assert len(mnei) == 3
        meps, neps, minf = mnei[0], mnei[1], mnei[2]

        # Dimensions should be consistent
        assert meps >= 0
        assert neps >= 0
        assert minf >= 0

        # After separation, the sum of inuk should equal meps
        # and sum of imuk should equal neps
        assert np.sum(inuk_out) == meps
        assert np.sum(imuk_out) == neps

    def test_identity_transformations(self):
        """
        Test that Q and Z are orthogonal when updated.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)

        m, n = 4, 5
        nblcks = 2

        inuk = np.array([2, 2], dtype=np.int32)
        imuk = np.array([3, 2], dtype=np.int32)

        # Create matrices
        a = np.random.randn(m, n).astype(float, order='F')
        e = np.random.randn(m, n).astype(float, order='F')
        q = np.eye(m, order='F', dtype=float)
        z = np.eye(n, order='F', dtype=float)

        a_out, e_out, q_out, z_out, inuk_out, imuk_out, mnei = mb04vx(
            a, e, q, z, nblcks, inuk, imuk, updatq=True, updatz=True
        )

        # Q should remain orthogonal
        qtq = q_out.T @ q_out
        np.testing.assert_allclose(qtq, np.eye(m), rtol=1e-14, atol=1e-14)

        # Z should remain orthogonal
        ztz = z_out.T @ z_out
        np.testing.assert_allclose(ztz, np.eye(n), rtol=1e-14, atol=1e-14)

    def test_no_update_q_z(self):
        """
        Test with UPDATQ=False and UPDATZ=False.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)

        m, n = 3, 4
        nblcks = 2

        inuk = np.array([2, 1], dtype=np.int32)
        imuk = np.array([2, 2], dtype=np.int32)

        a = np.random.randn(m, n).astype(float, order='F')
        e = np.random.randn(m, n).astype(float, order='F')

        # Dummy Q and Z (1x1)
        q = np.array([[1.0]], order='F', dtype=float)
        z = np.array([[1.0]], order='F', dtype=float)

        # Call without updating Q/Z
        a_out, e_out, q_out, z_out, inuk_out, imuk_out, mnei = mb04vx(
            a, e, q, z, nblcks, inuk, imuk, updatq=False, updatz=False
        )

        # Q and Z should be None when not updated
        assert q_out is None
        assert z_out is None

        # MNEI should still be valid
        assert len(mnei) == 3


class TestMB04VXEdgeCases:
    """Edge case tests for MB04VX."""

    def test_zero_dimensions(self):
        """Test with M=0 or N=0."""
        m, n = 0, 0
        nblcks = 0

        inuk = np.array([], dtype=np.int32)
        imuk = np.array([], dtype=np.int32)

        a = np.array([[]], order='F', dtype=float).reshape(0, 0)
        e = np.array([[]], order='F', dtype=float).reshape(0, 0)
        q = np.array([[1.0]], order='F', dtype=float)
        z = np.array([[1.0]], order='F', dtype=float)

        a_out, e_out, q_out, z_out, inuk_out, imuk_out, mnei = mb04vx(
            a, e, q, z, nblcks, inuk, imuk, updatq=False, updatz=False
        )

        # MNEI should be all zeros for empty input
        assert mnei[0] == 0
        assert mnei[1] == 0
        assert mnei[2] == 0

    def test_single_block(self):
        """
        Test with single block (nblcks=1).

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)

        m, n = 2, 3
        nblcks = 1

        inuk = np.array([2], dtype=np.int32)
        imuk = np.array([3], dtype=np.int32)

        a = np.random.randn(m, n).astype(float, order='F')
        e = np.random.randn(m, n).astype(float, order='F')
        q = np.eye(m, order='F', dtype=float)
        z = np.eye(n, order='F', dtype=float)

        a_out, e_out, q_out, z_out, inuk_out, imuk_out, mnei = mb04vx(
            a, e, q, z, nblcks, inuk, imuk, updatq=True, updatz=True
        )

        # Results should be valid
        assert len(mnei) == 3
        assert mnei[0] >= 0
        assert mnei[1] >= 0
        assert mnei[2] >= 0


class TestMB04VXNumerical:
    """Numerical property tests for MB04VX."""

    def test_transformation_consistency(self):
        """
        Test that Q' * A_orig * Z = A_out relationship holds.

        Random seed: 999 (for reproducibility)
        """
        np.random.seed(999)

        m, n = 4, 5
        nblcks = 2

        inuk = np.array([2, 2], dtype=np.int32)
        imuk = np.array([3, 2], dtype=np.int32)

        a_orig = np.random.randn(m, n).astype(float, order='F')
        e_orig = np.random.randn(m, n).astype(float, order='F')
        q = np.eye(m, order='F', dtype=float)
        z = np.eye(n, order='F', dtype=float)

        a_out, e_out, q_out, z_out, inuk_out, imuk_out, mnei = mb04vx(
            a_orig.copy(order='F'), e_orig.copy(order='F'),
            q, z, nblcks, inuk.copy(), imuk.copy(),
            updatq=True, updatz=True
        )

        # Q' * A_orig * Z should equal A_out
        a_transformed = q_out.T @ a_orig @ z_out
        np.testing.assert_allclose(a_transformed, a_out, rtol=1e-13, atol=1e-14)

        # Q' * E_orig * Z should equal E_out
        e_transformed = q_out.T @ e_orig @ z_out
        np.testing.assert_allclose(e_transformed, e_out, rtol=1e-13, atol=1e-14)

    def test_determinant_preservation(self):
        """
        Test that det(Q) and det(Z) have magnitude 1 (orthogonal).

        Random seed: 888 (for reproducibility)
        """
        np.random.seed(888)

        m, n = 4, 5
        nblcks = 2

        inuk = np.array([2, 2], dtype=np.int32)
        imuk = np.array([3, 2], dtype=np.int32)

        a = np.random.randn(m, n).astype(float, order='F')
        e = np.random.randn(m, n).astype(float, order='F')
        q = np.eye(m, order='F', dtype=float)
        z = np.eye(n, order='F', dtype=float)

        a_out, e_out, q_out, z_out, inuk_out, imuk_out, mnei = mb04vx(
            a, e, q, z, nblcks, inuk, imuk, updatq=True, updatz=True
        )

        # |det(Q)| = 1 for orthogonal matrix
        det_q = np.linalg.det(q_out)
        np.testing.assert_allclose(abs(det_q), 1.0, rtol=1e-14)

        # |det(Z)| = 1 for orthogonal matrix
        det_z = np.linalg.det(z_out)
        np.testing.assert_allclose(abs(det_z), 1.0, rtol=1e-14)
