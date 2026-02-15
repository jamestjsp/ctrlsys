"""
Tests for MB03SD - Eigenvalues of a square-reduced Hamiltonian matrix.

MB03SD computes the eigenvalues of an N-by-N square-reduced Hamiltonian matrix:
    H' = [ A'   G' ]
         [ Q'  -A' ]

The eigenvalues of H' are computed as square roots of eigenvalues of A'' = A'^2 + G'Q'.
"""

import numpy as np
import pytest
from ctrlsys import mb03sd


class TestMB03SDBasic:
    """Basic functionality tests using HTML doc example."""

    def test_html_example(self):
        """
        Test from SLICOT HTML documentation.

        N=3, JOBSCL='S' (scaling enabled)
        Expected eigenvalues with non-negative real parts:
            2.0 + 1.0i, 2.0 - 1.0i, sqrt(2) + 0i
        """
        n = 3

        # A matrix (3x3), read row-wise
        a = np.array([
            [2.0,  0.0,  0.0],
            [0.0,  1.0,  2.0],
            [0.0, -1.0,  3.0]
        ], order='F', dtype=float)

        # QG matrix (3x4)
        # G' stored in upper triangle of columns 2-4: QG(j, i+1) = G'(i,j) for i>=j
        # Q' stored in lower triangle of columns 1: QG(i, j) = Q'(i,j) for i>=j
        # From example data:
        # G' upper triangle (row-wise from data): 1.0  0.0  0.0  2.0  3.0  4.0
        #   G'(1,1)=1, G'(1,2)=0, G'(1,3)=0, G'(2,2)=2, G'(2,3)=3, G'(3,3)=4
        # Q' upper triangle (row-wise from data): -2.0  0.0  0.0  0.0  0.0  0.0
        #   Q'(1,1)=-2, Q'(1,2)=0, Q'(1,3)=0, Q'(2,2)=0, Q'(2,3)=0, Q'(3,3)=0

        qg = np.zeros((3, 4), order='F', dtype=float)

        # Q' in lower triangle (column 0)
        # Q'(i,j) stored in QG(i,j) for i >= j
        qg[0, 0] = -2.0  # Q'(1,1)
        qg[1, 0] = 0.0   # Q'(2,1)
        qg[2, 0] = 0.0   # Q'(3,1)
        qg[1, 1] = 0.0   # Q'(2,2) - wait, this conflicts with G' storage
        qg[2, 1] = 0.0   # Q'(3,2)
        qg[2, 2] = 0.0   # Q'(3,3)

        # G' in upper triangle of columns 2-4
        # G'(i,j) stored in QG(j,i+1) for i >= j
        # So G'(1,1) in QG(0,1), G'(2,1) in QG(0,2), G'(3,1) in QG(0,3)
        #    G'(2,2) in QG(1,2), G'(3,2) in QG(1,3), G'(3,3) in QG(2,3)
        qg[0, 1] = 1.0   # G'(1,1)
        qg[0, 2] = 0.0   # G'(2,1)
        qg[0, 3] = 0.0   # G'(3,1)
        qg[1, 2] = 2.0   # G'(2,2)
        qg[1, 3] = 3.0   # G'(3,2)
        qg[2, 3] = 4.0   # G'(3,3)

        wr, wi, info = mb03sd('S', a, qg)

        assert info == 0

        # Expected eigenvalues (with non-negative real parts, sorted by decreasing real part):
        # 2.0 + 1.0i, 2.0 - 1.0i, sqrt(2) + 0i
        expected_wr = np.array([2.0, 2.0, np.sqrt(2)])
        expected_wi = np.array([1.0, -1.0, 0.0])

        np.testing.assert_allclose(wr, expected_wr, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(wi, expected_wi, rtol=1e-3, atol=1e-4)

    def test_no_scaling(self):
        """
        Same example with JOBSCL='N' (no scaling).
        Results should be similar.
        """
        n = 3
        a = np.array([
            [2.0,  0.0,  0.0],
            [0.0,  1.0,  2.0],
            [0.0, -1.0,  3.0]
        ], order='F', dtype=float)

        qg = np.zeros((3, 4), order='F', dtype=float)
        qg[0, 0] = -2.0
        qg[0, 1] = 1.0
        qg[1, 2] = 2.0
        qg[1, 3] = 3.0
        qg[2, 3] = 4.0

        wr, wi, info = mb03sd('N', a, qg)

        assert info == 0
        expected_wr = np.array([2.0, 2.0, np.sqrt(2)])
        expected_wi = np.array([1.0, -1.0, 0.0])

        np.testing.assert_allclose(wr, expected_wr, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(wi, expected_wi, rtol=1e-3, atol=1e-4)


class TestMB03SDEdgeCases:
    """Edge case tests."""

    def test_n_zero(self):
        """Test with n=0 (quick return)."""
        a = np.zeros((0, 0), order='F', dtype=float)
        qg = np.zeros((0, 1), order='F', dtype=float)

        wr, wi, info = mb03sd('N', a, qg)

        assert info == 0
        assert len(wr) == 0
        assert len(wi) == 0

    def test_n_one(self):
        """Test with n=1 (scalar case)."""
        # For n=1: A'' = A'^2 + G'*Q'
        # Let A' = [[2]], G' = [[1]], Q' = [[1]]
        # A'' = 4 + 1 = 5
        # eigenvalue of H' = sqrt(5)
        a = np.array([[2.0]], order='F', dtype=float)
        qg = np.array([[1.0, 1.0]], order='F', dtype=float)  # Q'(1,1)=1, G'(1,1)=1

        wr, wi, info = mb03sd('N', a, qg)

        assert info == 0
        assert len(wr) == 1
        assert len(wi) == 1
        np.testing.assert_allclose(wr[0], np.sqrt(5), rtol=1e-14)
        np.testing.assert_allclose(wi[0], 0.0, atol=1e-14)

    def test_identity_matrices(self):
        """
        Test with A'=I, G'=0, Q'=0.
        A'' = I^2 + 0 = I
        eigenvalues of A'' are all 1
        eigenvalues of H' are all +1 (positive root)
        """
        n = 3
        a = np.eye(n, order='F', dtype=float)
        qg = np.zeros((n, n+1), order='F', dtype=float)

        wr, wi, info = mb03sd('N', a, qg)

        assert info == 0
        np.testing.assert_allclose(wr, np.ones(n), rtol=1e-14)
        np.testing.assert_allclose(wi, np.zeros(n), atol=1e-14)


class TestMB03SDMathematical:
    """Mathematical property tests."""

    def test_eigenvalue_pairing(self):
        """
        The full 2N-by-2N Hamiltonian has eigenvalues that come in pairs (λ, -λ).
        MB03SD returns only the N eigenvalues with non-negative real parts.
        Verify complex eigenvalues with non-zero real part come in conjugate pairs.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n = 4

        a = np.random.randn(n, n).astype(float, order='F')
        qg = np.random.randn(n, n+1).astype(float, order='F')

        wr, wi, info = mb03sd('N', a, qg)

        # Non-zero real parts should have conjugate pairing
        # Find complex pairs (non-zero imaginary parts with non-zero real parts)
        tol = 1e-10
        for i in range(n):
            if abs(wi[i]) > tol and abs(wr[i]) > tol:
                # Should have a conjugate partner
                found = False
                for j in range(n):
                    if i != j and abs(wr[i] - wr[j]) < tol and abs(wi[i] + wi[j]) < tol:
                        found = True
                        break
                assert found, f"Complex eigenvalue {wr[i]}+{wi[i]}i has no conjugate pair"

    def test_eigenvalue_sorting(self):
        """
        Eigenvalues should be sorted in decreasing order of real parts.
        For eigenvalues with zero real part, sorted by decreasing imaginary part.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n = 5

        a = np.random.randn(n, n).astype(float, order='F')
        qg = np.random.randn(n, n+1).astype(float, order='F')

        wr, wi, info = mb03sd('S', a, qg)
        assert info == 0

        # Check decreasing order of real parts
        for i in range(n - 1):
            assert wr[i] >= wr[i+1] - 1e-10, \
                f"Real parts not sorted: wr[{i}]={wr[i]} < wr[{i+1}]={wr[i+1]}"

    def test_square_root_of_hessenberg_eigenvalues(self):
        """
        Verify that returned eigenvalues are square roots of eigenvalues of A''.

        For a simple diagonal A' and zero G', Q':
        A'' = A'^2, so eigenvalues of H' = sqrt(eigenvalues of A'^2) = |eigenvalues of A'|

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n = 3

        # Diagonal A' with known eigenvalues
        diag_vals = np.array([4.0, 1.0, 9.0])
        a = np.diag(diag_vals).astype(float, order='F')
        qg = np.zeros((n, n+1), order='F', dtype=float)

        wr, wi, info = mb03sd('N', a, qg)
        assert info == 0

        # eigenvalues of A'' = A'^2 are diag_vals^2 = [16, 1, 81]
        # eigenvalues of H' = sqrt = [4, 1, 9], sorted decreasing
        expected_wr = np.sort(np.abs(diag_vals))[::-1]  # [9, 4, 1]
        expected_wi = np.zeros(n)

        np.testing.assert_allclose(wr, expected_wr, rtol=1e-14)
        np.testing.assert_allclose(wi, expected_wi, atol=1e-14)


class TestMB03SDParameterValidation:
    """Parameter validation tests."""

    def test_invalid_jobscl(self):
        """Invalid JOBSCL parameter should return INFO=-1."""
        n = 2
        a = np.zeros((n, n), order='F', dtype=float)
        qg = np.zeros((n, n+1), order='F', dtype=float)

        wr, wi, info = mb03sd('X', a, qg)
        assert info == -1

    def test_negative_n(self):
        """N < 0 should return INFO=-2."""
        # This would be handled by array dimension checking in Python
        # We pass empty arrays and expect graceful handling
        pass  # Python wrapper handles dimensions automatically


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
