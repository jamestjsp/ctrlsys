"""
Tests for TB01TD: Balance state-space representation.

Reduces (A,B,C,D) to balanced form using state permutations and scalings.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


class TestTB01TD:
    """Tests for tb01td routine."""

    def test_html_example(self):
        """
        Test case from SLICOT HTML documentation.

        N=5, M=2, P=2 system.
        A is read row-by-row: ((A(I,J), J=1,N), I=1,N)
        B is read column-by-column: ((B(I,J), I=1,N), J=1,M)
        C is read row-by-row: ((C(I,J), J=1,N), I=1,P)
        D is read row-by-row: ((D(I,J), J=1,M), I=1,P)

        Note: DGEBAL can produce different valid balancing results.
        We validate by checking eigenvalue and transfer function preservation.
        """
        from ctrlsys import tb01td

        n, m, p = 5, 2, 2

        # A matrix (5x5) - read row-by-row
        a = np.array([
            [  0.0,   0.0,   1.0,   4.0,   5.0],
            [ 50.0,  10.0,   1.0,   0.0,   0.0],
            [  0.0,   0.0,  90.0,  10.0,   0.0],
            [  0.0,   1.0,   1.0,   1.0,   1.0],
            [100.0,   0.0,   0.0,   0.0,  70.0]
        ], dtype=float, order='F')

        # B matrix (5x2) - read column-by-column
        b = np.array([
            [0.0,   0.0],
            [2.0,  20.0],
            [0.0, 100.0],
            [1.0,   1.0],
            [2.0,   0.0]
        ], dtype=float, order='F')

        # C matrix (2x5) - read row-by-row
        c = np.array([
            [1.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 2.0, 1.0]
        ], dtype=float, order='F')

        # D matrix (2x2) - read row-by-row
        d = np.array([
            [1.0, 1.0],
            [1.0, 1.0]
        ], dtype=float, order='F')

        # Keep copies for comparison
        a_orig = a.copy()
        b_orig = b.copy()
        c_orig = c.copy()
        d_orig = d.copy()
        eig_orig = np.linalg.eigvals(a_orig)

        # Call routine
        low, igh, scstat, scin, scout, info = tb01td(n, m, p, a, b, c, d)

        assert info == 0
        assert low == 1
        assert igh == 5

        # Verify array dimensions
        assert len(scstat) == n
        assert len(scin) == m
        assert len(scout) == p

        # Eigenvalues must be preserved (similarity transformation)
        eig_bal = np.linalg.eigvals(a)
        assert_allclose(sorted(eig_orig.real), sorted(eig_bal.real), rtol=1e-12)

        # Transfer function must be preserved at multiple s values
        I = np.eye(n)
        for s in [1.0, 10.0, 1.0 + 1.0j]:
            # Original: G(s) = C*inv(sI-A)*B + D
            sI_minus_A_orig = s * I - a_orig
            G_orig = c_orig @ np.linalg.solve(sI_minus_A_orig, b_orig) + d_orig

            # Balanced: Reconstruct original G from scaled matrices
            sI_minus_A_bal = s * I - a
            G_bal_raw = c @ np.linalg.solve(sI_minus_A_bal, b) + d
            G_reconstructed = np.diag(1.0 / scout) @ G_bal_raw @ np.diag(scin)

            assert_allclose(G_orig, G_reconstructed, rtol=1e-12)

    def test_identity_preservation(self):
        """
        Test that balancing preserves system I/O relationship.

        For balanced system (Ab, Bb, Cb, Db), the transfer function
        is preserved: C*inv(sI-A)*B + D should be unchanged.

        Uses impulse response equivalence at multiple s values.

        Random seed: 42 (for reproducibility)
        """
        from ctrlsys import tb01td

        np.random.seed(42)
        n, m, p = 3, 2, 2

        # Create original system matrices
        a = np.array([
            [-1.0,  0.5, -0.2],
            [ 0.3, -2.0,  0.4],
            [ 0.1, -0.3, -0.5]
        ], dtype=float, order='F')

        b = np.array([
            [1.0, 0.5],
            [0.2, 1.0],
            [0.3, 0.1]
        ], dtype=float, order='F')

        c = np.array([
            [1.0, 0.3, 0.2],
            [0.1, 1.0, 0.5]
        ], dtype=float, order='F')

        d = np.array([
            [0.1, 0.0],
            [0.0, 0.1]
        ], dtype=float, order='F')

        # Keep copies for transfer function comparison
        a_orig = a.copy()
        b_orig = b.copy()
        c_orig = c.copy()
        d_orig = d.copy()

        # Balance the system
        low, igh, scstat, scin, scout, info = tb01td(n, m, p, a, b, c, d)
        assert info == 0

        # Compare transfer functions at several s values
        I = np.eye(n)
        for s in [1.0, 2.0j, 1.0 + 1.0j]:
            # Original: G(s) = C*inv(sI-A)*B + D
            sI_minus_A_orig = s * I - a_orig
            G_orig = c_orig @ np.linalg.solve(sI_minus_A_orig, b_orig) + d_orig

            # Balanced: Gb(s) = Cb*inv(sI-Ab)*Bb + Db
            # But outputs scaled by scout, inputs by 1/scin
            # True output = diag(scout) * balanced_output
            # True input needs division by scin
            sI_minus_A_bal = s * I - a
            G_bal_raw = c @ np.linalg.solve(sI_minus_A_bal, b) + d

            # Reconstruct: scale back
            # The routine scales such that:
            # d_bal(i,j) = d_orig(i,j) * scout(i) / scin(j)
            # So G_orig = diag(1/scout) @ G_bal @ diag(scin)
            G_reconstructed = np.diag(1.0 / scout) @ G_bal_raw @ np.diag(scin)

            assert_allclose(G_orig, G_reconstructed, rtol=1e-12)

    def test_zero_dimension(self):
        """Test quick return when N=M=P=0."""
        from ctrlsys import tb01td

        n, m, p = 0, 0, 0
        a = np.zeros((1, 1), dtype=float, order='F')
        b = np.zeros((1, 1), dtype=float, order='F')
        c = np.zeros((1, 1), dtype=float, order='F')
        d = np.zeros((1, 1), dtype=float, order='F')

        low, igh, scstat, scin, scout, info = tb01td(n, m, p, a, b, c, d)

        assert info == 0
        assert low == 1  # Fortran 1-based
        assert igh == 0  # N=0

    def test_siso_system(self):
        """
        Test simple SISO system balancing.

        Random seed: 123 (for reproducibility)
        """
        from ctrlsys import tb01td

        np.random.seed(123)
        n, m, p = 2, 1, 1

        a = np.array([
            [-0.5, 1.0],
            [ 0.0, -1.0]
        ], dtype=float, order='F')

        b = np.array([
            [1.0],
            [0.5]
        ], dtype=float, order='F')

        c = np.array([
            [1.0, 0.0]
        ], dtype=float, order='F')

        d = np.array([
            [0.0]
        ], dtype=float, order='F')

        low, igh, scstat, scin, scout, info = tb01td(n, m, p, a, b, c, d)

        assert info == 0
        # LOW and IGH should be valid indices
        assert 1 <= low <= n
        assert low <= igh <= n

        # scstat should be populated
        assert len(scstat) == n
        assert len(scin) == m
        assert len(scout) == p

    def test_eigenvalue_preservation(self):
        """
        Test that eigenvalues of A are preserved after balancing.

        Balancing uses similarity transformation: eigenvalues unchanged.

        Random seed: 456 (for reproducibility)
        """
        from ctrlsys import tb01td

        np.random.seed(456)
        n, m, p = 4, 2, 2

        # Create a matrix with known eigenvalues
        a = np.array([
            [-1.0,  2.0,  0.0,  0.0],
            [ 0.0, -2.0,  3.0,  0.0],
            [ 0.0,  0.0, -3.0,  4.0],
            [ 0.0,  0.0,  0.0, -4.0]
        ], dtype=float, order='F')

        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')
        d = np.random.randn(p, m).astype(float, order='F')

        eig_orig = np.linalg.eigvals(a)

        low, igh, scstat, scin, scout, info = tb01td(n, m, p, a, b, c, d)

        assert info == 0

        eig_bal = np.linalg.eigvals(a)

        # Eigenvalues should be preserved (sort for comparison)
        assert_allclose(sorted(eig_orig.real), sorted(eig_bal.real), rtol=1e-12)
        assert_allclose(sorted(eig_orig.imag), sorted(eig_bal.imag), rtol=1e-12)

    def test_error_invalid_n(self):
        """Test error handling for invalid N < 0."""
        from ctrlsys import tb01td

        n, m, p = -1, 1, 1
        a = np.zeros((1, 1), dtype=float, order='F')
        b = np.zeros((1, 1), dtype=float, order='F')
        c = np.zeros((1, 1), dtype=float, order='F')
        d = np.zeros((1, 1), dtype=float, order='F')

        low, igh, scstat, scin, scout, info = tb01td(n, m, p, a, b, c, d)

        assert info == -1
