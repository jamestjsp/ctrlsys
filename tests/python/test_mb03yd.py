"""Tests for MB03YD - Periodic QR iteration for eigenvalues of periodic Hessenberg matrix."""

import numpy as np
import pytest


class TestMB03YDBasic:
    """Basic functionality tests for MB03YD."""

    def test_basic_2x2_real_eigenvalues(self):
        """
        Test 2x2 periodic matrix pair with real eigenvalues.

        Random seed: 42 (for reproducibility)
        """
        from ctrlsys import mb03yd

        np.random.seed(42)
        n = 2

        a = np.array([
            [3.0, 1.0],
            [0.5, 2.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0, 0.5],
            [0.0, 2.0]
        ], order='F', dtype=float)

        q = np.eye(n, order='F', dtype=float)
        z = np.eye(n, order='F', dtype=float)

        alphar, alphai, beta, info = mb03yd(
            wantt=True, wantq=True, wantz=True,
            n=n, ilo=1, ihi=n, iloq=1, ihiq=n,
            a=a, b=b, q=q, z=z
        )

        assert info == 0
        assert alphar.shape == (n,)
        assert alphai.shape == (n,)
        assert beta.shape == (n,)

        for i in range(n):
            assert beta[i] != 0.0 or abs(alphar[i]) < 1e-10

    def test_basic_3x3_complex_eigenvalues(self):
        """
        Test 3x3 matrix pair that produces complex conjugate eigenvalues.

        Random seed: 123 (for reproducibility)
        """
        from ctrlsys import mb03yd

        np.random.seed(123)
        n = 3

        a = np.array([
            [1.0,  2.0, 0.5],
            [1.5, -1.0, 1.0],
            [0.0,  0.5, 2.0]
        ], order='F', dtype=float)

        b = np.array([
            [2.0, 1.0, 0.0],
            [0.0, 1.5, 0.5],
            [0.0, 0.0, 1.0]
        ], order='F', dtype=float)

        q = np.eye(n, order='F', dtype=float)
        z = np.eye(n, order='F', dtype=float)

        alphar, alphai, beta, info = mb03yd(
            wantt=True, wantq=True, wantz=True,
            n=n, ilo=1, ihi=n, iloq=1, ihiq=n,
            a=a, b=b, q=q, z=z
        )

        assert info == 0
        assert len(alphar) == n
        assert len(alphai) == n
        assert len(beta) == n

        complex_pairs = sum(1 for x in alphai if abs(x) > 1e-10)
        assert complex_pairs % 2 == 0

    def test_eigenvalues_only_mode(self):
        """
        Test computing eigenvalues only (wantt=False).

        Random seed: 456 (for reproducibility)
        """
        from ctrlsys import mb03yd

        np.random.seed(456)
        n = 4

        a = np.array([
            [2.0, 1.0, 0.0, 0.0],
            [1.0, 2.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 1.0],
            [0.0, 0.0, 1.0, 2.0]
        ], order='F', dtype=float)

        b = np.diag([1.0, 2.0, 1.5, 1.0]).astype(float, order='F')

        q = np.eye(n, order='F', dtype=float)
        z = np.eye(n, order='F', dtype=float)

        alphar, alphai, beta, info = mb03yd(
            wantt=False, wantq=False, wantz=False,
            n=n, ilo=1, ihi=n, iloq=1, ihiq=n,
            a=a, b=b, q=q, z=z
        )

        assert info == 0
        assert len(alphar) == n


class TestMB03YDMathematicalProperties:
    """Mathematical property tests for MB03YD."""

    def test_orthogonal_transformation_q(self):
        """
        Validate Q remains orthogonal after transformation.

        Random seed: 789 (for reproducibility)
        """
        from ctrlsys import mb03yd

        np.random.seed(789)
        n = 4

        h = np.random.randn(n, n).astype(float, order='F')
        for i in range(n):
            for j in range(i + 2, n):
                h[j, i] = 0.0

        b = np.triu(np.random.randn(n, n)).astype(float, order='F')
        np.fill_diagonal(b, np.abs(np.diag(b)) + 0.5)

        q = np.eye(n, order='F', dtype=float)
        z = np.eye(n, order='F', dtype=float)

        alphar, alphai, beta, info = mb03yd(
            wantt=True, wantq=True, wantz=True,
            n=n, ilo=1, ihi=n, iloq=1, ihiq=n,
            a=h, b=b, q=q, z=z
        )

        assert info == 0

        qtq = q.T @ q
        np.testing.assert_allclose(qtq, np.eye(n), rtol=1e-13, atol=1e-14)

    def test_orthogonal_transformation_z(self):
        """
        Validate Z remains orthogonal after transformation.

        Random seed: 321 (for reproducibility)
        """
        from ctrlsys import mb03yd

        np.random.seed(321)
        n = 4

        h = np.random.randn(n, n).astype(float, order='F')
        for i in range(n):
            for j in range(i + 2, n):
                h[j, i] = 0.0

        b = np.triu(np.random.randn(n, n)).astype(float, order='F')
        np.fill_diagonal(b, np.abs(np.diag(b)) + 0.5)

        q = np.eye(n, order='F', dtype=float)
        z = np.eye(n, order='F', dtype=float)

        alphar, alphai, beta, info = mb03yd(
            wantt=True, wantq=True, wantz=True,
            n=n, ilo=1, ihi=n, iloq=1, ihiq=n,
            a=h, b=b, q=q, z=z
        )

        assert info == 0

        ztz = z.T @ z
        np.testing.assert_allclose(ztz, np.eye(n), rtol=1e-13, atol=1e-14)

    def test_schur_form_structure(self):
        """
        Validate output A is quasi-triangular (Schur form).

        Random seed: 654 (for reproducibility)
        """
        from ctrlsys import mb03yd

        np.random.seed(654)
        n = 5

        h = np.random.randn(n, n).astype(float, order='F')
        for i in range(n):
            for j in range(i + 2, n):
                h[j, i] = 0.0

        b = np.triu(np.random.randn(n, n)).astype(float, order='F')
        np.fill_diagonal(b, np.abs(np.diag(b)) + 0.5)

        q = np.eye(n, order='F', dtype=float)
        z = np.eye(n, order='F', dtype=float)

        alphar, alphai, beta, info = mb03yd(
            wantt=True, wantq=True, wantz=True,
            n=n, ilo=1, ihi=n, iloq=1, ihiq=n,
            a=h, b=b, q=q, z=z
        )

        assert info == 0

        for j in range(n - 2):
            subdiag_j = abs(h[j + 1, j])
            subdiag_j1 = abs(h[j + 2, j + 1])
            if subdiag_j > 1e-10:
                assert subdiag_j1 < 1e-10


class TestMB03YDEdgeCases:
    """Edge case tests for MB03YD."""

    def test_n_equals_zero(self):
        """Test with n=0 (empty matrices)."""
        from ctrlsys import mb03yd

        n = 0
        a = np.array([], order='F', dtype=float).reshape(0, 0)
        b = np.array([], order='F', dtype=float).reshape(0, 0)
        q = np.array([], order='F', dtype=float).reshape(0, 0)
        z = np.array([], order='F', dtype=float).reshape(0, 0)

        alphar, alphai, beta, info = mb03yd(
            wantt=True, wantq=True, wantz=True,
            n=n, ilo=1, ihi=0, iloq=1, ihiq=0,
            a=a, b=b, q=q, z=z
        )

        assert info == 0

    def test_n_equals_one(self):
        """Test with 1x1 matrices."""
        from ctrlsys import mb03yd

        n = 1
        a = np.array([[3.0]], order='F', dtype=float)
        b = np.array([[2.0]], order='F', dtype=float)
        q = np.array([[1.0]], order='F', dtype=float)
        z = np.array([[1.0]], order='F', dtype=float)

        alphar, alphai, beta, info = mb03yd(
            wantt=True, wantq=True, wantz=True,
            n=n, ilo=1, ihi=n, iloq=1, ihiq=n,
            a=a, b=b, q=q, z=z
        )

        assert info == 0
        assert abs(alphar[0] - 3.0) < 1e-14
        assert abs(alphai[0]) < 1e-14
        assert abs(beta[0] - 2.0) < 1e-14

    def test_partial_subproblem(self):
        """
        Test with ILO > 1 and IHI < N (partial subproblem).

        Random seed: 999 (for reproducibility)
        """
        from ctrlsys import mb03yd

        np.random.seed(999)
        n = 5
        ilo = 2
        ihi = 4

        a = np.zeros((n, n), order='F', dtype=float)
        a[0, 0] = 1.0
        a[n-1, n-1] = 5.0
        for i in range(ilo-1, ihi):
            for j in range(max(0, i-1), n):
                a[i, j] = np.random.randn()

        b = np.eye(n, order='F', dtype=float)
        for i in range(ilo-1, ihi):
            for j in range(i, n):
                b[i, j] = np.random.randn()
        np.fill_diagonal(b, np.abs(np.diag(b)) + 0.5)

        q = np.eye(n, order='F', dtype=float)
        z = np.eye(n, order='F', dtype=float)

        alphar, alphai, beta, info = mb03yd(
            wantt=True, wantq=True, wantz=True,
            n=n, ilo=ilo, ihi=ihi, iloq=1, ihiq=n,
            a=a, b=b, q=q, z=z
        )

        assert info == 0


class TestMB03YDErrorHandling:
    """Error handling tests for MB03YD."""

    def test_negative_n(self):
        """Test error for negative N."""
        from ctrlsys import mb03yd

        n = 2
        a = np.eye(n, order='F', dtype=float)
        b = np.eye(n, order='F', dtype=float)
        q = np.eye(n, order='F', dtype=float)
        z = np.eye(n, order='F', dtype=float)

        alphar, alphai, beta, info = mb03yd(
            wantt=True, wantq=True, wantz=True,
            n=-1, ilo=1, ihi=1, iloq=1, ihiq=1,
            a=a, b=b, q=q, z=z
        )

        assert info == -4

    def test_invalid_ilo(self):
        """Test error for invalid ILO."""
        from ctrlsys import mb03yd

        n = 3
        a = np.eye(n, order='F', dtype=float)
        b = np.eye(n, order='F', dtype=float)
        q = np.eye(n, order='F', dtype=float)
        z = np.eye(n, order='F', dtype=float)

        alphar, alphai, beta, info = mb03yd(
            wantt=True, wantq=True, wantz=True,
            n=n, ilo=0, ihi=n, iloq=1, ihiq=n,
            a=a, b=b, q=q, z=z
        )

        assert info == -5

    def test_invalid_ihi(self):
        """Test error for invalid IHI."""
        from ctrlsys import mb03yd

        n = 3
        a = np.eye(n, order='F', dtype=float)
        b = np.eye(n, order='F', dtype=float)
        q = np.eye(n, order='F', dtype=float)
        z = np.eye(n, order='F', dtype=float)

        alphar, alphai, beta, info = mb03yd(
            wantt=True, wantq=True, wantz=True,
            n=n, ilo=1, ihi=n+1, iloq=1, ihiq=n,
            a=a, b=b, q=q, z=z
        )

        assert info == -6
