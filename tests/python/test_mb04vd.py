"""
Tests for MB04VD: Upper block triangular form for rectangular pencil sE-A.

MB04VD computes orthogonal transformations Q and Z such that Q'(sE-A)Z
is in upper block triangular form, where E is in column echelon form.

MODE options:
- 'B': Basic reduction to staircase form
- 'T': Triangularization of full rank submatrices
- 'S': Separation of epsilon and infinite parts
"""

import numpy as np
import pytest
from ctrlsys import mb04ud, mb04vd


class TestMB04VDBasic:
    """Basic functionality tests for MB04VD."""

    def test_html_example_mode_s(self):
        """
        Test MB04VD using HTML documentation example with MODE='S'.

        From MB04VD.html:
        M=2, N=4, TOL=0.0, MODE='S'

        Input A (row-wise):
        1.0  0.0 -1.0  0.0
        1.0  1.0  0.0 -1.0

        Input E (row-wise):
        0.0 -1.0  0.0  0.0
        0.0 -1.0  0.0  0.0

        Expected NBLCKS=2, NBLCKI=1
        IMUK = [2, 1], INUK = [1, 0], IMUK0 = [1]
        MNEI = [1, 3, 1]

        Expected matrices (from HTML results):
        Q = [[0.7071, -0.7071], [0.7071, 0.7071]]
        E has staircase structure
        A has staircase structure
        """
        m, n = 2, 4

        a = np.array([
            [1.0, 0.0, -1.0, 0.0],
            [1.0, 1.0, 0.0, -1.0]
        ], dtype=float, order='F')

        e = np.array([
            [0.0, -1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0]
        ], dtype=float, order='F')

        # First call MB04UD to reduce E to column echelon form
        a_ud, e_ud, q_ud, z_ud, ranke, istair, info = mb04ud(
            'I', 'I', m, n, a.copy(order='F'), e.copy(order='F')
        )
        assert info == 0

        # Now call MB04VD with MODE='S' and JOBQ='U', JOBZ='U'
        (a_vd, e_vd, q_vd, z_vd, nblcks, nblcki,
         imuk, inuk, imuk0, mnei, info_vd) = mb04vd(
            'S', 'U', 'U', m, n, ranke, a_ud.copy(order='F'), e_ud.copy(order='F'),
            q_ud.copy(order='F'), z_ud.copy(order='F'), istair.copy()
        )

        assert info_vd == 0
        assert nblcks == 2
        assert nblcki == 1

        # Expected dimensions
        np.testing.assert_array_equal(imuk[:nblcks], [2, 1])
        np.testing.assert_array_equal(inuk[:nblcks], [1, 0])
        np.testing.assert_array_equal(imuk0[:nblcki], [1])
        np.testing.assert_array_equal(mnei, [1, 3, 1])

        # Expected Q matrix
        q_expected = np.array([
            [0.7071, -0.7071],
            [0.7071, 0.7071]
        ], dtype=float, order='F')
        np.testing.assert_allclose(np.abs(q_vd), np.abs(q_expected), rtol=1e-3, atol=1e-4)

        # Expected E matrix (from HTML)
        e_expected = np.array([
            [0.0000, 0.0000, -1.1547, 0.8165],
            [0.0000, 0.0000, 0.0000, 0.0000]
        ], dtype=float, order='F')
        np.testing.assert_allclose(e_vd, e_expected, rtol=1e-3, atol=1e-4)

        # Expected A matrix (from HTML)
        a_expected = np.array([
            [0.0000, 1.7321, 0.5774, -0.4082],
            [0.0000, 0.0000, 0.0000, -1.2247]
        ], dtype=float, order='F')
        np.testing.assert_allclose(a_vd, a_expected, rtol=1e-3, atol=1e-4)

    def test_mode_b_basic_reduction(self):
        """
        Test MB04VD with MODE='B' for basic reduction.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        m, n = 4, 5

        a = np.random.randn(m, n).astype(float, order='F')
        e = np.random.randn(m, n).astype(float, order='F')

        # First call MB04UD
        a_ud, e_ud, q_ud, z_ud, ranke, istair, info = mb04ud(
            'I', 'I', m, n, a.copy(order='F'), e.copy(order='F')
        )
        assert info == 0

        # Call MB04VD with MODE='B'
        (a_vd, e_vd, q_vd, z_vd, nblcks, nblcki,
         imuk, inuk, imuk0, mnei, info_vd) = mb04vd(
            'B', 'U', 'U', m, n, ranke, a_ud.copy(order='F'), e_ud.copy(order='F'),
            q_ud.copy(order='F'), z_ud.copy(order='F'), istair.copy()
        )

        assert info_vd == 0
        assert nblcki == 0  # MODE='B' sets NBLCKI=0
        assert mnei[2] == 0  # MNEI(3)=0 for MODE='B'
        assert nblcks >= 0

    def test_mode_t_triangular(self):
        """
        Test MB04VD with MODE='T' for triangularization.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        m, n = 4, 5

        a = np.random.randn(m, n).astype(float, order='F')
        e = np.random.randn(m, n).astype(float, order='F')

        # First call MB04UD
        a_ud, e_ud, q_ud, z_ud, ranke, istair, info = mb04ud(
            'I', 'I', m, n, a.copy(order='F'), e.copy(order='F')
        )
        assert info == 0

        # Call MB04VD with MODE='T'
        (a_vd, e_vd, q_vd, z_vd, nblcks, nblcki,
         imuk, inuk, imuk0, mnei, info_vd) = mb04vd(
            'T', 'U', 'U', m, n, ranke, a_ud.copy(order='F'), e_ud.copy(order='F'),
            q_ud.copy(order='F'), z_ud.copy(order='F'), istair.copy()
        )

        assert info_vd == 0
        assert nblcki == 0  # MODE='T' sets NBLCKI=0
        assert mnei[2] == 0  # MNEI(3)=0 for MODE='T'


class TestMB04VDOrthogonality:
    """Test orthogonality preservation in MB04VD."""

    def test_q_orthogonality(self):
        """
        Test that Q remains orthogonal: Q'*Q = I.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        m, n = 5, 4

        a = np.random.randn(m, n).astype(float, order='F')
        e = np.random.randn(m, n).astype(float, order='F')

        a_ud, e_ud, q_ud, z_ud, ranke, istair, info = mb04ud(
            'I', 'I', m, n, a.copy(order='F'), e.copy(order='F')
        )
        assert info == 0

        (a_vd, e_vd, q_vd, z_vd, nblcks, nblcki,
         imuk, inuk, imuk0, mnei, info_vd) = mb04vd(
            'S', 'U', 'U', m, n, ranke, a_ud.copy(order='F'), e_ud.copy(order='F'),
            q_ud.copy(order='F'), z_ud.copy(order='F'), istair.copy()
        )

        assert info_vd == 0

        # Q'*Q = I
        qtq = q_vd.T @ q_vd
        np.testing.assert_allclose(qtq, np.eye(m), rtol=1e-14, atol=1e-14)

    def test_z_orthogonality(self):
        """
        Test that Z remains orthogonal: Z'*Z = I.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        m, n = 4, 6

        a = np.random.randn(m, n).astype(float, order='F')
        e = np.random.randn(m, n).astype(float, order='F')

        a_ud, e_ud, q_ud, z_ud, ranke, istair, info = mb04ud(
            'I', 'I', m, n, a.copy(order='F'), e.copy(order='F')
        )
        assert info == 0

        (a_vd, e_vd, q_vd, z_vd, nblcks, nblcki,
         imuk, inuk, imuk0, mnei, info_vd) = mb04vd(
            'S', 'U', 'U', m, n, ranke, a_ud.copy(order='F'), e_ud.copy(order='F'),
            q_ud.copy(order='F'), z_ud.copy(order='F'), istair.copy()
        )

        assert info_vd == 0

        # Z'*Z = I
        ztz = z_vd.T @ z_vd
        np.testing.assert_allclose(ztz, np.eye(n), rtol=1e-14, atol=1e-14)


class TestMB04VDTransformation:
    """Test transformation properties of MB04VD."""

    def test_transformation_consistency(self):
        """
        Test that Q' * A_orig * Z = A_out relationship holds.

        For the full transformation chain:
        Q_total = Q_ud * Q_vd
        Z_total = Z_ud * Z_vd
        Q_total' * A_orig * Z_total = A_vd

        Random seed: 999 (for reproducibility)
        """
        np.random.seed(999)
        m, n = 4, 5

        a_orig = np.random.randn(m, n).astype(float, order='F')
        e_orig = np.random.randn(m, n).astype(float, order='F')

        a_ud, e_ud, q_ud, z_ud, ranke, istair, info = mb04ud(
            'I', 'I', m, n, a_orig.copy(order='F'), e_orig.copy(order='F')
        )
        assert info == 0

        (a_vd, e_vd, q_vd, z_vd, nblcks, nblcki,
         imuk, inuk, imuk0, mnei, info_vd) = mb04vd(
            'S', 'U', 'U', m, n, ranke, a_ud.copy(order='F'), e_ud.copy(order='F'),
            q_ud.copy(order='F'), z_ud.copy(order='F'), istair.copy()
        )

        assert info_vd == 0

        # Q_vd and Z_vd are the accumulated transformations
        # Q_vd = Q_ud * Q_vd_internal, Z_vd = Z_ud * Z_vd_internal
        # So Q_vd' * A_orig * Z_vd = A_vd
        a_transformed = q_vd.T @ a_orig @ z_vd
        np.testing.assert_allclose(a_transformed, a_vd, rtol=1e-13, atol=1e-14)

        e_transformed = q_vd.T @ e_orig @ z_vd
        np.testing.assert_allclose(e_transformed, e_vd, rtol=1e-13, atol=1e-14)

    def test_frobenius_norm_preservation(self):
        """
        Test that orthogonal transformations preserve Frobenius norm.

        ||A||_F = ||Q'*A*Z||_F for orthogonal Q, Z.

        Random seed: 888 (for reproducibility)
        """
        np.random.seed(888)
        m, n = 4, 5

        a = np.random.randn(m, n).astype(float, order='F')
        e = np.random.randn(m, n).astype(float, order='F')

        a_norm_before = np.linalg.norm(a, 'fro')
        e_norm_before = np.linalg.norm(e, 'fro')

        a_ud, e_ud, q_ud, z_ud, ranke, istair, info = mb04ud(
            'I', 'I', m, n, a.copy(order='F'), e.copy(order='F')
        )
        assert info == 0

        (a_vd, e_vd, q_vd, z_vd, nblcks, nblcki,
         imuk, inuk, imuk0, mnei, info_vd) = mb04vd(
            'S', 'U', 'U', m, n, ranke, a_ud.copy(order='F'), e_ud.copy(order='F'),
            q_ud.copy(order='F'), z_ud.copy(order='F'), istair.copy()
        )

        assert info_vd == 0

        a_norm_after = np.linalg.norm(a_vd, 'fro')
        e_norm_after = np.linalg.norm(e_vd, 'fro')

        np.testing.assert_allclose(a_norm_after, a_norm_before, rtol=1e-14)
        np.testing.assert_allclose(e_norm_after, e_norm_before, rtol=1e-14)


class TestMB04VDEdgeCases:
    """Edge case tests for MB04VD."""

    def test_zero_rows(self):
        """
        Test MB04VD with M=0 (quick return).
        """
        m, n = 0, 3

        a = np.zeros((1, n), dtype=float, order='F')
        e = np.zeros((1, n), dtype=float, order='F')
        q = np.array([[1.0]], dtype=float, order='F')
        z = np.eye(n, dtype=float, order='F')
        istair = np.array([], dtype=np.int32)

        (a_vd, e_vd, q_vd, z_vd, nblcks, nblcki,
         imuk, inuk, imuk0, mnei, info_vd) = mb04vd(
            'B', 'N', 'I', m, n, 0, a, e, q, z, istair
        )

        assert info_vd == 0
        assert nblcks == 1
        assert nblcki == 0
        assert imuk[0] == n
        assert inuk[0] == 0
        np.testing.assert_array_equal(mnei, [0, n, 0])

    def test_zero_cols(self):
        """
        Test MB04VD with N=0 (quick return).
        """
        m, n = 3, 0

        a = np.zeros((m, 1), dtype=float, order='F')
        e = np.zeros((m, 1), dtype=float, order='F')
        q = np.eye(m, dtype=float, order='F')
        z = np.array([[1.0]], dtype=float, order='F')
        istair = np.zeros(m, dtype=np.int32)

        (a_vd, e_vd, q_vd, z_vd, nblcks, nblcki,
         imuk, inuk, imuk0, mnei, info_vd) = mb04vd(
            'B', 'I', 'N', m, n, 0, a, e, q, z, istair
        )

        assert info_vd == 0
        assert nblcks == 0
        np.testing.assert_array_equal(mnei, [0, 0, 0])

    def test_no_q_update(self):
        """
        Test MB04VD with JOBQ='N' (no Q accumulation).

        Random seed: 111 (for reproducibility)
        """
        np.random.seed(111)
        m, n = 3, 4

        a = np.random.randn(m, n).astype(float, order='F')
        e = np.random.randn(m, n).astype(float, order='F')

        a_ud, e_ud, q_ud, z_ud, ranke, istair, info = mb04ud(
            'I', 'I', m, n, a.copy(order='F'), e.copy(order='F')
        )
        assert info == 0

        # Dummy Q
        q_dummy = np.array([[1.0]], dtype=float, order='F')

        (a_vd, e_vd, q_vd, z_vd, nblcks, nblcki,
         imuk, inuk, imuk0, mnei, info_vd) = mb04vd(
            'B', 'N', 'U', m, n, ranke, a_ud.copy(order='F'), e_ud.copy(order='F'),
            q_dummy, z_ud.copy(order='F'), istair.copy()
        )

        assert info_vd == 0
        assert q_vd is None

    def test_no_z_update(self):
        """
        Test MB04VD with JOBZ='N' (no Z accumulation).

        Random seed: 222 (for reproducibility)
        """
        np.random.seed(222)
        m, n = 3, 4

        a = np.random.randn(m, n).astype(float, order='F')
        e = np.random.randn(m, n).astype(float, order='F')

        a_ud, e_ud, q_ud, z_ud, ranke, istair, info = mb04ud(
            'I', 'I', m, n, a.copy(order='F'), e.copy(order='F')
        )
        assert info == 0

        # Dummy Z
        z_dummy = np.array([[1.0]], dtype=float, order='F')

        (a_vd, e_vd, q_vd, z_vd, nblcks, nblcki,
         imuk, inuk, imuk0, mnei, info_vd) = mb04vd(
            'B', 'U', 'N', m, n, ranke, a_ud.copy(order='F'), e_ud.copy(order='F'),
            q_ud.copy(order='F'), z_dummy, istair.copy()
        )

        assert info_vd == 0
        assert z_vd is None


class TestMB04VDDeterminant:
    """Test determinant properties for orthogonal transformations."""

    def test_determinant_q(self):
        """
        Test that |det(Q)| = 1 (orthogonal matrix property).

        Random seed: 333 (for reproducibility)
        """
        np.random.seed(333)
        m, n = 5, 4

        a = np.random.randn(m, n).astype(float, order='F')
        e = np.random.randn(m, n).astype(float, order='F')

        a_ud, e_ud, q_ud, z_ud, ranke, istair, info = mb04ud(
            'I', 'I', m, n, a.copy(order='F'), e.copy(order='F')
        )
        assert info == 0

        (a_vd, e_vd, q_vd, z_vd, nblcks, nblcki,
         imuk, inuk, imuk0, mnei, info_vd) = mb04vd(
            'S', 'U', 'U', m, n, ranke, a_ud.copy(order='F'), e_ud.copy(order='F'),
            q_ud.copy(order='F'), z_ud.copy(order='F'), istair.copy()
        )

        assert info_vd == 0

        det_q = np.linalg.det(q_vd)
        np.testing.assert_allclose(abs(det_q), 1.0, rtol=1e-14)

    def test_determinant_z(self):
        """
        Test that |det(Z)| = 1 (orthogonal matrix property).

        Random seed: 444 (for reproducibility)
        """
        np.random.seed(444)
        m, n = 4, 6

        a = np.random.randn(m, n).astype(float, order='F')
        e = np.random.randn(m, n).astype(float, order='F')

        a_ud, e_ud, q_ud, z_ud, ranke, istair, info = mb04ud(
            'I', 'I', m, n, a.copy(order='F'), e.copy(order='F')
        )
        assert info == 0

        (a_vd, e_vd, q_vd, z_vd, nblcks, nblcki,
         imuk, inuk, imuk0, mnei, info_vd) = mb04vd(
            'S', 'U', 'U', m, n, ranke, a_ud.copy(order='F'), e_ud.copy(order='F'),
            q_ud.copy(order='F'), z_ud.copy(order='F'), istair.copy()
        )

        assert info_vd == 0

        det_z = np.linalg.det(z_vd)
        np.testing.assert_allclose(abs(det_z), 1.0, rtol=1e-14)


class TestMB04VDWideAndTall:
    """Test MB04VD with wide and tall matrices."""

    def test_wide_matrix(self):
        """
        Test MB04VD with wide matrix (M < N).

        Random seed: 555 (for reproducibility)
        """
        np.random.seed(555)
        m, n = 3, 6

        a = np.random.randn(m, n).astype(float, order='F')
        e = np.random.randn(m, n).astype(float, order='F')

        a_ud, e_ud, q_ud, z_ud, ranke, istair, info = mb04ud(
            'I', 'I', m, n, a.copy(order='F'), e.copy(order='F')
        )
        assert info == 0

        (a_vd, e_vd, q_vd, z_vd, nblcks, nblcki,
         imuk, inuk, imuk0, mnei, info_vd) = mb04vd(
            'S', 'U', 'U', m, n, ranke, a_ud.copy(order='F'), e_ud.copy(order='F'),
            q_ud.copy(order='F'), z_ud.copy(order='F'), istair.copy()
        )

        assert info_vd == 0
        assert q_vd.shape == (m, m)
        assert z_vd.shape == (n, n)

    def test_tall_matrix(self):
        """
        Test MB04VD with tall matrix (M > N).

        Random seed: 666 (for reproducibility)
        """
        np.random.seed(666)
        m, n = 6, 3

        a = np.random.randn(m, n).astype(float, order='F')
        e = np.random.randn(m, n).astype(float, order='F')

        a_ud, e_ud, q_ud, z_ud, ranke, istair, info = mb04ud(
            'I', 'I', m, n, a.copy(order='F'), e.copy(order='F')
        )
        assert info == 0

        (a_vd, e_vd, q_vd, z_vd, nblcks, nblcki,
         imuk, inuk, imuk0, mnei, info_vd) = mb04vd(
            'S', 'U', 'U', m, n, ranke, a_ud.copy(order='F'), e_ud.copy(order='F'),
            q_ud.copy(order='F'), z_ud.copy(order='F'), istair.copy()
        )

        assert info_vd == 0
        assert q_vd.shape == (m, m)
        assert z_vd.shape == (n, n)


class TestMB04VDInitQ:
    """Test MB04VD with JOBQ='I' initialization."""

    def test_jobq_i_initializes_q(self):
        """
        Test that JOBQ='I' initializes Q to identity before transformations.

        Random seed: 777 (for reproducibility)
        """
        np.random.seed(777)
        m, n = 4, 5

        a = np.random.randn(m, n).astype(float, order='F')
        e = np.random.randn(m, n).astype(float, order='F')

        a_ud, e_ud, q_ud, z_ud, ranke, istair, info = mb04ud(
            'N', 'N', m, n, a.copy(order='F'), e.copy(order='F')
        )
        assert info == 0

        # Dummy Q (will be initialized)
        q_dummy = np.zeros((m, m), dtype=float, order='F')

        (a_vd, e_vd, q_vd, z_vd, nblcks, nblcki,
         imuk, inuk, imuk0, mnei, info_vd) = mb04vd(
            'B', 'I', 'N', m, n, ranke, a_ud.copy(order='F'), e_ud.copy(order='F'),
            q_dummy, np.array([[1.0]], dtype=float, order='F'), istair.copy()
        )

        assert info_vd == 0
        assert q_vd is not None

        # Q should be orthogonal
        qtq = q_vd.T @ q_vd
        np.testing.assert_allclose(qtq, np.eye(m), rtol=1e-14, atol=1e-14)

    def test_jobz_i_initializes_z(self):
        """
        Test that JOBZ='I' initializes Z to identity before transformations.

        Random seed: 888 (for reproducibility)
        """
        np.random.seed(888)
        m, n = 4, 5

        a = np.random.randn(m, n).astype(float, order='F')
        e = np.random.randn(m, n).astype(float, order='F')

        a_ud, e_ud, q_ud, z_ud, ranke, istair, info = mb04ud(
            'N', 'N', m, n, a.copy(order='F'), e.copy(order='F')
        )
        assert info == 0

        # Dummy Z (will be initialized)
        z_dummy = np.zeros((n, n), dtype=float, order='F')

        (a_vd, e_vd, q_vd, z_vd, nblcks, nblcki,
         imuk, inuk, imuk0, mnei, info_vd) = mb04vd(
            'B', 'N', 'I', m, n, ranke, a_ud.copy(order='F'), e_ud.copy(order='F'),
            np.array([[1.0]], dtype=float, order='F'), z_dummy, istair.copy()
        )

        assert info_vd == 0
        assert z_vd is not None

        # Z should be orthogonal
        ztz = z_vd.T @ z_vd
        np.testing.assert_allclose(ztz, np.eye(n), rtol=1e-14, atol=1e-14)
