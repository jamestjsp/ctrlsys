"""
Tests for TB04CD: State-space to minimal pole-zero-gain form.

TB04CD computes the transfer function matrix G of a state-space
representation (A,B,C,D) in minimal pole-zero-gain form using the
pole-zeros method.
"""
import numpy as np
import pytest
from ctrlsys import tb04cd


class TestTB04CDBasic:
    """Basic functionality tests using HTML doc example."""

    def test_html_example(self):
        """
        Validate against SLICOT HTML documentation example.

        System: 3x3 diagonal state matrix, 2 inputs, 2 outputs.
        Tests pole-zero-gain computation for MIMO system.
        """
        n, m, p = 3, 2, 2
        npz = n

        # A matrix: 3x3 diagonal
        # READ ( NIN, FMT = * ) ( ( A(I,J), J = 1,N ), I = 1,N ) - row-wise
        a = np.array([
            [-1.0,  0.0,  0.0],
            [ 0.0, -2.0,  0.0],
            [ 0.0,  0.0, -3.0]
        ], order='F', dtype=float)

        # B matrix: 3x2
        # READ ( NIN, FMT = * ) ( ( B(I,J), I = 1,N ), J = 1,M ) - column-wise
        # Data: 0.0 1.0 -1.0 / 1.0 1.0 0.0
        b = np.array([
            [ 0.0,  1.0],
            [ 1.0,  1.0],
            [-1.0,  0.0]
        ], order='F', dtype=float)

        # C matrix: 2x3
        # READ ( NIN, FMT = * ) ( ( C(I,J), J = 1,N ), I = 1,P ) - row-wise
        # Data: 0.0 1.0 1.0 / 1.0 1.0 1.0
        c = np.array([
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ], order='F', dtype=float)

        # D matrix: 2x2
        # READ ( NIN, FMT = * ) ( ( D(I,J), J = 1,M ), I = 1,P ) - row-wise
        # Data: 1.0 0.0 / 0.0 1.0
        d = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        # Call TB04CD
        nz, np_arr, zerosr, zerosi, polesr, polesi, gains, info = tb04cd(
            'D', 'N', a, b, c, d, npz, tol=0.0
        )

        assert info == 0

        # Validate results from Fortran reference (note: HTML doc shows idealized minimal results,
        # but actual SLICOT returns non-minimal for element (2,1) due to numerical tolerance)
        # Element (1,1): 2 zeros, 2 poles, gain=1
        # zeros: -2.5 +/- 0.866i, poles: -2, -3
        assert nz[0, 0] == 2
        assert np_arr[0, 0] == 2
        np.testing.assert_allclose(gains[0, 0], 1.0, rtol=1e-3)

        # Element (2,1): Platform-dependent - numerical tolerance affects
        # controllability/observability detection.
        # macOS (Accelerate): 1 zero, 3 poles (non-minimal)
        # Linux (OpenBLAS): 0 zeros, 2 poles (minimal, matches HTML doc)
        assert nz[1, 0] in (0, 1)
        assert np_arr[1, 0] in (2, 3)
        np.testing.assert_allclose(gains[1, 0], 1.0, rtol=1e-3)

        # Element (1,2): 0 zeros, 1 pole, gain=1
        assert nz[0, 1] == 0
        assert np_arr[0, 1] == 1
        np.testing.assert_allclose(gains[0, 1], 1.0, rtol=1e-3)

        # Element (2,2): 2 zeros, 2 poles, gain=1
        # zeros: -3.618, -1.382 (golden ratio related)
        assert nz[1, 1] == 2
        assert np_arr[1, 1] == 2
        np.testing.assert_allclose(gains[1, 1], 1.0, rtol=1e-3)

        # Validate specific poles and zeros
        # G(1,1): zeros at -2.5 +/- 0.866i
        idx_11 = 0  # ((1-1)*2 + 1-1)*3 = 0
        zeros_11 = np.sort(zerosr[idx_11:idx_11+nz[0, 0]])
        # Both real parts should be -2.5
        np.testing.assert_allclose(zeros_11, [-2.5, -2.5], rtol=1e-3)
        # Imaginary parts: +/- sqrt(3)/2 ~ 0.866 (complex conjugate pair)
        zeros_11_imag = np.sort(np.abs(zerosi[idx_11:idx_11+nz[0, 0]]))
        np.testing.assert_allclose(zeros_11_imag, [0.866, 0.866], atol=0.01)

        # G(1,1): poles at -2, -3
        poles_11 = np.sort(polesr[idx_11:idx_11+np_arr[0, 0]])
        np.testing.assert_allclose(poles_11, [-3.0, -2.0], rtol=1e-3)

    def test_zero_d_matrix(self):
        """Test with JOBD='Z' (zero D matrix)."""
        n, m, p = 3, 2, 2
        npz = n

        a = np.array([
            [-1.0,  0.0,  0.0],
            [ 0.0, -2.0,  0.0],
            [ 0.0,  0.0, -3.0]
        ], order='F', dtype=float)

        b = np.array([
            [ 0.0,  1.0],
            [ 1.0,  1.0],
            [-1.0,  0.0]
        ], order='F', dtype=float)

        c = np.array([
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ], order='F', dtype=float)

        # D not used when jobd='Z'
        d = np.zeros((p, m), order='F', dtype=float)

        nz, np_arr, zerosr, zerosi, polesr, polesi, gains, info = tb04cd(
            'Z', 'N', a, b, c, d, npz, tol=0.0
        )

        assert info == 0
        # Gains should be different from D case since D=0

    def test_equilibration(self):
        """Test with EQUIL='S' (scaling enabled)."""
        n, m, p = 3, 2, 2
        npz = n

        a = np.array([
            [-1.0,  0.0,  0.0],
            [ 0.0, -2.0,  0.0],
            [ 0.0,  0.0, -3.0]
        ], order='F', dtype=float)

        b = np.array([
            [ 0.0,  1.0],
            [ 1.0,  1.0],
            [-1.0,  0.0]
        ], order='F', dtype=float)

        c = np.array([
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ], order='F', dtype=float)

        d = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        nz, np_arr, zerosr, zerosi, polesr, polesi, gains, info = tb04cd(
            'D', 'S', a, b, c, d, npz, tol=0.0
        )

        assert info == 0
        # Results should be similar (equilibration improves conditioning)
        np.testing.assert_allclose(gains[0, 0], 1.0, rtol=1e-2)


class TestTB04CDEdgeCases:
    """Edge case tests."""

    def test_siso_system(self):
        """Test single-input single-output system."""
        n, m, p = 2, 1, 1
        npz = n

        # Simple SISO system: (s+1)/(s^2+3s+2) = (s+1)/((s+1)(s+2)) = 1/(s+2)
        a = np.array([
            [-1.0,  0.0],
            [ 0.0, -2.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0],
            [1.0]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 1.0]
        ], order='F', dtype=float)

        d = np.array([
            [0.0]
        ], order='F', dtype=float)

        nz, np_arr, zerosr, zerosi, polesr, polesi, gains, info = tb04cd(
            'Z', 'N', a, b, c, d, npz, tol=0.0
        )

        assert info == 0
        # Should have minimal realization after controllability/observability reduction

    def test_n_zero(self):
        """Test with n=0 (direct feedthrough only)."""
        n, m, p = 0, 2, 2
        npz = 0

        a = np.zeros((1, 1), order='F', dtype=float)  # Dummy
        b = np.zeros((1, m), order='F', dtype=float)  # Dummy
        c = np.zeros((p, 1), order='F', dtype=float)  # Dummy
        d = np.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ], order='F', dtype=float)

        nz, np_arr, zerosr, zerosi, polesr, polesi, gains, info = tb04cd(
            'D', 'N', a, b, c, d, npz, tol=0.0
        )

        assert info == 0
        # All transfer functions are just D(i,j)
        for i in range(p):
            for j in range(m):
                assert nz[i, j] == 0
                assert np_arr[i, j] == 0
                np.testing.assert_allclose(gains[i, j], d[i, j], rtol=1e-14)


class TestTB04CDMathematical:
    """Mathematical property tests."""

    def test_pole_count_matches_minimal_order(self):
        """
        Validate that pole count reflects minimal realization order.

        For a controllable and observable system, pole count equals n.
        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n, m, p = 3, 1, 1
        npz = n

        # Create a controllable and observable system
        # Use companion form to ensure controllability
        a = np.array([
            [ 0.0,  1.0,  0.0],
            [ 0.0,  0.0,  1.0],
            [-6.0, -11.0, -6.0]  # char poly: s^3 + 6s^2 + 11s + 6 = (s+1)(s+2)(s+3)
        ], order='F', dtype=float)

        b = np.array([
            [0.0],
            [0.0],
            [1.0]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.0, 0.0]
        ], order='F', dtype=float)

        d = np.zeros((1, 1), order='F', dtype=float)

        nz, np_arr, zerosr, zerosi, polesr, polesi, gains, info = tb04cd(
            'Z', 'N', a, b, c, d, npz, tol=0.0
        )

        assert info == 0
        # Full order system should have n poles
        assert np_arr[0, 0] == n

        # Poles should be -1, -2, -3
        poles = np.sort(polesr[:np_arr[0, 0]])
        np.testing.assert_allclose(poles, [-3.0, -2.0, -1.0], rtol=1e-10)

    def test_zero_gain_zero_element(self):
        """
        Validate that zero gain implies zero transfer function element.

        When a column of B is zero, the corresponding transfer functions are zero.
        """
        n, m, p = 2, 2, 1
        npz = n

        a = np.array([
            [-1.0,  0.0],
            [ 0.0, -2.0]
        ], order='F', dtype=float)

        # First column of B is zero
        b = np.array([
            [0.0, 1.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 1.0]
        ], order='F', dtype=float)

        d = np.zeros((p, m), order='F', dtype=float)

        nz, np_arr, zerosr, zerosi, polesr, polesi, gains, info = tb04cd(
            'Z', 'N', a, b, c, d, npz, tol=0.0
        )

        assert info == 0
        # First column of transfer matrix should be zero (no poles, no zeros)
        assert np_arr[0, 0] == 0
        assert nz[0, 0] == 0
