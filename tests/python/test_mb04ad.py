"""
Tests for mb04ad - Eigenvalues of skew-Hamiltonian/Hamiltonian pencil.

MB04AD computes eigenvalues of a real N-by-N skew-Hamiltonian/Hamiltonian
pencil aS - bH with S = T Z = J Z' J' Z via generalized symplectic URV
decomposition.
"""

import numpy as np
import pytest
from ctrlsys import mb04ad


def test_mb04ad_html_example():
    """
    Test MB04AD using HTML documentation example (N=8).

    This validates basic functionality using the example from SLICOT docs.
    Tests eigenvalue computation with JOB='T' and all transformation matrices.
    """
    n = 8
    m = n // 2

    z = np.array([
        [3.1472, 4.5751, -0.7824, 1.7874, -2.2308, -0.6126, 2.0936, 4.5974],
        [4.0579, 4.6489, 4.1574, 2.5774, -4.5383, -1.1844, 2.5469, -1.5961],
        [-3.7301, -3.4239, 2.9221, 2.4313, -4.0287, 2.6552, -2.2397, 0.8527],
        [4.1338, 4.7059, 4.5949, -1.0777, 3.2346, 2.9520, 1.7970, -2.7619],
        [1.3236, 4.5717, 1.5574, 1.5548, 1.9483, -3.1313, 1.5510, 2.5127],
        [-4.0246, -0.1462, -4.6429, -3.2881, -1.8290, -0.1024, -3.3739, -2.4490],
        [-2.2150, 3.0028, 3.4913, 2.0605, 4.5022, -0.5441, -3.8100, 0.0596],
        [0.4688, -3.5811, 4.3399, -4.6817, -4.6555, 1.4631, -0.0164, 1.9908],
    ], dtype=float, order='F')

    h = np.array([
        [3.9090, -3.5071, 3.1428, -3.0340, -1.4834, 3.7401, -0.1715, 0.4026],
        [4.5929, -2.4249, -2.5648, -2.4892, 3.7401, -2.1416, 1.6251, 2.6645],
        [0.4722, 3.4072, 4.2926, 1.1604, -0.1715, 1.6251, -4.2415, -0.0602],
        [-3.6138, -2.4572, -1.5002, -0.2671, 0.4026, 2.6645, -0.0602, -3.7009],
        [0.6882, -1.8421, -4.1122, 0.1317, -3.9090, -4.5929, -0.4722, 3.6138],
        [-1.8421, 2.9428, -0.4340, 1.3834, 3.5071, 2.4249, -3.4072, 2.4572],
        [-4.1122, -0.4340, -2.3703, 0.5231, -3.1428, 2.5648, -4.2926, 1.5002],
        [0.1317, 1.3834, 0.5231, -4.1618, 3.0340, 2.4892, -1.1604, 0.2671],
    ], dtype=float, order='F')

    result = mb04ad('T', 'I', 'I', 'I', 'I', z, h)

    t_out, z_out, h_out, q1, q2, u11, u12, u21, u22, alphar, alphai, beta, info = result

    assert info == 0, f"Expected info=0, got {info}"

    assert alphar.shape == (m,)
    assert alphai.shape == (m,)
    assert beta.shape == (m,)

    alphar_expected = np.array([0.0000, 0.7122, 0.0000, 0.7450])
    alphai_expected = np.array([0.7540, 0.0000, 0.7465, 0.0000])
    beta_expected = np.array([4.0000, 4.0000, 8.0000, 16.0000])

    np.testing.assert_allclose(alphar, alphar_expected, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(alphai, alphai_expected, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(beta, beta_expected, rtol=1e-3, atol=1e-3)

    assert q1.shape == (n, n)
    assert q2.shape == (n, n)
    np.testing.assert_allclose(q1 @ q1.T, np.eye(n), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(q2 @ q2.T, np.eye(n), rtol=1e-10, atol=1e-10)

    assert u11.shape == (m, m)
    assert u12.shape == (m, m)
    assert u21.shape == (m, m)
    assert u22.shape == (m, m)


def test_mb04ad_eigenvalues_only():
    """
    Test MB04AD with JOB='E' (eigenvalues only, no transformation matrices).

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 4
    m = n // 2

    z = np.random.randn(n, n).astype(float, order='F')
    h11 = np.random.randn(m, m)
    h12 = np.random.randn(m, m)
    h12 = (h12 + h12.T) / 2
    h21 = np.random.randn(m, m)
    h21 = (h21 + h21.T) / 2
    h = np.zeros((n, n), dtype=float, order='F')
    h[:m, :m] = h11
    h[:m, m:] = h12
    h[m:, :m] = h21
    h[m:, m:] = -h11.T

    result = mb04ad('E', 'N', 'N', 'N', 'N', z, h)

    t_out, z_out, h_out, q1, q2, u11, u12, u21, u22, alphar, alphai, beta, info = result

    assert info == 0, f"Expected info=0, got {info}"
    assert alphar.shape == (m,)
    assert alphai.shape == (m,)
    assert beta.shape == (m,)


def test_mb04ad_small_system():
    """
    Test MB04AD with smallest valid system (N=2).

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 2
    m = 1

    z = np.array([[1.0, 0.5], [0.3, 1.0]], dtype=float, order='F')

    h11 = np.array([[1.0]])
    h12 = np.array([[0.5]])
    h21 = np.array([[0.5]])
    h = np.array([
        [h11[0, 0], h12[0, 0]],
        [h21[0, 0], -h11[0, 0]]
    ], dtype=float, order='F')

    result = mb04ad('T', 'I', 'I', 'I', 'I', z, h)
    t_out, z_out, h_out, q1, q2, u11, u12, u21, u22, alphar, alphai, beta, info = result

    assert info == 0, f"Expected info=0, got {info}"
    assert alphar.shape == (m,)
    assert alphai.shape == (m,)
    assert beta.shape == (m,)

    np.testing.assert_allclose(q1 @ q1.T, np.eye(n), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(q2 @ q2.T, np.eye(n), rtol=1e-10, atol=1e-10)


def test_mb04ad_zero_dimension():
    """
    Test MB04AD with N=0 (quick return case).
    """
    n = 0

    z = np.array([], dtype=float, order='F').reshape(0, 0)
    h = np.array([], dtype=float, order='F').reshape(0, 0)

    result = mb04ad('E', 'N', 'N', 'N', 'N', z, h)
    t_out, z_out, h_out, q1, q2, u11, u12, u21, u22, alphar, alphai, beta, info = result

    assert info == 0, f"Expected info=0 for quick return, got {info}"
    assert len(alphar) == 0
    assert len(alphai) == 0
    assert len(beta) == 0


def test_mb04ad_transformation_orthogonality():
    """
    Test orthogonality of transformation matrices Q1, Q2, U1, U2.

    Property test: Q1'*Q1 = I, Q2'*Q2 = I, and symplectic structure of U1, U2.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 6
    m = n // 2

    z = np.random.randn(n, n).astype(float, order='F')
    h11 = np.random.randn(m, m)
    h12 = np.random.randn(m, m)
    h12 = (h12 + h12.T) / 2
    h21 = np.random.randn(m, m)
    h21 = (h21 + h21.T) / 2
    h = np.zeros((n, n), dtype=float, order='F')
    h[:m, :m] = h11
    h[:m, m:] = h12
    h[m:, :m] = h21
    h[m:, m:] = -h11.T

    result = mb04ad('T', 'I', 'I', 'I', 'I', z, h)
    t_out, z_out, h_out, q1, q2, u11, u12, u21, u22, alphar, alphai, beta, info = result

    assert info == 0 or info == 3, f"Unexpected info={info}"

    np.testing.assert_allclose(q1 @ q1.T, np.eye(n), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(q1.T @ q1, np.eye(n), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(q2 @ q2.T, np.eye(n), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(q2.T @ q2, np.eye(n), rtol=1e-10, atol=1e-10)

    u1 = np.block([[u11, u12], [-u12, u11]])
    np.testing.assert_allclose(u1 @ u1.T, np.eye(n), rtol=1e-10, atol=1e-10)

    u2 = np.block([[u21, u22], [-u22, u21]])
    np.testing.assert_allclose(u2 @ u2.T, np.eye(n), rtol=1e-10, atol=1e-10)


def test_mb04ad_invalid_n_odd():
    """
    Test MB04AD error handling for odd N (must be even).
    """
    n = 3

    z = np.eye(n, dtype=float, order='F')
    h = np.eye(n, dtype=float, order='F')

    result = mb04ad('E', 'N', 'N', 'N', 'N', z, h)
    _, _, _, _, _, _, _, _, _, _, _, _, info = result

    assert info == -6, f"Expected info=-6 for odd N, got {info}"


def test_mb04ad_update_mode():
    """
    Test MB04AD with COMPQ1='U' (update existing Q matrix).

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 4
    m = n // 2

    z = np.random.randn(n, n).astype(float, order='F')

    h11 = np.random.randn(m, m)
    h12 = (np.random.randn(m, m) + np.random.randn(m, m).T) / 2
    h21 = (np.random.randn(m, m) + np.random.randn(m, m).T) / 2
    h = np.zeros((n, n), dtype=float, order='F')
    h[:m, :m] = h11
    h[:m, m:] = h12
    h[m:, :m] = h21
    h[m:, m:] = -h11.T

    q1_init = np.eye(n, dtype=float, order='F')
    q2_init = np.eye(n, dtype=float, order='F')
    u11_init = np.eye(m, dtype=float, order='F')
    u12_init = np.zeros((m, m), dtype=float, order='F')
    u21_init = np.eye(m, dtype=float, order='F')
    u22_init = np.zeros((m, m), dtype=float, order='F')

    result = mb04ad('T', 'U', 'U', 'U', 'U', z, h,
                    q1=q1_init, q2=q2_init,
                    u11=u11_init, u12=u12_init,
                    u21=u21_init, u22=u22_init)

    t_out, z_out, h_out, q1, q2, u11, u12, u21, u22, alphar, alphai, beta, info = result

    assert info == 0 or info == 3, f"Unexpected info={info}"

    np.testing.assert_allclose(q1 @ q1.T, np.eye(n), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(q2 @ q2.T, np.eye(n), rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
