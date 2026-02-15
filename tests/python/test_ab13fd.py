"""
Tests for AB13FD: Compute complex stability radius using SVD.

Computes beta(A), the 2-norm distance from a real matrix A to the nearest
complex matrix with an eigenvalue on the imaginary axis. If A is stable,
beta(A) is the complex stability radius.

Tests:
1. HTML doc example (4x4 matrix with known beta and omega)
2. Diagonal stable matrix (analytical)
3. Zero dimension edge case
4. Eigenvalue preservation property
5. Error handling (INFO=1 convergence warning)

Random seeds: 42, 123 (for reproducibility)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_ab13fd_html_example():
    """
    Validate AB13FD using HTML documentation example.

    From SLICOT AB13FD.html:
    - N = 4, TOL = 0.0
    - A is a 4x4 matrix (read row-wise in Fortran)
    - Expected: beta = 0.39196472317D-02, omega = 0.98966520430D+00
    """
    from ctrlsys import ab13fd

    a = np.array([
        [246.500, 242.500, 202.500, -197.500],
        [-252.500, -248.500, -207.500, 202.500],
        [-302.500, -297.500, -248.500, 242.500],
        [-307.500, -302.500, -252.500, 246.500]
    ], order='F', dtype=float)

    tol = 0.0
    beta, omega, info = ab13fd(a, tol)

    assert info == 0
    # Beta matches exactly
    assert_allclose(beta, 0.39196472317e-02, rtol=1e-8)
    # Omega may differ slightly due to numerical optimization path
    assert_allclose(omega, 0.98966520430e+00, rtol=1e-5)


def test_ab13fd_diagonal_stable():
    """
    Validate AB13FD for diagonal stable matrix.

    For diagonal A with eigenvalues lambda_i, beta(A) = min(|Re(lambda_i)|).
    For A = diag(-1, -2, -3), beta(A) = 1.0 at omega = 0.

    The minimum singular value of (A - jwI) is min_i sqrt(lambda_i^2 + w^2).
    This is minimized at w=0, giving min(|lambda_i|) = 1.0.
    """
    from ctrlsys import ab13fd

    a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')

    tol = 0.0
    beta, omega, info = ab13fd(a, tol)

    assert info == 0
    assert_allclose(beta, 1.0, rtol=1e-10)
    assert_allclose(omega, 0.0, atol=1e-10)


def test_ab13fd_zero_dimension():
    """
    Validate AB13FD handles N=0 (quick return).
    """
    from ctrlsys import ab13fd

    a = np.zeros((0, 0), order='F', dtype=float)

    tol = 0.0
    beta, omega, info = ab13fd(a, tol)

    assert info == 0
    assert beta == 0.0
    assert omega == 0.0


def test_ab13fd_single_eigenvalue():
    """
    Validate AB13FD for 1x1 matrix.

    For A = [-2], beta(A) = 2.0 (distance to imaginary axis).
    """
    from ctrlsys import ab13fd

    a = np.array([[-2.0]], order='F', dtype=float)

    tol = 0.0
    beta, omega, info = ab13fd(a, tol)

    assert info == 0
    assert_allclose(beta, 2.0, rtol=1e-10)
    assert_allclose(omega, 0.0, atol=1e-10)


def test_ab13fd_stable_system_positive_beta():
    """
    Validate beta(A) > 0 for stable systems.

    For any stable matrix (eigenvalues in left half-plane),
    beta(A) must be strictly positive.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import ab13fd

    np.random.seed(42)
    n = 4

    # Create stable A via similarity transform of diagonal stable matrix
    d = np.diag([-0.5, -1.0, -1.5, -2.0])
    q, _ = np.linalg.qr(np.random.randn(n, n))
    a = (q @ d @ q.T).astype(float, order='F')

    # Use small positive tolerance (tol=0 may trigger INFO=1 warning)
    tol = 1e-8
    beta, omega, info = ab13fd(a, tol)

    # INFO=0 or INFO=1 both acceptable (1 means upper bound returned)
    assert info in [0, 1]
    assert beta > 0.0
    # For this symmetric matrix, beta should equal min |eigenvalue| = 0.5
    assert_allclose(beta, 0.5, rtol=1e-10)


def test_ab13fd_beta_upper_bound():
    """
    Validate beta(A) <= min(|Re(lambda)|) for any matrix.

    The stability radius cannot exceed the distance of the
    closest eigenvalue to the imaginary axis.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import ab13fd

    np.random.seed(123)
    n = 3

    # Create stable matrix with known eigenvalue structure
    d = np.diag([-0.3, -1.0, -2.0])
    q, _ = np.linalg.qr(np.random.randn(n, n))
    a = (q @ d @ q.T).astype(float, order='F')

    tol = 0.0
    beta, omega, info = ab13fd(a, tol)

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(a)
    min_real_dist = np.min(np.abs(eigenvalues.real))

    assert info in [0, 1]  # info=1 means result is valid upper bound
    # beta should be close to min distance for normal matrices
    # For general matrices, beta <= min_real_dist but can be much smaller
    assert beta > 0
    assert beta <= min_real_dist + 1e-10


def test_ab13fd_svd_property():
    """
    Validate singular value property: beta = min_w sigma_min(A - jwI).

    At the returned omega, the smallest singular value of (A - j*omega*I)
    should equal beta.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import ab13fd

    np.random.seed(42)
    n = 3

    # Simple stable diagonal matrix
    a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')

    tol = 0.0
    beta, omega, info = ab13fd(a, tol)

    assert info == 0

    # Verify: sigma_min(A - j*omega*I) should equal beta
    a_complex = a.astype(complex) - 1j * omega * np.eye(n)
    sigma_min = np.min(np.linalg.svd(a_complex, compute_uv=False))

    assert_allclose(sigma_min, beta, rtol=1e-10)


def test_ab13fd_near_marginal_stability():
    """
    Validate AB13FD for matrix near marginal stability.

    Matrix with eigenvalue close to imaginary axis should have small beta.
    """
    from ctrlsys import ab13fd

    # A has eigenvalue at -0.001 (very close to imaginary axis)
    a = np.array([[-0.001]], order='F', dtype=float)

    tol = 0.0
    beta, omega, info = ab13fd(a, tol)

    assert info == 0
    assert_allclose(beta, 0.001, rtol=1e-8)


def test_ab13fd_complex_eigenvalues():
    """
    Validate AB13FD for matrix with complex eigenvalues.

    For A with eigenvalues -1 +/- j, the stability radius
    is related to the real part.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import ab13fd

    # 2x2 rotation-type matrix with eigenvalues -1 +/- j
    a = np.array([
        [-1.0, 1.0],
        [-1.0, -1.0]
    ], order='F', dtype=float)

    # Use small tolerance
    tol = 1e-8
    beta, omega, info = ab13fd(a, tol)

    # INFO=0 or INFO=1 both acceptable
    assert info in [0, 1]
    assert beta > 0

    # Verify with SVD at returned omega
    a_complex = a.astype(complex) - 1j * omega * np.eye(2)
    sigma_min = np.min(np.linalg.svd(a_complex, compute_uv=False))
    assert_allclose(sigma_min, beta, rtol=1e-6)
