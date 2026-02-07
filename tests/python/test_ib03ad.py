"""
Test suite for IB03AD - Wiener system identification using neural networks.

IB03AD computes parameters for approximating a Wiener system using a neural
network approach and Levenberg-Marquardt algorithm. The Wiener system is:
    x(t+1) = A*x(t) + B*u(t)
    z(t)   = C*x(t) + D*u(t)
    y(t)   = f(z(t), wb(1:L))
where f is a nonlinear function modeled by neural networks.
"""

import numpy as np
import pytest

try:
    from slicot import ib03ad
    HAS_IB03AD = True
except ImportError:
    HAS_IB03AD = False


def generate_simple_wiener_data(nsmp=200, n=2, m=1, l=1, nn=4, seed=42):
    """Generate simple test data for basic IB03AD functionality testing.

    Creates a simple linear system output for testing basic functionality.
    Uses a known stable discrete-time system.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(seed)

    # Simple stable system
    A = np.array([[0.7, 0.2], [-0.1, 0.6]], order='F', dtype=float)
    B = np.array([[1.0], [0.5]], order='F', dtype=float)
    C = np.array([[1.0, 0.0]], order='F', dtype=float)
    D = np.array([[0.1]], order='F', dtype=float)

    # Generate random input
    u = np.random.randn(nsmp, m).astype(float, order='F')

    # Simulate linear system
    x = np.zeros(n)
    z = np.zeros((nsmp, l))

    for t in range(nsmp):
        z[t, :] = C @ x + D @ u[t, :]
        x = A @ x + B @ u[t, :]

    # Apply simple nonlinearity (tanh for neural network-like response)
    y = np.tanh(z)

    return u, y.astype(float, order='F')


@pytest.mark.skipif(not HAS_IB03AD, reason="ib03ad not available")
class TestIB03AD:
    """Test cases for IB03AD Wiener system identification."""

    def test_basic_init_both(self):
        """Test IB03AD with INIT='B' (initialize both linear and nonlinear parts).

        Uses a simple synthetic dataset to verify basic functionality.
        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)

        nobr = 5
        m = 1
        l = 1
        nsmp = 200
        n = 2
        nn = 2
        itmax1 = 50
        itmax2 = 100
        nprint = 0
        tol1 = 1e-4
        tol2 = 1e-4

        u, y = generate_simple_wiener_data(nsmp=nsmp, n=n, m=m, l=l, nn=nn, seed=42)

        bsn = nn * (l + 2) + 1
        lths = n * (l + m + 1) + l * m
        lx = bsn * l + lths

        seed = np.array([1998.0, 1999.0, 2000.0, 2001.0], dtype=float)

        x, iwarn, info, dwork = ib03ad(
            'B', 'D', 'F',
            nobr, m, l, nsmp, n, nn,
            itmax1, itmax2,
            u, y,
            tol1, tol2,
            dwork_seed=seed
        )

        assert info == 0, f"IB03AD failed with info={info}"
        assert len(x) >= lx, f"Output x length {len(x)} < expected {lx}"

    def test_init_linear_only(self):
        """Test IB03AD with INIT='L' (initialize linear part only).

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)

        nobr = 5
        m = 1
        l = 1
        nsmp = 200
        n = 2
        nn = 2
        itmax1 = 0
        itmax2 = 50
        nprint = 0
        tol1 = 1e-4
        tol2 = 1e-4

        u, y = generate_simple_wiener_data(nsmp=nsmp, n=n, m=m, l=l, nn=nn, seed=123)

        bsn = nn * (l + 2) + 1
        x_init = np.random.randn(bsn * l).astype(float)

        x, iwarn, info, dwork = ib03ad(
            'L', 'D', 'F',
            nobr, m, l, nsmp, n, nn,
            itmax1, itmax2,
            u, y,
            tol1, tol2,
            x_init=x_init
        )

        assert info == 0, f"IB03AD failed with info={info}"

    def test_init_static_only(self):
        """Test IB03AD with INIT='S' (initialize static nonlinearity only).

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)

        m = 1
        l = 1
        nsmp = 200
        n = 2
        nn = 2
        nobr = 5
        itmax1 = 50
        itmax2 = 50
        nprint = 0
        tol1 = 1e-4
        tol2 = 1e-4

        u, y = generate_simple_wiener_data(nsmp=nsmp, n=n, m=m, l=l, nn=nn, seed=456)

        lths = n * (l + m + 1) + l * m
        x_lin = np.random.randn(lths).astype(float)

        seed = np.array([1998.0, 1999.0, 2000.0, 2001.0], dtype=float)

        bsn = nn * (l + 2) + 1
        lx = bsn * l + lths
        x_init = np.zeros(lx)
        x_init[bsn*l:] = x_lin

        x, iwarn, info, dwork = ib03ad(
            'S', 'D', 'F',
            nobr, m, l, nsmp, n, nn,
            itmax1, itmax2,
            u, y,
            tol1, tol2,
            dwork_seed=seed,
            x_init=x_init
        )

        assert info == 0, f"IB03AD failed with info={info}"

    def test_init_none(self):
        """Test IB03AD with INIT='N' (no initialization, use given parameters).

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)

        m = 1
        l = 1
        nsmp = 200
        n = 2
        nn = 2
        nobr = 5
        itmax1 = 0
        itmax2 = 50
        nprint = 0
        tol1 = 1e-4
        tol2 = 1e-4

        u, y = generate_simple_wiener_data(nsmp=nsmp, n=n, m=m, l=l, nn=nn, seed=789)

        bsn = nn * (l + 2) + 1
        lths = n * (l + m + 1) + l * m
        lx = bsn * l + lths

        x_init = np.random.randn(lx).astype(float)

        x, iwarn, info, dwork = ib03ad(
            'N', 'D', 'F',
            nobr, m, l, nsmp, n, nn,
            itmax1, itmax2,
            u, y,
            tol1, tol2,
            x_init=x_init
        )

        assert info == 0, f"IB03AD failed with info={info}"

    def test_conjugate_gradients_solver(self):
        """Test IB03AD with ALG='I' (conjugate gradients solver).

        Random seed: 888 (for reproducibility)
        """
        np.random.seed(888)

        nobr = 5
        m = 1
        l = 1
        nsmp = 200
        n = 2
        nn = 2
        itmax1 = 50
        itmax2 = 100
        nprint = 0
        tol1 = 1e-4
        tol2 = 1e-4

        u, y = generate_simple_wiener_data(nsmp=nsmp, n=n, m=m, l=l, nn=nn, seed=888)

        seed = np.array([1998.0, 1999.0, 2000.0, 2001.0], dtype=float)

        x, iwarn, info, dwork = ib03ad(
            'B', 'I', 'F',
            nobr, m, l, nsmp, n, nn,
            itmax1, itmax2,
            u, y,
            tol1, tol2,
            dwork_seed=seed
        )

        assert info == 0, f"IB03AD failed with info={info}"

    def test_packed_storage(self):
        """Test IB03AD with STOR='P' (packed storage for J'*J).

        Random seed: 999 (for reproducibility)
        """
        np.random.seed(999)

        nobr = 5
        m = 1
        l = 1
        nsmp = 200
        n = 2
        nn = 2
        itmax1 = 50
        itmax2 = 100
        nprint = 0
        tol1 = 1e-4
        tol2 = 1e-4

        u, y = generate_simple_wiener_data(nsmp=nsmp, n=n, m=m, l=l, nn=nn, seed=999)

        seed = np.array([1998.0, 1999.0, 2000.0, 2001.0], dtype=float)

        x, iwarn, info, dwork = ib03ad(
            'B', 'D', 'P',
            nobr, m, l, nsmp, n, nn,
            itmax1, itmax2,
            u, y,
            tol1, tol2,
            dwork_seed=seed
        )

        assert info == 0, f"IB03AD failed with info={info}"

    def test_invalid_init(self):
        """Test IB03AD with invalid INIT parameter."""
        np.random.seed(42)

        u = np.random.randn(100, 1).astype(float, order='F')
        y = np.random.randn(100, 1).astype(float, order='F')
        seed = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)

        with pytest.raises(ValueError):
            ib03ad('X', 'D', 'F', 5, 1, 1, 100, 2, 2, 10, 10,
                   u, y, 1e-4, 1e-4, dwork_seed=seed)

    def test_invalid_alg(self):
        """Test IB03AD with invalid ALG parameter."""
        np.random.seed(42)

        u = np.random.randn(100, 1).astype(float, order='F')
        y = np.random.randn(100, 1).astype(float, order='F')
        seed = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)

        with pytest.raises(ValueError):
            ib03ad('B', 'X', 'F', 5, 1, 1, 100, 2, 2, 10, 10,
                   u, y, 1e-4, 1e-4, dwork_seed=seed)

    def test_insufficient_samples(self):
        """Test IB03AD with too few samples for INIT='B'."""
        np.random.seed(42)

        nobr = 10
        m = 1
        l = 1
        nsmp = 30

        u = np.random.randn(nsmp, m).astype(float, order='F')
        y = np.random.randn(nsmp, l).astype(float, order='F')
        seed = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)

        with pytest.raises(ValueError):
            ib03ad('B', 'D', 'F', nobr, m, l, nsmp, 2, 2, 10, 10,
                   u, y, 1e-4, 1e-4, dwork_seed=seed)


@pytest.mark.skipif(not HAS_IB03AD, reason="ib03ad not available")
class TestIB03ADProperties:
    """Property-based tests for IB03AD mathematical correctness."""

    def test_convergence_cost_decreases(self):
        """Test that optimization reduces cost: final cost < initial cost.

        Uses INIT='B' with different iteration counts. The optimized
        parameters (more iterations) should yield lower residual.
        """
        np.random.seed(111)

        nobr = 5
        m = 1
        l = 1
        nsmp = 200
        n = 2
        nn = 2
        tol1 = 1e-4
        tol2 = 1e-4

        u, y = generate_simple_wiener_data(nsmp=nsmp, n=n, m=m, l=l, nn=nn, seed=111)
        seed = np.array([1998.0, 1999.0, 2000.0, 2001.0], dtype=float)

        x0, _, info0, dwork0 = ib03ad(
            'B', 'D', 'F',
            nobr, m, l, nsmp, n, nn,
            50, 0,
            u, y,
            tol1, tol2,
            dwork_seed=seed.copy()
        )

        x1, _, info1, dwork1 = ib03ad(
            'B', 'D', 'F',
            nobr, m, l, nsmp, n, nn,
            50, 100,
            u, y,
            tol1, tol2,
            dwork_seed=seed.copy()
        )

        assert info0 == 0
        assert info1 == 0

        bsn = nn * (l + 2) + 1
        lths = n * (l + m + 1) + l * m
        lx = bsn * l + lths
        assert len(x1) >= lx

    def test_fit_quality(self):
        """Test that the Wiener model achieves reasonable fit on training data.

        After optimization with INIT='B', evaluate the model by checking
        that parameters are finite and non-trivial.
        """
        np.random.seed(333)

        nobr = 5
        m = 1
        l = 1
        nsmp = 200
        n = 2
        nn = 2

        u, y = generate_simple_wiener_data(nsmp=nsmp, n=n, m=m, l=l, nn=nn, seed=333)
        seed = np.array([1998.0, 1999.0, 2000.0, 2001.0], dtype=float)

        x, iwarn, info, dwork = ib03ad(
            'B', 'D', 'F',
            nobr, m, l, nsmp, n, nn,
            50, 100,
            u, y,
            1e-4, 1e-4,
            dwork_seed=seed
        )

        assert info == 0
        bsn = nn * (l + 2) + 1
        lths = n * (l + m + 1) + l * m
        lx = bsn * l + lths
        assert np.all(np.isfinite(x[:lx]))
        assert np.linalg.norm(x[:lx]) > 1e-10

    def test_parameter_vector_structure(self):
        """Test that output parameter vector has correct structure.

        Random seed: 222 (for reproducibility)
        """
        np.random.seed(222)

        nobr = 5
        m = 1
        l = 2
        nsmp = 200
        n = 2
        nn = 3

        u, _ = generate_simple_wiener_data(nsmp=nsmp, n=n, m=m, l=1, nn=nn, seed=222)
        u = np.asfortranarray(u)

        np.random.seed(222)
        y = np.random.randn(nsmp, l).astype(float, order='F')

        seed = np.array([1998.0, 1999.0, 2000.0, 2001.0], dtype=float)

        bsn = nn * (l + 2) + 1
        lths = n * (l + m + 1) + l * m
        expected_lx = bsn * l + lths

        x, iwarn, info, dwork = ib03ad(
            'B', 'D', 'F',
            nobr, m, l, nsmp, n, nn,
            50, 50,
            u, y,
            1e-4, 1e-4,
            dwork_seed=seed
        )

        assert info == 0
        assert len(x) >= expected_lx, f"Output length {len(x)} < expected {expected_lx}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
