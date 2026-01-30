import numpy as np
import pytest
from slicot import sb03md


@pytest.fixture(params=[10, 50, 100, 200], ids=lambda n: f"n={n}")
def n(request):
    return request.param


def generate_ctlyap_problem(n, seed=42):
    """Generate stable continuous-time Lyapunov problem."""
    rng = np.random.default_rng(seed)
    A = -np.eye(n) + 0.1 * rng.standard_normal((n, n))
    A = np.asfortranarray(A)
    C_half = rng.standard_normal((n, n))
    C = np.asfortranarray(C_half @ C_half.T + np.eye(n))
    return A, C


def generate_dtlyap_problem(n, seed=42):
    """Generate stable discrete-time Lyapunov problem."""
    rng = np.random.default_rng(seed)
    A = 0.5 * rng.standard_normal((n, n))
    A = np.asfortranarray(A)
    C_half = rng.standard_normal((n, n))
    C = np.asfortranarray(C_half @ C_half.T + np.eye(n))
    return A, C


def test_sb03md_continuous(benchmark, n):
    """Benchmark SB03MD - Continuous Lyapunov."""
    A, C = generate_ctlyap_problem(n)

    def run():
        return sb03md('C', 'X', 'N', 'N', n,
                      A.copy(order='F'), C.copy(order='F'))

    result = benchmark(run)
    assert result[-1] == 0


def test_sb03md_discrete(benchmark, n):
    """Benchmark SB03MD - Discrete Lyapunov."""
    A, C = generate_dtlyap_problem(n)

    def run():
        return sb03md('D', 'X', 'N', 'N', n,
                      A.copy(order='F'), C.copy(order='F'))

    result = benchmark(run)
    assert result[-1] == 0
