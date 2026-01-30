import numpy as np
import pytest
from slicot import sb02md, sb02od

from conftest import get_carex_files, get_darex_files, load_carex_data, load_darex_data

CAREX_FILES = get_carex_files()
DAREX_FILES = get_darex_files()


@pytest.mark.parametrize("dat_file", CAREX_FILES, ids=lambda p: p.stem)
def test_sb02md_carex(benchmark, dat_file):
    """Benchmark SB02MD using CAREX (continuous-time ARE) data files."""
    A, B, Q = load_carex_data(dat_file)
    n = A.shape[0]
    G = np.asfortranarray(B @ B.T)

    def run():
        return sb02md('C', 'D', 'U', 'N', 'S', n,
                      A.copy(order='F'), G.copy(order='F'), Q.copy(order='F'))

    result = benchmark(run)
    assert result[-1] <= 1


@pytest.mark.parametrize("dat_file", DAREX_FILES, ids=lambda p: p.stem)
def test_sb02md_darex(benchmark, dat_file):
    """Benchmark SB02MD using DAREX (discrete-time ARE) data files."""
    A, B, Q = load_darex_data(dat_file)
    n = A.shape[0]
    G = np.asfortranarray(B @ B.T + 0.01 * np.eye(n))

    def run():
        return sb02md('D', 'D', 'U', 'N', 'U', n,
                      A.copy(order='F'), G.copy(order='F'), Q.copy(order='F'))

    result = benchmark(run)
    assert result[-1] <= 1


@pytest.mark.parametrize("dat_file", CAREX_FILES, ids=lambda p: p.stem)
def test_sb02od_carex(benchmark, dat_file):
    """Benchmark SB02OD using CAREX data files."""
    A, B, Q = load_carex_data(dat_file)
    n, m = A.shape[0], B.shape[1]
    R = np.asfortranarray(np.eye(m))
    L = np.zeros((n, m), dtype=np.float64, order='F')

    def run():
        return sb02od('C', 'B', 'N', 'U', 'Z', 'S',
                      n, m, 0, A.copy(order='F'), B.copy(order='F'),
                      Q.copy(order='F'), R.copy(order='F'), L.copy(order='F'), 0.0)

    result = benchmark(run)
    assert result[-1] <= 1


@pytest.mark.parametrize("dat_file", DAREX_FILES, ids=lambda p: p.stem)
def test_sb02od_darex(benchmark, dat_file):
    """Benchmark SB02OD using DAREX data files."""
    A, B, Q = load_darex_data(dat_file)
    n, m = A.shape[0], B.shape[1]
    R = np.asfortranarray(np.eye(m))
    L = np.zeros((n, m), dtype=np.float64, order='F')

    def run():
        return sb02od('D', 'B', 'N', 'U', 'Z', 'S',
                      n, m, 0, A.copy(order='F'), B.copy(order='F'),
                      Q.copy(order='F'), R.copy(order='F'), L.copy(order='F'), 0.0)

    result = benchmark(run)
    assert result[-1] <= 1
