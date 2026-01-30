import re
from pathlib import Path

import numpy as np
import pytest

SIZES = [10, 50, 100, 200]
BENCHMARK_DATA_DIR = Path(__file__).parent.parent / "SLICOT-Reference" / "benchmark_data"

CAREX_DIMS = {
    103: (4, 2), 104: (8, 2), 105: (9, 3), 106: (30, 3),
    404: (21, 1),
    2091: (4, 2), 2092: (4, 2),  # Boeing 767 - use small dims, file has extra data
}

DAREX_DIMS = {
    105: (4, 2), 106: (4, 2), 107: (4, 2), 108: (4, 2),
    110: (4, 2), 111: (4, 1), 113: (5, 2),
}

CTDSX_DIMS = {
    103: (4, 2, 2), 104: (8, 2, 2), 105: (9, 3, 9), 106: (30, 3, 5),
    107: (6, 1, 1), 108: (6, 1, 6), 109: (55, 4, 1), 110: (8, 2, 2),
    203: (8, 2, 2), 206: (5, 1, 1), 304: (4, 21, 21),
    2051: (2, 1, 1), 2052: (4, 1, 1), 2053: (6, 1, 1), 2054: (8, 1, 1),
    2055: (10, 1, 1), 2056: (12, 1, 1), 2057: (20, 1, 1),
}

DTDSX_DIMS = {
    106: (4, 2, 2), 107: (4, 2, 2), 108: (4, 2, 4), 109: (5, 1, 1),
    111: (9, 3, 9), 112: (3, 1, 1),
}


@pytest.fixture(params=SIZES, ids=lambda n: f"n={n}")
def n(request):
    return request.param


def fortran_array(shape, dtype=np.float64):
    return np.zeros(shape, dtype=dtype, order='F')


def random_fortran(shape, dtype=np.float64, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    arr = rng.standard_normal(shape)
    return np.asfortranarray(arr, dtype=dtype)


def random_symmetric(n, dtype=np.float64, rng=None):
    A = random_fortran((n, n), dtype=dtype, rng=rng)
    return np.asfortranarray((A + A.T) / 2, dtype=dtype)


def random_spd(n, dtype=np.float64, rng=None):
    A = random_fortran((n, n), dtype=dtype, rng=rng)
    return np.asfortranarray(A @ A.T + np.eye(n), dtype=dtype)


def parse_fortran_double(s):
    """Parse Fortran D-notation and implied exponent formats."""
    s = s.replace('D', 'E').replace('d', 'e')
    if 'E' not in s and 'e' not in s:
        match = re.match(r'^([+-]?\d*\.?\d+)([+-]\d+)$', s)
        if match:
            s = match.group(1) + 'e' + match.group(2)
    return float(s)


def load_dat_file(filepath):
    """Load array of doubles from Fortran-formatted .dat file."""
    values = []
    with open(filepath) as f:
        for line in f:
            for token in line.split():
                values.append(parse_fortran_double(token))
    return np.array(values, dtype=np.float64)


def load_carex_data(filepath):
    """Load CAREX (BB01*.dat) benchmark: returns A, B, Q matrices."""
    filepath = Path(filepath)
    example_num = int(re.search(r'BB01(\d+)\.dat', filepath.name).group(1))
    n, m = CAREX_DIMS.get(example_num, (4, 2))

    values = load_dat_file(filepath)
    offset = 0
    A = values[offset:offset + n * n].reshape((n, n), order='F')
    offset += n * n
    B = values[offset:offset + n * m].reshape((n, m), order='F')
    offset += n * m
    if offset + n * n <= len(values):
        Q = values[offset:offset + n * n].reshape((n, n), order='F')
    else:
        Q = np.eye(n, dtype=np.float64, order='F')

    return (np.asfortranarray(A), np.asfortranarray(B), np.asfortranarray(Q))


def load_darex_data(filepath):
    """Load DAREX (BB02*.dat) benchmark: returns A, B, Q matrices."""
    filepath = Path(filepath)
    example_num = int(re.search(r'BB02(\d+)\.dat', filepath.name).group(1))
    n, m = DAREX_DIMS.get(example_num, (4, 2))

    values = load_dat_file(filepath)
    offset = 0
    A = values[offset:offset + n * n].reshape((n, n), order='F')
    offset += n * n
    B = values[offset:offset + n * m].reshape((n, m), order='F')
    offset += n * m
    if offset + n * n <= len(values):
        Q = values[offset:offset + n * n].reshape((n, n), order='F')
    else:
        Q = np.eye(n, dtype=np.float64, order='F')

    return (np.asfortranarray(A), np.asfortranarray(B), np.asfortranarray(Q))


def load_ctdsx_data(filepath):
    """Load CTDSX (BD01*.dat) benchmark: returns A, B, C matrices."""
    filepath = Path(filepath)
    example_num = int(re.search(r'BD01(\d+)\.dat', filepath.name).group(1))
    n, m, p = CTDSX_DIMS.get(example_num, (4, 2, 2))

    values = load_dat_file(filepath)
    offset = 0
    A = values[offset:offset + n * n].reshape((n, n), order='F')
    offset += n * n
    B = values[offset:offset + n * m].reshape((n, m), order='F')
    offset += n * m
    if offset + p * n <= len(values):
        C = values[offset:offset + p * n].reshape((p, n), order='F')
    else:
        C = np.eye(p, n, dtype=np.float64, order='F')

    return (np.asfortranarray(A), np.asfortranarray(B), np.asfortranarray(C))


def load_dtdsx_data(filepath):
    """Load DTDSX (BD02*.dat) benchmark: returns A, B, C matrices."""
    filepath = Path(filepath)
    example_num = int(re.search(r'BD02(\d+)\.dat', filepath.name).group(1))
    n, m, p = DTDSX_DIMS.get(example_num, (4, 2, 2))

    values = load_dat_file(filepath)
    offset = 0
    A = values[offset:offset + n * n].reshape((n, n), order='F')
    offset += n * n
    B = values[offset:offset + n * m].reshape((n, m), order='F')
    offset += n * m
    if offset + p * n <= len(values):
        C = values[offset:offset + p * n].reshape((p, n), order='F')
    else:
        C = np.eye(p, n, dtype=np.float64, order='F')

    return (np.asfortranarray(A), np.asfortranarray(B), np.asfortranarray(C))


def get_carex_files():
    """List all available CAREX benchmark files."""
    return sorted(BENCHMARK_DATA_DIR.glob("BB01*.dat"))


def get_darex_files():
    """List all available DAREX benchmark files."""
    return sorted(BENCHMARK_DATA_DIR.glob("BB02*.dat"))


def get_ctdsx_files():
    """List all available CTDSX benchmark files."""
    return sorted(BENCHMARK_DATA_DIR.glob("BD01*.dat"))


def get_dtdsx_files():
    """List all available DTDSX benchmark files."""
    return sorted(BENCHMARK_DATA_DIR.glob("BD02*.dat"))


@pytest.fixture
def carex_files():
    return get_carex_files()


@pytest.fixture
def darex_files():
    return get_darex_files()


@pytest.fixture
def ctdsx_files():
    return get_ctdsx_files()


@pytest.fixture
def dtdsx_files():
    return get_dtdsx_files()
