import pytest
from ctrlsys import ab01nd, ab01od

from conftest import get_ctdsx_files, get_dtdsx_files, load_ctdsx_data, load_dtdsx_data

CTDSX_FILES = get_ctdsx_files()
DTDSX_FILES = get_dtdsx_files()


@pytest.mark.parametrize("dat_file", CTDSX_FILES, ids=lambda p: p.stem)
def test_ab01nd_ctdsx(benchmark, dat_file):
    """Benchmark AB01ND (controllability staircase) using CTDSX data files."""
    A, B, C = load_ctdsx_data(dat_file)

    def run():
        return ab01nd('N', A.copy(order='F'), B.copy(order='F'), 0.0)

    result = benchmark(run)
    assert result[-1] == 0


@pytest.mark.parametrize("dat_file", DTDSX_FILES, ids=lambda p: p.stem)
def test_ab01nd_dtdsx(benchmark, dat_file):
    """Benchmark AB01ND (controllability staircase) using DTDSX data files."""
    A, B, C = load_dtdsx_data(dat_file)

    def run():
        return ab01nd('N', A.copy(order='F'), B.copy(order='F'), 0.0)

    result = benchmark(run)
    assert result[-1] == 0


@pytest.mark.parametrize("dat_file", CTDSX_FILES, ids=lambda p: p.stem)
def test_ab01od_ctdsx(benchmark, dat_file):
    """Benchmark AB01OD (staircase form) using CTDSX data files."""
    A, B, C = load_ctdsx_data(dat_file)

    def run():
        return ab01od('A', 'N', 'N', A.copy(order='F'), B.copy(order='F'), 0.0)

    result = benchmark(run)
    assert result[-1] == 0


@pytest.mark.parametrize("dat_file", DTDSX_FILES, ids=lambda p: p.stem)
def test_ab01od_dtdsx(benchmark, dat_file):
    """Benchmark AB01OD (staircase form) using DTDSX data files."""
    A, B, C = load_dtdsx_data(dat_file)

    def run():
        return ab01od('A', 'N', 'N', A.copy(order='F'), B.copy(order='F'), 0.0)

    result = benchmark(run)
    assert result[-1] == 0
