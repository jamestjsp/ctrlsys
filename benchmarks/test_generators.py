import numpy as np
from ctrlsys import bb01ad, bb02ad, bb03ad, bb04ad


def test_bb01ad_carex(benchmark):
    """Benchmark BB01AD - CAREX generator (continuous-time ARE examples)."""
    nr = [4, 1]
    dpar = np.array([1.5, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    ipar = np.array([10, 0, 0, 0], dtype=np.int32)
    bpar = np.array([True, True, True, True, True, True], dtype=bool)

    def run():
        return bb01ad('N', nr, dpar.copy(), ipar.copy(), bpar.copy())

    result = benchmark(run)
    assert result[-1] == 0


def test_bb02ad_darex(benchmark):
    """Benchmark BB02AD - DAREX generator (discrete-time ARE examples)."""
    nr = np.array([4, 1], dtype=np.int32)
    dpar = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    ipar = np.array([10, 0, 0], dtype=np.int32)
    bpar = np.array([True, True, True, True, True, True, False], dtype=bool)

    def run():
        return bb02ad('N', nr, dpar.copy(), ipar.copy(), bpar.copy())

    result = benchmark(run)
    assert result[-1] == 0


def test_bb03ad_ctlex(benchmark):
    """Benchmark BB03AD - CTLEX generator (continuous-time Lyapunov examples)."""
    nr = np.array([4, 1], dtype=np.int32)
    dpar = np.array([1.5, 1.5], dtype=np.float64)
    ipar = np.array([10], dtype=np.int32)

    def run():
        return bb03ad('N', nr, dpar.copy(), ipar.copy())

    result = benchmark(run)
    assert result[-1] == 0


def test_bb04ad_dtlex(benchmark):
    """Benchmark BB04AD - DTLEX generator (discrete-time Lyapunov examples)."""
    nr = np.array([4, 1], dtype=np.int32)
    dpar = np.array([1.5, 1.5], dtype=np.float64)
    ipar = np.array([10], dtype=np.int32)

    def run():
        return bb04ad('N', nr, dpar.copy(), ipar.copy())

    result = benchmark(run)
    assert result[-1] == 0
