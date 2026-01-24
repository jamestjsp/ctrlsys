#!/usr/bin/env python3
"""
pytest configuration for SLICOT tests.

KNOWN ISSUE (macOS):
When running the full test suite (2600+ tests) on macOS, pytest may crash
with "Fatal Python error: Segmentation fault" or "Aborted" after ~750 tests. 
This is caused by a bug in macOS Accelerate framework (BLAS/LAPACK) that 
corrupts internal state or stack after many LAPACK calls.

Workaround: Run tests in batches.
A script is provided: `tools/run_tests_batched.py`
Usage: python3 tools/run_tests_batched.py

Note: Mixing OpenBLAS with Accelerate-linked NumPy causes immediate crashes,
so switching BLAS is not a viable solution unless NumPy is also rebuilt.

ASAN on Linux passes all tests - no memory issues detected.
"""
import ctypes
import os
import sys

import pytest

@pytest.fixture(autouse=True)
def flush_c_stderr():
    """Flush C stdio buffers after each test.

    This is a partial mitigation for macOS Accelerate issues - it flushes
    C-level stdio buffers after each test to reduce accumulation effects.
    Does not fully prevent the crash but may delay it.
    """
    yield
    if sys.platform == 'darwin':
        try:
            libc = ctypes.CDLL(None)
            libc.fflush(None)
        except Exception:
            pass


@pytest.fixture
def suppress_xerbla():
    """
    Context manager fixture to suppress XERBLA stderr output.

    XERBLA is a LAPACK error handler that prints to stderr when called
    with invalid parameters. This fixture redirects stderr to /dev/null
    during test execution to prevent delayed output appearing after
    test completion (especially on macOS).

    Usage:
        def test_invalid_param(suppress_xerbla):
            with suppress_xerbla():
                result = slicot_function('invalid', ...)
            assert result.info == -1
    """
    from contextlib import contextmanager

    @contextmanager
    def _suppress():
        stderr_fd = sys.stderr.fileno()
        saved_stderr = os.dup(stderr_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stderr_fd)
        try:
            yield
        finally:
            # Flush C-level stderr before restoring
            try:
                libc = ctypes.CDLL(None)
                libc.fflush(None)
            except Exception:
                pass
            os.dup2(saved_stderr, stderr_fd)
            os.close(saved_stderr)
            os.close(devnull)

    return _suppress
