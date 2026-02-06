#!/usr/bin/env python3
"""pytest configuration for SLICOT tests."""
import ctypes
import os
import sys

import pytest

# Add Python package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python'))


@pytest.fixture
def suppress_xerbla():
    """
    Context manager fixture to suppress XERBLA stderr output.

    XERBLA is a LAPACK error handler that prints to stderr when called
    with invalid parameters. This fixture redirects stderr to /dev/null
    during test execution to prevent delayed output appearing after
    test completion.

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
            try:
                libc = ctypes.CDLL(None)
                libc.fflush(None)
            except Exception:
                pass
            os.dup2(saved_stderr, stderr_fd)
            os.close(saved_stderr)
            os.close(devnull)

    return _suppress
