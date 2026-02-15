"""
Tests for ab13id - Check properness of transfer function matrix.

AB13ID checks whether the transfer function G(lambda) = C*(lambda*E - A)^(-1)*B
of a descriptor system is proper. Optionally reduces the system.
"""

import numpy as np
import pytest


def test_ab13id_html_example_improper():
    """
    Test from SLICOT HTML documentation example.

    System: 9x9 descriptor system with 2 inputs and 2 outputs.
    Result: improper transfer function, NR=7, RANKE=5.
    """
    from ctrlsys import ab13id

    n, m, p = 9, 2, 2

    # A matrix (9x9) - read row-by-row from HTML example
    a = np.array([
        [-2, -3, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -2, -3, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
    ], dtype=float, order='F')

    # E matrix (9x9)
    e = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
    ], dtype=float, order='F')

    # B matrix (9x2)
    b = np.array([
        [1, 0],
        [0, 0],
        [0, 1],
        [0, 0],
        [-1, 0],
        [0, 0],
        [0, -1],
        [0, 0],
        [0, 0],
    ], dtype=float, order='F')

    # C matrix (2x9)
    c = np.array([
        [1, 0, 1, -3, 0, 1, 0, 2, 0],
        [0, 1, 1, 3, 0, 1, 0, 0, 1],
    ], dtype=float, order='F')

    # Options from example: R I N N N U
    jobsys = 'R'  # Reduce
    jobeig = 'I'  # Remove infinite eigenvalues only
    equil = 'N'   # No scaling
    cksing = 'N'  # No singularity check
    restor = 'N'  # No restore
    update = 'U'  # Update matrices

    tol = np.array([0.0, 0.0, 0.0], dtype=float)

    is_proper, nr, ranke, a_out, e_out, b_out, c_out, iwarn, info = ab13id(
        jobsys, jobeig, equil, cksing, restor, update,
        a, e, b, c, tol
    )

    assert info == 0
    assert is_proper == False
    assert nr == 7
    assert ranke == 5

    assert ranke < nr


def test_ab13id_proper_system():
    """
    Test with a proper descriptor system.

    A simple proper system: standard state-space with E=I.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import ab13id

    np.random.seed(42)
    n, m, p = 3, 1, 1

    # Simple stable system with E = I (proper by construction)
    a = np.array([
        [-1.0, 0.0, 0.0],
        [0.0, -2.0, 0.0],
        [0.0, 0.0, -3.0],
    ], dtype=float, order='F')

    e = np.eye(n, dtype=float, order='F')

    b = np.array([[1.0], [0.0], [0.0]], dtype=float, order='F')
    c = np.array([[1.0, 0.0, 0.0]], dtype=float, order='F')

    tol = np.array([0.0, 0.0, 0.0], dtype=float)

    is_proper, nr, ranke, a_out, e_out, b_out, c_out, iwarn, info = ab13id(
        'N', 'I', 'N', 'N', 'N', 'N',
        a, e, b, c, tol
    )

    assert info == 0
    assert is_proper == True
    assert nr == n
    assert ranke == n

    assert np.linalg.matrix_rank(e) == n

    gain_large_w = np.linalg.norm(
        c @ np.linalg.solve(1j * 1e6 * e - a, b))
    assert gain_large_w < 1e-3


def test_ab13id_zero_dimensions():
    """
    Test with zero system dimensions (quick return).
    """
    from ctrlsys import ab13id

    n, m, p = 0, 0, 0

    a = np.zeros((1, 1), dtype=float, order='F')  # Placeholder (won't be used)
    e = np.zeros((1, 1), dtype=float, order='F')
    b = np.zeros((1, 1), dtype=float, order='F')
    c = np.zeros((1, 1), dtype=float, order='F')

    tol = np.array([0.0, 0.0, 0.0], dtype=float)

    # N=0 case: should return proper with nr=0, ranke=0
    is_proper, nr, ranke, a_out, e_out, b_out, c_out, iwarn, info = ab13id(
        'N', 'I', 'N', 'N', 'N', 'N',
        a, e, b, c, tol, n=0, m=0, p=0
    )

    assert info == 0
    assert is_proper == True  # Empty system is proper
    assert nr == 0
    assert ranke == 0


def test_ab13id_invalid_parameter():
    """
    Test error handling for invalid parameters.
    """
    from ctrlsys import ab13id

    n, m, p = 3, 1, 1

    a = np.zeros((n, n), dtype=float, order='F')
    e = np.eye(n, dtype=float, order='F')
    b = np.zeros((n, m), dtype=float, order='F')
    c = np.zeros((p, n), dtype=float, order='F')

    # Invalid tol(1) >= 1
    tol = np.array([2.0, 0.0, 0.0], dtype=float)

    is_proper, nr, ranke, a_out, e_out, b_out, c_out, iwarn, info = ab13id(
        'N', 'I', 'N', 'N', 'N', 'N',
        a, e, b, c, tol
    )

    assert info == -20  # TOL error


def test_ab13id_remove_all_eigenvalues():
    """
    Test with JOBEIG='A' to remove all uncontrollable/unobservable eigenvalues.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import ab13id

    np.random.seed(123)
    n, m, p = 4, 1, 1

    # Create a diagonal A matrix
    a = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(float, order='F')
    e = np.eye(n, dtype=float, order='F')
    b = np.array([[1.0], [1.0], [1.0], [1.0]], dtype=float, order='F')
    c = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=float, order='F')

    tol = np.array([0.0, 0.0, 0.0], dtype=float)

    is_proper, nr, ranke, a_out, e_out, b_out, c_out, iwarn, info = ab13id(
        'R', 'A', 'N', 'N', 'N', 'N',
        a.copy(order='F'), e.copy(order='F'),
        b.copy(order='F'), c.copy(order='F'), tol
    )

    assert info == 0
    assert is_proper == True
    assert ranke == nr


def test_ab13id_with_scaling():
    """
    Test with equilibration (EQUIL='S').

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import ab13id

    np.random.seed(456)
    n, m, p = 3, 2, 2

    # Create system matrices with varying scales
    a = np.array([
        [-100.0, 0.0, 0.0],
        [0.0, -0.01, 0.0],
        [0.0, 0.0, -1.0],
    ], dtype=float, order='F')

    e = np.eye(n, dtype=float, order='F')
    b = np.array([[100.0, 0.0], [0.0, 0.01], [1.0, 1.0]], dtype=float, order='F')
    c = np.array([[1.0, 0.0, 100.0], [0.0, 0.01, 1.0]], dtype=float, order='F')

    tol = np.array([0.0, 0.0, -1.0], dtype=float)  # -1 for auto threshold

    is_proper, nr, ranke, a_out, e_out, b_out, c_out, iwarn, info = ab13id(
        'N', 'I', 'S', 'N', 'N', 'N',
        a.copy(order='F'), e.copy(order='F'),
        b.copy(order='F'), c.copy(order='F'), tol
    )

    assert info == 0
    assert is_proper == True
    assert ranke == n
