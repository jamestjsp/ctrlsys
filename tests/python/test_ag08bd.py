"""
Tests for AG08BD: Zeros and Kronecker structure of a descriptor system pencil.

AG08BD extracts from the system pencil S(lambda) = [A-lambda*E, B; C, D]
a regular pencil Af-lambda*Ef which has the finite Smith zeros of S(lambda)
as generalized eigenvalues. It also computes the orders of infinite Smith
zeros and determines the singular and infinite Kronecker structure.
"""

import numpy as np
import pytest
from ctrlsys import ag08bd


class TestAG08BDBasic:
    """Basic functionality tests using HTML doc example."""

    def test_html_doc_example(self):
        """
        Test AG08BD with the example from SLICOT HTML documentation.

        The example uses a 9x9 descriptor system with L=9, N=9, M=3, P=3.
        Expected results:
        - NFZ = 1 (one finite zero at lambda = 1.0)
        - NIZ = 2 (two infinite zeros)
        - DINFZ = 2 (max multiplicity 2)
        - Normal rank of transfer function = 2
        - Right Kronecker indices: [2]
        - Left Kronecker indices: [1]
        """
        l, n, m, p = 9, 9, 3, 3

        A = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=float, order='F')

        E = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
        ], dtype=float, order='F')

        B = np.array([
            [-1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, -1, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, -1],
            [0, 0, 0],
            [0, 0, 0],
        ], dtype=float, order='F')

        C = np.array([
            [0, 1, 1, 0, 3, 4, 0, 0, 2],
            [0, 1, 0, 0, 4, 0, 0, 2, 0],
            [0, 0, 1, 0, -1, 4, 0, -2, 2],
        ], dtype=float, order='F')

        D = np.array([
            [1, 2, -2],
            [0, -1, -2],
            [0, 0, 0],
        ], dtype=float, order='F')

        result = ag08bd('N', l, n, m, p, A, E, B, C, D, 1e-7)
        (a_out, e_out, nfz, nrank, niz, dinfz, nkror, ninfe, nkrol,
         infz, kronr, infe, kronl, info) = result

        assert info == 0, f"AG08BD returned info = {info}"

        assert nfz == 1
        assert nrank == n + 2
        assert niz == 2
        assert dinfz == 2
        assert nkror == 1
        assert nkrol == 1
        assert ninfe == 5

        np.testing.assert_array_equal(kronr[:nkror], [2])
        np.testing.assert_array_equal(kronl[:nkrol], [1])

        if nfz > 0:
            eigvals = np.linalg.eigvals(
                np.linalg.solve(e_out[:nfz, :nfz], a_out[:nfz, :nfz])
            )
            np.testing.assert_allclose(np.real(eigvals), [1.0], rtol=1e-3)

    def test_poles_only(self):
        """
        Test computing poles (M=0, P=0) from the HTML doc example.

        From the HTML doc output:
        - The number of infinite poles = 6
        - 0 infinite pole(s) of order 1
        - 3 infinite pole(s) of order 2
        - The system has no finite poles
        """
        l, n = 9, 9

        A = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=float, order='F')

        E = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
        ], dtype=float, order='F')

        B = np.zeros((l, 1), dtype=float, order='F')
        C = np.zeros((1, n), dtype=float, order='F')
        D = np.zeros((1, 1), dtype=float, order='F')

        result = ag08bd('N', l, n, 0, 0, A, E, B, C, D, 1e-7)
        (a_out, e_out, nfz, nrank, niz, dinfz, nkror, ninfe, nkrol,
         infz, kronr, infe, kronl, info) = result

        assert info == 0
        assert nfz == 0
        assert niz == 6
        assert dinfz == 2
        assert ninfe == 3
        np.testing.assert_array_equal(infe[:ninfe], [3, 3, 3])


class TestAG08BDControllability:
    """Test controllability analysis (P=0)."""

    def test_controllability_indices(self):
        """
        Test controllability indices from HTML doc example (P=0).

        From the HTML doc:
        - Right Kronecker indices of [A-lambda*E,B] are 2  2  2
        - The system (A-lambda*E,B) is completely controllable
        - No finite input decoupling zeros
        """
        l, n, m = 9, 9, 3

        A = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=float, order='F')

        E = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
        ], dtype=float, order='F')

        B = np.array([
            [-1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, -1, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, -1],
            [0, 0, 0],
            [0, 0, 0],
        ], dtype=float, order='F')

        C = np.zeros((1, n), dtype=float, order='F')
        D = np.zeros((1, m), dtype=float, order='F')

        result = ag08bd('N', l, n, m, 0, A, E, B, C, D, 1e-7)
        (a_out, e_out, nfz, nrank, niz, dinfz, nkror, ninfe, nkrol,
         infz, kronr, infe, kronl, info) = result

        assert info == 0
        assert nfz == 0
        assert niz == 0
        assert ninfe == 3
        np.testing.assert_array_equal(infe[:ninfe], [1, 1, 1])


class TestAG08BDObservability:
    """Test observability analysis (M=0)."""

    def test_observability_indices(self):
        """
        Test observability indices from HTML doc example (M=0).

        From the HTML doc:
        - The left Kronecker indices of [A-lambda*E;C] are 0  1  1
        - The system (A-lambda*E,C) has no finite output decoupling zeros
        - The number of unobservable infinite poles = 4
        """
        l, n, p = 9, 9, 3

        A = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=float, order='F')

        E = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
        ], dtype=float, order='F')

        B = np.zeros((l, 1), dtype=float, order='F')

        C = np.array([
            [0, 1, 1, 0, 3, 4, 0, 0, 2],
            [0, 1, 0, 0, 4, 0, 0, 2, 0],
            [0, 0, 1, 0, -1, 4, 0, -2, 2],
        ], dtype=float, order='F')

        D = np.zeros((p, 1), dtype=float, order='F')

        result = ag08bd('N', l, n, 0, p, A, E, B, C, D, 1e-7)
        (a_out, e_out, nfz, nrank, niz, dinfz, nkror, ninfe, nkrol,
         infz, kronr, infe, kronl, info) = result

        assert info == 0
        assert nfz == 0
        assert niz == 4
        assert nkrol == 3
        np.testing.assert_array_equal(kronl[:nkrol], [0, 1, 1])


class TestAG08BDEdgeCases:
    """Edge case tests."""

    def test_empty_system(self):
        """Test with all dimensions zero."""
        A = np.zeros((1, 1), dtype=float, order='F')
        E = np.zeros((1, 1), dtype=float, order='F')
        B = np.zeros((1, 1), dtype=float, order='F')
        C = np.zeros((1, 1), dtype=float, order='F')
        D = np.zeros((1, 1), dtype=float, order='F')

        result = ag08bd('N', 0, 0, 0, 0, A, E, B, C, D, 0.0)
        (a_out, e_out, nfz, nrank, niz, dinfz, nkror, ninfe, nkrol,
         infz, kronr, infe, kronl, info) = result

        assert info == 0
        assert nfz == 0
        assert nrank == 0
        assert niz == 0
        assert nkror == 0
        assert nkrol == 0

    def test_standard_state_space(self):
        """
        Test with E = I (standard state-space system).

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        l, n, m, p = 3, 3, 2, 2

        A = np.random.randn(l, n).astype(float, order='F')
        E = np.eye(l, n, dtype=float, order='F')
        B = np.random.randn(l, m).astype(float, order='F')
        C = np.random.randn(p, n).astype(float, order='F')
        D = np.random.randn(p, m).astype(float, order='F')

        result = ag08bd('N', l, n, m, p, A, E, B, C, D, 0.0)
        (a_out, e_out, nfz, nrank, niz, dinfz, nkror, ninfe, nkrol,
         infz, kronr, infe, kronl, info) = result

        assert info == 0
        assert niz == 0


class TestAG08BDEquilibration:
    """Test with equilibration (scaling)."""

    def test_with_scaling(self):
        """
        Test AG08BD with equilibration (EQUIL='S').

        Uses the HTML doc example with scaling enabled.
        """
        l, n, m, p = 9, 9, 3, 3

        A = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=float, order='F')

        E = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
        ], dtype=float, order='F')

        B = np.array([
            [-1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, -1, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, -1],
            [0, 0, 0],
            [0, 0, 0],
        ], dtype=float, order='F')

        C = np.array([
            [0, 1, 1, 0, 3, 4, 0, 0, 2],
            [0, 1, 0, 0, 4, 0, 0, 2, 0],
            [0, 0, 1, 0, -1, 4, 0, -2, 2],
        ], dtype=float, order='F')

        D = np.array([
            [1, 2, -2],
            [0, -1, -2],
            [0, 0, 0],
        ], dtype=float, order='F')

        result = ag08bd('S', l, n, m, p, A, E, B, C, D, 1e-7)
        (a_out, e_out, nfz, nrank, niz, dinfz, nkror, ninfe, nkrol,
         infz, kronr, infe, kronl, info) = result

        assert info == 0
        assert nfz == 1
        assert niz == 2


class TestAG08BDErrorHandling:
    """Test error handling."""

    def test_invalid_equil(self):
        """Test that invalid EQUIL parameter returns error."""
        l, n, m, p = 2, 2, 1, 1
        A = np.eye(l, n, dtype=float, order='F')
        E = np.eye(l, n, dtype=float, order='F')
        B = np.ones((l, m), dtype=float, order='F')
        C = np.ones((p, n), dtype=float, order='F')
        D = np.ones((p, m), dtype=float, order='F')

        result = ag08bd('X', l, n, m, p, A, E, B, C, D, 0.0)
        info = result[-1]
        assert info == -1

    def test_invalid_tol(self):
        """Test that invalid TOL parameter returns error."""
        l, n, m, p = 2, 2, 1, 1
        A = np.eye(l, n, dtype=float, order='F')
        E = np.eye(l, n, dtype=float, order='F')
        B = np.ones((l, m), dtype=float, order='F')
        C = np.ones((p, n), dtype=float, order='F')
        D = np.ones((p, m), dtype=float, order='F')

        result = ag08bd('N', l, n, m, p, A, E, B, C, D, 1.5)
        info = result[-1]
        assert info == -27


class TestAG08BDMathematicalProperties:
    """Test mathematical properties and invariants."""

    def test_infz_structure(self):
        """
        Test that INFZ correctly counts infinite elementary divisors.

        The sum of i*INFZ(i) for i=1..DINFZ should equal NIZ.
        Random seed: 123 (for reproducibility)
        """
        l, n, m, p = 9, 9, 3, 3

        A = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=float, order='F')

        E = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
        ], dtype=float, order='F')

        B = np.array([
            [-1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, -1, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, -1],
            [0, 0, 0],
            [0, 0, 0],
        ], dtype=float, order='F')

        C = np.array([
            [0, 1, 1, 0, 3, 4, 0, 0, 2],
            [0, 1, 0, 0, 4, 0, 0, 2, 0],
            [0, 0, 1, 0, -1, 4, 0, -2, 2],
        ], dtype=float, order='F')

        D = np.array([
            [1, 2, -2],
            [0, -1, -2],
            [0, 0, 0],
        ], dtype=float, order='F')

        result = ag08bd('N', l, n, m, p, A, E, B, C, D, 1e-7)
        (a_out, e_out, nfz, nrank, niz, dinfz, nkror, ninfe, nkrol,
         infz, kronr, infe, kronl, info) = result

        assert info == 0

        niz_computed = 0
        for i in range(dinfz):
            niz_computed += (i + 1) * infz[i]
        assert niz_computed == niz, f"NIZ mismatch: computed {niz_computed}, expected {niz}"

    def test_reduced_pencil_nonsingular(self):
        """
        Test that the reduced pencil Ef is nonsingular when NFZ > 0.

        For a regular pencil, Ef should be nonsingular (full rank).
        """
        l, n, m, p = 9, 9, 3, 3

        A = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=float, order='F')

        E = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
        ], dtype=float, order='F')

        B = np.array([
            [-1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, -1, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, -1],
            [0, 0, 0],
            [0, 0, 0],
        ], dtype=float, order='F')

        C = np.array([
            [0, 1, 1, 0, 3, 4, 0, 0, 2],
            [0, 1, 0, 0, 4, 0, 0, 2, 0],
            [0, 0, 1, 0, -1, 4, 0, -2, 2],
        ], dtype=float, order='F')

        D = np.array([
            [1, 2, -2],
            [0, -1, -2],
            [0, 0, 0],
        ], dtype=float, order='F')

        result = ag08bd('N', l, n, m, p, A, E, B, C, D, 1e-7)
        (a_out, e_out, nfz, nrank, niz, dinfz, nkror, ninfe, nkrol,
         infz, kronr, infe, kronl, info) = result

        assert info == 0
        assert nfz == 1

        Ef = e_out[:nfz, :nfz]
        assert np.linalg.matrix_rank(Ef) == nfz, "Ef should be nonsingular"
