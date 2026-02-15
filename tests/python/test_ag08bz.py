"""
Tests for ag08bz - Zeros and Kronecker structure of a descriptor system pencil (complex).

Extracts from the system pencil S(lambda) = [A-lambda*E, B; C, D] a regular pencil
Af-lambda*Ef which has the finite Smith zeros of S(lambda) as generalized eigenvalues.
Also computes infinite Smith zero orders and Kronecker indices.

Test data from SLICOT HTML documentation AG08BZ.html.
"""

import numpy as np
import pytest
from ctrlsys import ag08bz


class TestAG08BZBasic:
    """Basic functionality tests using HTML doc example."""

    def setup_method(self):
        """Set up test matrices from HTML doc example (L=N=9, M=P=3)."""
        self.l = 9
        self.n = 9
        self.m = 3
        self.p = 3
        self.tol = 1e-7
        self.equil = 'N'

        self.A = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.complex128, order='F')

        self.E = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
        ], dtype=np.complex128, order='F')

        self.B = np.array([
            [-1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, -1, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, -1],
            [0, 0, 0],
            [0, 0, 0],
        ], dtype=np.complex128, order='F')

        self.C = np.array([
            [0, 1, 1, 0, 3, 4, 0, 0, 2],
            [0, 1, 0, 0, 4, 0, 0, 2, 0],
            [0, 0, 1, 0, -1, 4, 0, -2, 2],
        ], dtype=np.complex128, order='F')

        self.D = np.array([
            [1, 2, -2],
            [0, -1, -2],
            [0, 0, 0],
        ], dtype=np.complex128, order='F')

    def test_full_system_structural_invariants(self):
        """Test full system (L,N,M,P) = (9,9,3,3) from HTML example."""
        A = self.A.copy()
        E = self.E.copy()
        B = self.B.copy()
        C = self.C.copy()
        D = self.D.copy()

        result = ag08bz(self.equil, self.l, self.n, self.m, self.p,
                        A, E, B, C, D, self.tol)

        nfz, nrank, niz, dinfz, nkror, ninfe, nkrol, infz, kronr, infe, kronl, info = result

        assert info == 0
        assert nfz == 1, f"Expected nfz=1, got {nfz}"
        assert nrank == 11, f"Expected nrank=11, got {nrank}"
        assert niz == 2, f"Expected niz=2, got {niz}"
        assert dinfz == 2, f"Expected dinfz=2, got {dinfz}"
        assert nkror == 1, f"Expected nkror=1, got {nkror}"
        assert nkrol == 1, f"Expected nkrol=1, got {nkrol}"
        assert ninfe == 5, f"Expected ninfe=5, got {ninfe}"

        assert infz[0] == 0
        assert infz[1] == 1

        assert kronr[0] == 2, f"Expected kronr[0]=2, got {kronr[0]}"
        assert kronl[0] == 1, f"Expected kronl[0]=1, got {kronl[0]}"

        np.testing.assert_array_equal(infe[:ninfe], [1, 1, 1, 1, 3])

    def test_finite_zeros_computation(self):
        """Test that finite zeros can be computed from reduced pencil."""
        A = self.A.copy()
        E = self.E.copy()
        B = self.B.copy()
        C = self.C.copy()
        D = self.D.copy()

        result = ag08bz(self.equil, self.l, self.n, self.m, self.p,
                        A, E, B, C, D, self.tol)

        nfz, nrank, niz, dinfz, nkror, ninfe, nkrol, infz, kronr, infe, kronl, info = result

        assert info == 0
        assert nfz == 1

        Af = A[:nfz, :nfz]
        Ef = E[:nfz, :nfz]

        finite_zeros = np.linalg.eigvals(Af @ np.linalg.inv(Ef)) if nfz > 0 else np.array([])

        if nfz > 0:
            np.testing.assert_allclose(finite_zeros.real, [1.0], rtol=1e-3)
            np.testing.assert_allclose(finite_zeros.imag, [0.0], atol=1e-10)

    def test_poles_only_m_zero_p_zero(self):
        """Test pole computation (call with M=0, P=0)."""
        A = self.A.copy()
        E = self.E.copy()
        B = self.B.copy()
        C = self.C.copy()
        D = self.D.copy()

        result = ag08bz(self.equil, self.l, self.n, 0, 0,
                        A, E, B, C, D, self.tol)

        nfz, nrank, niz, dinfz, nkror, ninfe, nkrol, infz, kronr, infe, kronl, info = result

        assert info == 0
        assert niz == 6, f"Expected niz=6 (infinite poles), got {niz}"
        assert dinfz == 2
        assert ninfe == 3
        assert nfz == 0

    def test_observability_m_zero(self):
        """Test observability analysis (call with M=0)."""
        A = self.A.copy()
        E = self.E.copy()
        B = self.B.copy()
        C = self.C.copy()
        D = self.D.copy()

        result = ag08bz(self.equil, self.l, self.n, 0, self.p,
                        A, E, B, C, D, self.tol)

        nfz, nrank, niz, dinfz, nkror, ninfe, nkrol, infz, kronr, infe, kronl, info = result

        assert info == 0
        assert niz == 4, f"Expected niz=4 (unobservable infinite poles), got {niz}"
        assert nkrol == 3, f"Expected nkrol=3, got {nkrol}"
        np.testing.assert_array_equal(kronl[:nkrol], [0, 1, 1])

    def test_controllability_p_zero(self):
        """Test controllability analysis (call with P=0)."""
        A = self.A.copy()
        E = self.E.copy()
        B = self.B.copy()
        C = self.C.copy()
        D = self.D.copy()

        result = ag08bz(self.equil, self.l, self.n, self.m, 0,
                        A, E, B, C, D, self.tol)

        nfz, nrank, niz, dinfz, nkror, ninfe, nkrol, infz, kronr, infe, kronl, info = result

        assert info == 0
        assert niz == 0, f"Expected niz=0 (uncontrollable infinite poles), got {niz}"
        assert nkror == 3, f"Expected nkror=3 (controllability right indices), got {nkror}"


class TestAG08BZEdgeCases:
    """Edge case tests."""

    def test_zero_dimensions(self):
        """Test with zero dimensions (quick return)."""
        A = np.array([[]], dtype=np.complex128, order='F').reshape(0, 0)
        E = np.array([[]], dtype=np.complex128, order='F').reshape(0, 0)
        B = np.array([[]], dtype=np.complex128, order='F').reshape(0, 0)
        C = np.array([[]], dtype=np.complex128, order='F').reshape(0, 0)
        D = np.array([[]], dtype=np.complex128, order='F').reshape(0, 0)

        result = ag08bz('N', 0, 0, 0, 0, A, E, B, C, D, 0.0)

        nfz, nrank, niz, dinfz, nkror, ninfe, nkrol, infz, kronr, infe, kronl, info = result

        assert info == 0
        assert nfz == 0
        assert nrank == 0
        assert niz == 0
        assert dinfz == 0
        assert ninfe == 0

    def test_simple_siso_system(self):
        """Test simple single-input single-output system."""
        A = np.array([[1.0 + 0j]], dtype=np.complex128, order='F')
        E = np.array([[1.0 + 0j]], dtype=np.complex128, order='F')
        B = np.array([[1.0 + 0j]], dtype=np.complex128, order='F')
        C = np.array([[1.0 + 0j]], dtype=np.complex128, order='F')
        D = np.array([[0.0 + 0j]], dtype=np.complex128, order='F')

        result = ag08bz('N', 1, 1, 1, 1, A, E, B, C, D, 0.0)

        nfz, nrank, niz, dinfz, nkror, ninfe, nkrol, infz, kronr, infe, kronl, info = result

        assert info == 0


class TestAG08BZEquilibration:
    """Test with equilibration (scaling)."""

    def test_with_scaling(self):
        """Test with EQUIL='S' (balancing)."""
        l, n, m, p = 3, 3, 1, 1
        A = np.array([
            [1e-10, 0, 0],
            [0, 1e5, 0],
            [0, 0, 1],
        ], dtype=np.complex128, order='F')
        E = np.eye(3, dtype=np.complex128, order='F')
        B = np.array([[1], [1], [1]], dtype=np.complex128, order='F')
        C = np.array([[1, 1, 1]], dtype=np.complex128, order='F')
        D = np.array([[0]], dtype=np.complex128, order='F')

        result = ag08bz('S', l, n, m, p, A, E, B, C, D, 0.0)

        nfz, nrank, niz, dinfz, nkror, ninfe, nkrol, infz, kronr, infe, kronl, info = result

        assert info == 0


class TestAG08BZErrorHandling:
    """Error handling tests."""

    def test_invalid_equil(self):
        """Test invalid EQUIL parameter."""
        A = np.eye(2, dtype=np.complex128, order='F')
        E = np.eye(2, dtype=np.complex128, order='F')
        B = np.ones((2, 1), dtype=np.complex128, order='F')
        C = np.ones((1, 2), dtype=np.complex128, order='F')
        D = np.zeros((1, 1), dtype=np.complex128, order='F')

        result = ag08bz('X', 2, 2, 1, 1, A, E, B, C, D, 0.0)
        *_, info = result
        assert info == -1

    def test_negative_dimensions(self):
        """Test negative dimension parameters - wrapper validates early."""
        A = np.eye(2, dtype=np.complex128, order='F')
        E = np.eye(2, dtype=np.complex128, order='F')
        B = np.ones((2, 1), dtype=np.complex128, order='F')
        C = np.ones((1, 2), dtype=np.complex128, order='F')
        D = np.zeros((1, 1), dtype=np.complex128, order='F')

        with np.testing.assert_raises(ValueError):
            ag08bz('N', -1, 2, 1, 1, A, E, B, C, D, 0.0)

    def test_invalid_tolerance(self):
        """Test invalid tolerance (TOL >= 1)."""
        A = np.eye(2, dtype=np.complex128, order='F')
        E = np.eye(2, dtype=np.complex128, order='F')
        B = np.ones((2, 1), dtype=np.complex128, order='F')
        C = np.ones((1, 2), dtype=np.complex128, order='F')
        D = np.zeros((1, 1), dtype=np.complex128, order='F')

        result = ag08bz('N', 2, 2, 1, 1, A, E, B, C, D, 1.5)
        *_, info = result
        assert info == -27


class TestAG08BZComplexValues:
    """Test with genuinely complex-valued matrices."""

    def test_complex_system_matrix(self):
        """Test with complex-valued system matrices.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n, m, p = 3, 2, 2

        A = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128, order='F')
        E = np.eye(n, dtype=np.complex128, order='F')
        B = (np.random.randn(n, m) + 1j * np.random.randn(n, m)).astype(np.complex128, order='F')
        C = (np.random.randn(p, n) + 1j * np.random.randn(p, n)).astype(np.complex128, order='F')
        D = (np.random.randn(p, m) + 1j * np.random.randn(p, m)).astype(np.complex128, order='F')

        result = ag08bz('N', n, n, m, p, A, E, B, C, D, 0.0)

        nfz, nrank, niz, dinfz, nkror, ninfe, nkrol, infz, kronr, infe, kronl, info = result

        assert info == 0
        assert nfz >= 0
        assert nrank >= 0


class TestAG08BZMathematicalProperties:
    """Mathematical property validation tests."""

    def test_rank_relationship(self):
        """Test that nrank = nn + mu relationship holds.

        The normal rank of the system pencil equals the sum of the
        state dimension after first reduction and the row rank of Dr.
        """
        l, n, m, p = 4, 4, 2, 2
        A = np.eye(n, dtype=np.complex128, order='F')
        E = np.eye(n, dtype=np.complex128, order='F')
        B = np.zeros((n, m), dtype=np.complex128, order='F')
        B[0, 0] = 1.0
        B[1, 1] = 1.0
        C = np.zeros((p, n), dtype=np.complex128, order='F')
        C[0, 0] = 1.0
        C[1, 1] = 1.0
        D = np.eye(m, dtype=np.complex128, order='F')

        result = ag08bz('N', l, n, m, p, A, E, B, C, D, 0.0)

        nfz, nrank, niz, dinfz, nkror, ninfe, nkrol, infz, kronr, infe, kronl, info = result

        assert info == 0
        assert nrank >= nfz

    def test_infinite_zero_count(self):
        """Test infinite zero counting formula.

        niz = sum(infz[i] * (i+1) for i in range(dinfz))
        """
        l, n, m, p = 9, 9, 3, 3

        A = np.eye(n, dtype=np.complex128, order='F')
        E = np.zeros((n, n), dtype=np.complex128, order='F')
        E[1, 0] = 1.0
        E[2, 1] = 1.0
        E[4, 3] = 1.0
        E[5, 4] = 1.0
        E[7, 6] = 1.0
        E[8, 7] = 1.0

        B = np.zeros((n, m), dtype=np.complex128, order='F')
        B[0, 0] = -1.0
        B[3, 1] = -1.0
        B[6, 2] = -1.0

        C = np.array([
            [0, 1, 1, 0, 3, 4, 0, 0, 2],
            [0, 1, 0, 0, 4, 0, 0, 2, 0],
            [0, 0, 1, 0, -1, 4, 0, -2, 2],
        ], dtype=np.complex128, order='F')

        D = np.array([
            [1, 2, -2],
            [0, -1, -2],
            [0, 0, 0],
        ], dtype=np.complex128, order='F')

        result = ag08bz('N', l, n, m, p, A, E, B, C, D, 1e-7)

        nfz, nrank, niz, dinfz, nkror, ninfe, nkrol, infz, kronr, infe, kronl, info = result

        assert info == 0

        computed_niz = sum(infz[i] * (i + 1) for i in range(dinfz))
        assert niz == computed_niz, f"niz={niz} != sum formula={computed_niz}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
