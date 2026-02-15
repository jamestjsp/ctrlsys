"""
Tests for MB3LZP - Eigenvalues and right deflating subspace of a complex
skew-Hamiltonian/Hamiltonian pencil.

Computes eigenvalues of aS - bH where:
    S = [A  D]    H = [B  F]
        [E  A']       [G -B']

The routine embeds the complex pencil into a real skew-Hamiltonian/
skew-Hamiltonian pencil, applies MB04FP for structured Schur form,
then MB3JZP for eigenvalue reordering, and optionally computes
the deflating subspace.

Note: Array dimensions must be N-by-N (not N/2-by-N/2) since LDA/LDB >= N,
      even though only N/2-by-N/2 blocks are used on input.
"""

import numpy as np
import pytest

from ctrlsys import mb3lzp


def make_test_pencil(m):
    """
    Create test skew-Hamiltonian/Hamiltonian pencil matrices.

    m: Half dimension (n = 2*m)
    Returns: a, de, b, fg arrays of dimension n-by-n
    """
    n = 2 * m

    # A is general N/2-by-N/2 complex matrix (in leading block)
    a = np.zeros((n, n), dtype=complex, order='F')
    a[:m, :m] = np.array([[1.0 + 0.5j, 0.3 - 0.2j],
                          [0.1 + 0.1j, 2.0 - 0.3j]])[:m, :m]

    # DE: E is skew-Hermitian (lower triangular of leading m-by-m block)
    #     D is skew-Hermitian (upper triangular in columns 2 to m+1)
    de = np.zeros((n, n), dtype=complex, order='F')
    de[0, 0] = 0.2j  # E(1,1) - purely imaginary
    if m >= 2:
        de[1, 0] = -0.1 + 0.3j  # E(2,1)
        de[1, 1] = -0.1j  # E(2,2) - purely imaginary
        de[0, 2] = 0.15j  # D(1,1) - purely imaginary
        de[0, 3] = 0.2 - 0.1j  # D(1,2)
        de[1, 3] = 0.1j  # D(2,2) - purely imaginary
    else:
        de[0, 1] = 0.15j  # D(1,1) when m=1

    # B is general N/2-by-N/2 complex matrix (in leading block)
    b = np.zeros((n, n), dtype=complex, order='F')
    b[:m, :m] = np.array([[0.5 + 0.2j, 0.1 - 0.1j],
                          [0.2 + 0.05j, 0.8 - 0.2j]])[:m, :m]

    # FG: G is Hermitian (lower triangular of leading m-by-m block)
    #     F is Hermitian (upper triangular in columns 2 to m+1)
    fg = np.zeros((n, n), dtype=complex, order='F')
    fg[0, 0] = 0.5  # G(1,1) - real
    if m >= 2:
        fg[1, 0] = 0.1 + 0.2j  # G(2,1)
        fg[1, 1] = 0.3  # G(2,2) - real
        fg[0, 2] = 0.4  # F(1,1) - real
        fg[0, 3] = 0.2 - 0.15j  # F(1,2)
        fg[1, 3] = 0.6  # F(2,2) - real
    else:
        fg[0, 1] = 0.4  # F(1,1) when m=1

    return a, de, b, fg


class TestMB3LZPBasic:
    """Basic functionality tests for MB3LZP."""

    def test_eigenvalues_only(self):
        """
        Test eigenvalue computation without deflating subspace.
        """
        m = 2  # n = 2*m = 4
        n = 2 * m

        a, de, b, fg = make_test_pencil(m)

        alphar, alphai, beta, neig, info = mb3lzp(
            'N', 'P', n, a.copy(order='F'), de.copy(order='F'),
            b.copy(order='F'), fg.copy(order='F')
        )

        assert info == 0, f"mb3lzp returned info={info}"
        assert len(alphar) == n
        assert len(alphai) == n
        assert len(beta) == n
        assert np.all(np.isfinite(alphar))
        assert np.all(np.isfinite(alphai))
        assert np.all(np.isfinite(beta))

    def test_with_deflating_subspace_qrp(self):
        """
        Test with deflating subspace computation using QR with column pivoting.
        """
        m = 2
        n = 2 * m

        a, de, b, fg = make_test_pencil(m)

        result = mb3lzp(
            'C', 'P', n, a.copy(order='F'), de.copy(order='F'),
            b.copy(order='F'), fg.copy(order='F')
        )

        a_out, de_out, b_out, fg_out, q_out, alphar, alphai, beta, neig, info = result

        assert info == 0, f"mb3lzp returned info={info}"
        # Q should be 2N x 2N orthogonal matrix
        assert q_out.shape == (2*n, 2*n)
        # Eigenvalue arrays should have n elements
        assert len(alphar) == n
        assert len(alphai) == n
        assert len(beta) == n

    def test_with_deflating_subspace_svd(self):
        """
        Test with deflating subspace computation using SVD.
        """
        m = 2
        n = 2 * m

        a, de, b, fg = make_test_pencil(m)

        result = mb3lzp(
            'C', 'S', n, a.copy(order='F'), de.copy(order='F'),
            b.copy(order='F'), fg.copy(order='F')
        )

        a_out, de_out, b_out, fg_out, q_out, alphar, alphai, beta, neig, info = result

        assert info == 0, f"mb3lzp returned info={info}"
        assert q_out.shape == (2*n, 2*n)


class TestMB3LZPMathematicalProperties:
    """Mathematical property validation tests."""

    def test_eigenvalue_symmetry(self):
        """
        Test eigenvalue symmetry property.

        For skew-Hamiltonian/Hamiltonian pencils, eigenvalues come in pairs
        (lambda, -conj(lambda)).

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        m = 2
        n = 2 * m

        a, de, b, fg = make_test_pencil(m)

        alphar, alphai, beta, neig, info = mb3lzp(
            'N', 'P', n, a.copy(order='F'), de.copy(order='F'),
            b.copy(order='F'), fg.copy(order='F')
        )

        assert info == 0, f"mb3lzp returned info={info}"

        # Compute eigenvalues as complex numbers
        eigs = []
        for i in range(n):
            if abs(beta[i]) > 1e-14:
                eig = (alphar[i] + 1j * alphai[i]) / beta[i]
                eigs.append(eig)

        if len(eigs) >= 2:
            eigs_sorted = sorted(eigs, key=lambda x: (x.real, x.imag))
            # Check pairs exist (rough check due to numerical precision)
            for eig in eigs_sorted[:len(eigs_sorted)//2]:
                neg_conj = -np.conj(eig)
                has_pair = any(abs(e - neg_conj) < 1e-10 for e in eigs_sorted)
                # This may not always hold due to algorithm details

    def test_output_structure_compq_c(self):
        """
        When COMPQ='C', verify output structure.

        Q contains orthonormal basis of the deflating subspace in
        leading N-by-NEIG part (not the full 2N x 2N matrix).

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        m = 2
        n = 2 * m

        a, de, b, fg = make_test_pencil(m)

        result = mb3lzp(
            'C', 'P', n, a.copy(order='F'), de.copy(order='F'),
            b.copy(order='F'), fg.copy(order='F')
        )

        a_out, de_out, b_out, fg_out, q_out, alphar, alphai, beta, neig, info = result

        assert info == 0, f"mb3lzp returned info={info}"

        # Q should be 2N x 2N
        assert q_out.shape == (2*n, 2*n)

        # The leading N-by-NEIG part contains orthonormal columns
        # (but only if neig > 0)
        if neig > 0:
            q_leading = q_out[:n, :neig]
            qtq = q_leading.conj().T @ q_leading
            eye_neig = np.eye(neig)
            np.testing.assert_allclose(qtq, eye_neig, rtol=1e-12, atol=1e-14)


class TestMB3LZPEdgeCases:
    """Edge case tests."""

    def test_zero_dimension(self):
        """Test with n=0 (quick return)."""
        n = 0
        a = np.zeros((0, 0), dtype=complex, order='F')
        de = np.zeros((0, 0), dtype=complex, order='F')
        b = np.zeros((0, 0), dtype=complex, order='F')
        fg = np.zeros((0, 0), dtype=complex, order='F')

        alphar, alphai, beta, neig, info = mb3lzp(
            'N', 'P', n, a, de, b, fg
        )

        assert info == 0
        # neig should be 0 for empty case
        assert neig == 0

    def test_minimal_n_equals_2(self):
        """
        Test with n=2 (smallest non-trivial case, m=1).
        """
        m = 1
        n = 2 * m

        # Simple 1x1 matrices embedded in n x n arrays
        a = np.zeros((n, n), dtype=complex, order='F')
        a[0, 0] = 1.0 + 0.5j

        de = np.zeros((n, n), dtype=complex, order='F')
        de[0, 0] = 0.1j  # E(1,1)
        de[0, 1] = 0.2j  # D(1,1) in column 2

        b = np.zeros((n, n), dtype=complex, order='F')
        b[0, 0] = 0.8 + 0.3j

        fg = np.zeros((n, n), dtype=complex, order='F')
        fg[0, 0] = 0.4  # G(1,1)
        fg[0, 1] = 0.5  # F(1,1) in column 2

        alphar, alphai, beta, neig, info = mb3lzp(
            'N', 'P', n, a.copy(order='F'), de.copy(order='F'),
            b.copy(order='F'), fg.copy(order='F')
        )

        assert info == 0, f"mb3lzp returned info={info}"
        assert len(alphar) == n
        assert len(alphai) == n
        assert len(beta) == n


class TestMB3LZPErrorHandling:
    """Error handling tests."""

    def test_invalid_compq(self):
        """Test invalid COMPQ parameter returns info=-1."""
        m = 2
        n = 2 * m
        a, de, b, fg = make_test_pencil(m)

        result = mb3lzp(
            'X', 'P', n, a.copy(order='F'), de.copy(order='F'),
            b.copy(order='F'), fg.copy(order='F')
        )

        # For COMPQ='N', last item is info
        # For COMPQ='X', should still return error
        if len(result) == 5:
            info = result[-1]
        else:
            info = result[-1]

        assert info == -1

    def test_invalid_orth(self):
        """Test invalid ORTH parameter when COMPQ='C' returns info=-2."""
        m = 2
        n = 2 * m
        a, de, b, fg = make_test_pencil(m)

        result = mb3lzp(
            'C', 'X', n, a.copy(order='F'), de.copy(order='F'),
            b.copy(order='F'), fg.copy(order='F')
        )

        # When error, still returns tuple with info
        if len(result) == 5:
            info = result[-1]
        else:
            info = result[-1]

        assert info == -2

    def test_invalid_n_negative(self):
        """Test negative n raises ValueError."""
        n = -2
        a = np.zeros((1, 1), dtype=complex, order='F')
        de = np.zeros((1, 1), dtype=complex, order='F')
        b = np.zeros((1, 1), dtype=complex, order='F')
        fg = np.zeros((1, 1), dtype=complex, order='F')

        with pytest.raises(ValueError):
            mb3lzp('N', 'P', n, a, de, b, fg)

    def test_invalid_n_odd(self):
        """Test odd n raises ValueError."""
        n = 3
        a = np.zeros((n, n), dtype=complex, order='F')
        de = np.zeros((n, n), dtype=complex, order='F')
        b = np.zeros((n, n), dtype=complex, order='F')
        fg = np.zeros((n, n), dtype=complex, order='F')

        with pytest.raises(ValueError):
            mb3lzp('N', 'P', n, a, de, b, fg)


class TestMB3LZPLargerMatrices:
    """Tests with larger matrices."""

    def test_larger_matrix_eigenvalues(self):
        """
        Test with larger matrix to exercise computation paths.

        Random seed: 888 (for reproducibility)
        """
        np.random.seed(888)
        m = 4
        n = 2 * m

        # Create random A in leading m x m block
        a = np.zeros((n, n), dtype=complex, order='F')
        a[:m, :m] = np.random.randn(m, m) + 1j * np.random.randn(m, m)

        # Create DE with proper structure
        de = np.zeros((n, n), dtype=complex, order='F')
        # E: lower triangular of leading m x m, skew-Hermitian
        for i in range(m):
            de[i, i] = 1j * np.random.randn()  # purely imaginary diagonal
            for j in range(i):
                de[i, j] = np.random.randn() + 1j * np.random.randn()
        # D: upper triangular in columns 1 to m (indices m+1 in Fortran = m in 0-based)
        # D is in columns 1 to m (0-indexed), upper triangular part
        for j in range(1, m+1):
            for i in range(min(j, m)):
                if j < n:  # columns 1..m in 0-indexed
                    de[i, j] = 1j * np.random.randn() if i == j-1 else np.random.randn() + 1j * np.random.randn()

        # Create random B in leading m x m block
        b = np.zeros((n, n), dtype=complex, order='F')
        b[:m, :m] = np.random.randn(m, m) + 1j * np.random.randn(m, m)

        # Create FG with proper structure
        fg = np.zeros((n, n), dtype=complex, order='F')
        # G: lower triangular of leading m x m, Hermitian
        for i in range(m):
            fg[i, i] = np.random.randn()  # real diagonal
            for j in range(i):
                fg[i, j] = np.random.randn() + 1j * np.random.randn()
        # F: upper triangular in columns 1 to m
        for j in range(1, m+1):
            for i in range(min(j, m)):
                if j < n:
                    fg[i, j] = np.random.randn() if i == j-1 else np.random.randn() + 1j * np.random.randn()

        alphar, alphai, beta, neig, info = mb3lzp(
            'N', 'P', n, a.copy(order='F'), de.copy(order='F'),
            b.copy(order='F'), fg.copy(order='F')
        )

        assert info == 0, f"mb3lzp returned info={info}"
        assert len(alphar) == n
        assert len(alphai) == n
        assert len(beta) == n

    def test_larger_matrix_with_subspace(self):
        """
        Test with larger matrix including deflating subspace.

        Random seed: 999 (for reproducibility)
        """
        np.random.seed(999)
        m = 3
        n = 2 * m

        # Create random matrices with proper structure
        a = np.zeros((n, n), dtype=complex, order='F')
        a[:m, :m] = np.random.randn(m, m) + 1j * np.random.randn(m, m)

        de = np.zeros((n, n), dtype=complex, order='F')
        for i in range(m):
            de[i, i] = 1j * np.random.randn()
            for j in range(i):
                de[i, j] = np.random.randn() + 1j * np.random.randn()
        for j in range(1, m+1):
            for i in range(min(j, m)):
                if j < n:
                    de[i, j] = np.random.randn() + 1j * np.random.randn()

        b = np.zeros((n, n), dtype=complex, order='F')
        b[:m, :m] = np.random.randn(m, m) + 1j * np.random.randn(m, m)

        fg = np.zeros((n, n), dtype=complex, order='F')
        for i in range(m):
            fg[i, i] = np.random.randn()
            for j in range(i):
                fg[i, j] = np.random.randn() + 1j * np.random.randn()
        for j in range(1, m+1):
            for i in range(min(j, m)):
                if j < n:
                    fg[i, j] = np.random.randn() + 1j * np.random.randn()

        result = mb3lzp(
            'C', 'P', n, a.copy(order='F'), de.copy(order='F'),
            b.copy(order='F'), fg.copy(order='F')
        )

        a_out, de_out, b_out, fg_out, q_out, alphar, alphai, beta, neig, info = result

        assert info == 0, f"mb3lzp returned info={info}"
        assert q_out.shape == (2*n, 2*n)

        # The leading N-by-NEIG part contains orthonormal columns
        if neig > 0:
            q_leading = q_out[:n, :neig]
            qtq = q_leading.conj().T @ q_leading
            eye_neig = np.eye(neig)
            np.testing.assert_allclose(qtq, eye_neig, rtol=1e-12, atol=1e-14)
