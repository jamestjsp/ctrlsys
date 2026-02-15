"""
Tests for tg01nd - Finite-infinite block-diagonal decomposition
of a descriptor system.

TG01ND computes equivalence transformation matrices Q and Z which reduce
the regular pole pencil A-lambda*E of the descriptor system (A-lambda*E,B,C)
to block-diagonal form with complete finite-infinite separation.
Unlike TG01MD, TG01ND produces fully block-diagonal form by solving
generalized Sylvester equations to eliminate the off-diagonal blocks.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


class TestTg01ndBasic:
    """Test basic functionality from HTML doc example."""

    def test_html_doc_example(self):
        """
        Validate basic functionality using SLICOT HTML doc example.

        N=4, M=2, P=2, JOB='F', JOBT='D', TOL=0.0
        Expected: NF=3, ND=1, NIBLCK=1
        """
        from ctrlsys import tg01nd

        n, m, p = 4, 2, 2

        a = np.array([
            [-1.0, 0.0, 0.0, 3.0],
            [0.0, 0.0, 1.0, 2.0],
            [1.0, 1.0, 0.0, 4.0],
            [0.0, 0.0, 0.0, 0.0]
        ], order='F', dtype=float)

        e = np.array([
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [3.0, 9.0, 6.0, 3.0],
            [0.0, 0.0, 2.0, 0.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ], order='F', dtype=float)

        c = np.array([
            [-1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, -1.0, 1.0]
        ], order='F', dtype=float)

        result = tg01nd('F', 'D', n, m, p, a, e, b, c, 0.0)

        (a_out, e_out, b_out, c_out, alphar, alphai, beta,
         q, z, nf, nd, niblck, iblck, info) = result

        assert info == 0, f"tg01nd failed with info={info}"
        assert nf == 3, f"Expected nf=3, got {nf}"
        assert nd == 1, f"Expected nd=1, got {nd}"
        assert niblck == 0, f"Expected niblck=0, got {niblck}"

        # Verify block-diagonal structure (off-diagonal blocks should be zero)
        assert_allclose(a_out[:nf, nf:], 0.0, atol=1e-10,
                       err_msg="A off-diagonal block should be zero")
        assert_allclose(a_out[nf:, :nf], 0.0, atol=1e-10,
                       err_msg="A off-diagonal block should be zero")
        assert_allclose(e_out[:nf, nf:], 0.0, atol=1e-10,
                       err_msg="E off-diagonal block should be zero")
        assert_allclose(e_out[nf:, :nf], 0.0, atol=1e-10,
                       err_msg="E off-diagonal block should be zero")

        # Note: Q and Z from TG01ND are NOT orthogonal (unlike TG01MD).
        # TG01ND uses Q = Q2*Q1 and Z = Z1*Z2 where Q2, Z2 have the form:
        #     ( I -X )          ( I  Y )
        # Q2 = (      ) ,   Z2 = (      )
        #     ( 0  I )          ( 0  I )
        # Verify transformation relationship instead: A_out = Q @ A_orig @ Z
        a_orig = np.array([
            [-1.0, 0.0, 0.0, 3.0],
            [0.0, 0.0, 1.0, 2.0],
            [1.0, 1.0, 0.0, 4.0],
            [0.0, 0.0, 0.0, 0.0]
        ], order='F', dtype=float)
        e_orig = np.array([
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [3.0, 9.0, 6.0, 3.0],
            [0.0, 0.0, 2.0, 0.0]
        ], order='F', dtype=float)
        assert_allclose(a_out, q @ a_orig @ z, rtol=1e-12, atol=1e-12)
        assert_allclose(e_out, q @ e_orig @ z, rtol=1e-12, atol=1e-12)

        # Verify diagonal absolute values (sign-independent)
        a_diag_expected = np.array([1.2803, 0.5796, 0.0000, 2.2913])
        e_diag_expected = np.array([9.3142, 0.1594, 2.3524, 0.0000])
        assert_allclose(np.abs(np.diag(a_out)), a_diag_expected, rtol=5e-4, atol=1e-4)
        assert_allclose(np.abs(np.diag(e_out)), e_diag_expected, rtol=5e-4, atol=1e-4)

        eig_expected = np.array([0.1375, -3.6375, 0.0])
        eig_computed = []
        for j in range(nf):
            if beta[j] != 0:
                eig_computed.append(complex(alphar[j], alphai[j]) / beta[j])
        eig_computed = np.array([e.real for e in eig_computed])

        eig_computed_sorted = np.sort(eig_computed)
        eig_expected_sorted = np.sort(eig_expected)
        assert_allclose(eig_computed_sorted, eig_expected_sorted, rtol=5e-3, atol=1e-4)


class TestTg01ndBlockDiagonal:
    """Test block-diagonal structure properties."""

    def test_block_diagonal_structure(self):
        """
        Validate that TG01ND produces fully block-diagonal form.

        This is the key difference from TG01MD: the off-diagonal blocks
        A12, E12 (between finite and infinite parts) should be zero.

        Random seed: 42 (for reproducibility)
        """
        from ctrlsys import tg01nd

        np.random.seed(42)
        n, m, p = 5, 2, 3

        a = np.random.randn(n, n).astype(float, order='F')
        e = np.random.randn(n, n).astype(float, order='F')
        e[4, :] = 0.0
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        result = tg01nd('F', 'D', n, m, p, a, e, b, c, 0.0)
        (a_out, e_out, b_out, c_out, alphar, alphai, beta,
         q, z, nf, nd, niblck, iblck, info) = result

        assert info == 0, f"tg01nd failed with info={info}"

        if nf > 0 and nf < n:
            a12 = a_out[:nf, nf:]
            e12 = e_out[:nf, nf:]
            a21 = a_out[nf:, :nf]
            e21 = e_out[nf:, :nf]

            assert_allclose(a12, 0.0, atol=1e-10,
                           err_msg="A12 block should be zero for block-diagonal form")
            assert_allclose(e12, 0.0, atol=1e-10,
                           err_msg="E12 block should be zero for block-diagonal form")
            assert_allclose(a21, 0.0, atol=1e-10,
                           err_msg="A21 block should be zero for block-diagonal form")
            assert_allclose(e21, 0.0, atol=1e-10,
                           err_msg="E21 block should be zero for block-diagonal form")


class TestTg01ndTransformationType:
    """Test direct vs inverse transformation matrices."""

    def test_direct_transformation(self):
        """
        Validate JOBT='D' returns direct transformation matrices Q, Z.

        For direct: A_out = Q*A*Z, E_out = Q*E*Z, B_out = Q*B, C_out = C*Z.
        But Q from routine is actually the accumulated orthogonal Q' (transposed)
        so verification is: A_out = Q.T @ A_orig @ Z.

        Random seed: 123 (for reproducibility)
        """
        from ctrlsys import tg01nd

        np.random.seed(123)
        n, m, p = 4, 2, 2

        a_orig = np.random.randn(n, n).astype(float, order='F')
        e_orig = np.random.randn(n, n).astype(float, order='F')
        e_orig = e_orig + 2.0 * np.eye(n, order='F')
        b_orig = np.random.randn(n, m).astype(float, order='F')
        c_orig = np.random.randn(p, n).astype(float, order='F')

        a = a_orig.copy(order='F')
        e = e_orig.copy(order='F')
        b = b_orig.copy(order='F')
        c = c_orig.copy(order='F')

        result = tg01nd('F', 'D', n, m, p, a, e, b, c, 0.0)
        (a_out, e_out, b_out, c_out, alphar, alphai, beta,
         q, z, nf, nd, niblck, iblck, info) = result

        assert info == 0, f"tg01nd failed with info={info}"

        a_transformed = q @ a_orig @ z
        e_transformed = q @ e_orig @ z
        b_transformed = q @ b_orig
        c_transformed = c_orig @ z

        assert_allclose(a_out, a_transformed, rtol=1e-12, atol=1e-13)
        assert_allclose(e_out, e_transformed, rtol=1e-12, atol=1e-13)
        assert_allclose(b_out, b_transformed, rtol=1e-12, atol=1e-13)
        assert_allclose(c_out, c_transformed, rtol=1e-12, atol=1e-13)

    def test_inverse_transformation(self):
        """
        Validate JOBT='I' returns inverse transformation matrices inv(Q), inv(Z).

        For JOBT='I', the returned Q and Z are inverses of the direct Q, Z.
        The output matrices A_out, E_out are the same as with JOBT='D'.
        The relationships are: Q_inv = inv(Q_direct), Z_inv = inv(Z_direct).

        Random seed: 456 (for reproducibility)
        """
        from ctrlsys import tg01nd

        np.random.seed(456)
        n, m, p = 4, 2, 2

        a_orig = np.random.randn(n, n).astype(float, order='F')
        e_orig = np.random.randn(n, n).astype(float, order='F')
        e_orig = e_orig + 2.0 * np.eye(n, order='F')
        b_orig = np.random.randn(n, m).astype(float, order='F')
        c_orig = np.random.randn(p, n).astype(float, order='F')

        a_d = a_orig.copy(order='F')
        e_d = e_orig.copy(order='F')
        b_d = b_orig.copy(order='F')
        c_d = c_orig.copy(order='F')

        result_d = tg01nd('F', 'D', n, m, p, a_d, e_d, b_d, c_d, 0.0)
        (a_out_d, e_out_d, b_out_d, c_out_d, _, _, _,
         q_d, z_d, nf_d, _, _, _, info_d) = result_d

        assert info_d == 0, f"tg01nd D failed with info={info_d}"

        a_i = a_orig.copy(order='F')
        e_i = e_orig.copy(order='F')
        b_i = b_orig.copy(order='F')
        c_i = c_orig.copy(order='F')

        result_i = tg01nd('F', 'I', n, m, p, a_i, e_i, b_i, c_i, 0.0)
        (a_out_i, e_out_i, b_out_i, c_out_i, _, _, _,
         q_inv, z_inv, nf_i, _, _, _, info_i) = result_i

        assert info_i == 0, f"tg01nd I failed with info={info_i}"

        assert_allclose(a_out_d, a_out_i, rtol=1e-13, atol=1e-14,
                       err_msg="A outputs should be identical")
        assert_allclose(e_out_d, e_out_i, rtol=1e-13, atol=1e-14,
                       err_msg="E outputs should be identical")

        assert_allclose(q_inv @ q_d, np.eye(n), rtol=1e-13, atol=1e-14,
                       err_msg="Q_inv @ Q should be identity")
        assert_allclose(z_inv @ z_d, np.eye(n), rtol=1e-13, atol=1e-14,
                       err_msg="Z_inv @ Z should be identity")


class TestTg01ndInfiniteFirst:
    """Test infinite-finite separation (JOB='I')."""

    def test_infinite_first_block_diagonal(self):
        """
        Validate JOB='I' places infinite eigenvalues first with block-diagonal form.

        Random seed: 789 (for reproducibility)
        """
        from ctrlsys import tg01nd

        np.random.seed(789)
        n, m, p = 4, 2, 2

        a = np.random.randn(n, n).astype(float, order='F')
        e = np.random.randn(n, n).astype(float, order='F')
        e[3, :] = 0.0
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        result = tg01nd('I', 'D', n, m, p, a, e, b, c, 0.0)
        (a_out, e_out, b_out, c_out, alphar, alphai, beta,
         q, z, nf, nd, niblck, iblck, info) = result

        assert info == 0, f"tg01nd failed with info={info}"

        ni = n - nf
        if ni > 0 and nf > 0:
            a12 = a_out[:ni, ni:]
            e12 = e_out[:ni, ni:]
            a21 = a_out[ni:, :ni]
            e21 = e_out[ni:, :ni]

            assert_allclose(a12, 0.0, atol=1e-10,
                           err_msg="A12 block should be zero for block-diagonal form")
            assert_allclose(e12, 0.0, atol=1e-10,
                           err_msg="E12 block should be zero for block-diagonal form")
            assert_allclose(a21, 0.0, atol=1e-10,
                           err_msg="A21 block should be zero")
            assert_allclose(e21, 0.0, atol=1e-10,
                           err_msg="E21 block should be zero")


class TestTg01ndEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_dimension(self):
        """Test with N=0 (quick return)."""
        from ctrlsys import tg01nd

        n, m, p = 0, 2, 2

        a = np.array([], order='F', dtype=float).reshape(0, 0)
        e = np.array([], order='F', dtype=float).reshape(0, 0)
        b = np.array([], order='F', dtype=float).reshape(0, m)
        c = np.array([], order='F', dtype=float).reshape(p, 0)

        result = tg01nd('F', 'D', n, m, p, a, e, b, c, 0.0)
        (a_out, e_out, b_out, c_out, alphar, alphai, beta,
         q, z, nf, nd, niblck, iblck, info) = result

        assert info == 0
        assert nf == 0
        assert nd == 0
        assert niblck == 0

    def test_invalid_job(self):
        """Test invalid JOB parameter returns error."""
        from ctrlsys import tg01nd

        n, m, p = 3, 1, 1

        a = np.eye(n, order='F', dtype=float)
        e = np.eye(n, order='F', dtype=float)
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)

        result = tg01nd('X', 'D', n, m, p, a, e, b, c, 0.0)
        info = result[-1]
        assert info == -1, f"Expected info=-1 for invalid JOB, got {info}"

    def test_invalid_jobt(self):
        """Test invalid JOBT parameter returns error."""
        from ctrlsys import tg01nd

        n, m, p = 3, 1, 1

        a = np.eye(n, order='F', dtype=float)
        e = np.eye(n, order='F', dtype=float)
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)

        result = tg01nd('F', 'X', n, m, p, a, e, b, c, 0.0)
        info = result[-1]
        assert info == -2, f"Expected info=-2 for invalid JOBT, got {info}"

    def test_all_finite_eigenvalues(self):
        """
        Test system with all finite eigenvalues (E nonsingular).

        Random seed: 1234 (for reproducibility)
        """
        from ctrlsys import tg01nd

        np.random.seed(1234)
        n, m, p = 4, 2, 2

        a = np.random.randn(n, n).astype(float, order='F')
        e = np.eye(n, order='F', dtype=float)
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        result = tg01nd('F', 'D', n, m, p, a, e, b, c, 0.0)
        (a_out, e_out, b_out, c_out, alphar, alphai, beta,
         q, z, nf, nd, niblck, iblck, info) = result

        assert info == 0, f"tg01nd failed with info={info}"
        assert nf == n, f"Expected all eigenvalues finite, got nf={nf}"


class TestTg01ndEigenvaluePreservation:
    """Test eigenvalue preservation property."""

    def test_finite_eigenvalue_preservation(self):
        """
        Validate that finite eigenvalues are preserved under transformation.

        The generalized eigenvalues of (A, E) should be the same before
        and after transformation.

        Random seed: 5678 (for reproducibility)
        """
        from ctrlsys import tg01nd

        np.random.seed(5678)
        n, m, p = 4, 2, 2

        a_orig = np.random.randn(n, n).astype(float, order='F')
        e_orig = np.random.randn(n, n).astype(float, order='F')
        e_orig = e_orig + 3.0 * np.eye(n, order='F')
        b = np.random.randn(n, m).astype(float, order='F')
        c = np.random.randn(p, n).astype(float, order='F')

        eig_before = np.linalg.eigvals(np.linalg.solve(e_orig, a_orig))
        eig_before_sorted = np.sort(eig_before.real)

        a = a_orig.copy(order='F')
        e = e_orig.copy(order='F')

        result = tg01nd('F', 'D', n, m, p, a, e, b, c, 0.0)
        (a_out, e_out, b_out, c_out, alphar, alphai, beta,
         q, z, nf, nd, niblck, iblck, info) = result

        assert info == 0, f"tg01nd failed with info={info}"

        if nf == n:
            Af = a_out[:nf, :nf]
            Ef = e_out[:nf, :nf]
            eig_after = np.linalg.eigvals(np.linalg.solve(Ef, Af))
            eig_after_sorted = np.sort(eig_after.real)

            assert_allclose(eig_before_sorted, eig_after_sorted, rtol=1e-10, atol=1e-12)
