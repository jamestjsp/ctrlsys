"""
Tests for MB03ZD - Computing stable/unstable invariant subspaces for Hamiltonian matrix.

MB03ZD computes the stable and unstable invariant subspaces for a Hamiltonian
matrix with no eigenvalues on the imaginary axis, using the output of MB03XD.
"""

import numpy as np
import pytest


class TestMB03ZD:
    """Tests for mb03zd routine."""

    def test_basic_html_example(self):
        """
        Test MB03ZD using the example from SLICOT HTML documentation.

        Uses METH='L' (compute basis from 2*n vectors), STAB='B' (both subspaces).
        WHICH='A' selects all n eigenvalues, BALANC='N' (no balancing).
        """
        from ctrlsys import mb03zd

        n = 5
        ilo = 1

        # Input matrices from HTML doc (row-major read in Fortran = row-by-row)
        # S matrix (real Schur form)
        s = np.array([
            [-3.1844761777714732,  0.1612357243439331, -0.0628592203751138,  0.2449004200921981,  0.1974400149992579],
            [ 0.0000000000000000, -0.1510667773167784,  0.4260444411622838, -0.1775026035208615,  0.3447278421198472],
            [ 0.0000000000000000, -0.1386140422054264, -0.3006779624777515,  0.2944143257134196,  0.3456440339120323],
            [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000, -0.2710128384740570,  0.0933189808067138],
            [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,  0.4844146572359603,  0.2004347508746697]
        ], dtype=float, order='F')

        # T matrix (upper triangular)
        t = np.array([
            [ 3.2038208121776366,  0.1805955192510651,  0.2466389119377561, -0.2539149302433368, -0.0359238844381195],
            [ 0.0000000000000000, -0.7196686433290816,  0.0000000000000000,  0.2428659121580384, -0.0594190100670832],
            [ 0.0000000000000000,  0.0000000000000000, -0.1891741194498107, -0.3309578443491266, -0.0303520731950515],
            [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000, -0.4361574461961550,  0.0000000000000000],
            [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,  0.0000000000000000,  0.1530894573304220]
        ], dtype=float, order='F')

        # G matrix (general)
        g = np.array([
            [-0.0370982242678464,  0.0917788436945724, -0.0560402416315252,  0.1345152517579192,  0.0256668227276700],
            [ 0.0652183678916931, -0.0700457231988297,  0.0350041175858839, -0.2233868768749268, -0.1171980260782843],
            [-0.0626428681377119,  0.2327575351902772, -0.1251515732208170, -0.0177816046663201,  0.3696921118421182],
            [ 0.0746042309265599, -0.0828007611045140,  0.0217427473546043, -0.1157775118548851, -0.3161183681200527],
            [ 0.1374372236164812,  0.1002727885506992,  0.4021556774753973, -0.0431072263235579,  0.1067394572547867]
        ], dtype=float, order='F')

        # U1 matrix (orthogonal symplectic block)
        u1 = np.array([
            [ 0.3806883009357247, -0.0347810363019649, -0.5014665065895758,  0.5389691288472394,  0.2685446895251367],
            [ 0.4642712665555326, -0.5942766860716395,  0.4781179763952615,  0.2334370556238151,  0.0166790369048933],
            [ 0.2772789197782788, -0.0130145392695876, -0.2123817030594055, -0.2550292626960107, -0.5049268366774490],
            [ 0.4209268575081796,  0.1499593172661228, -0.1925590746592156, -0.5472292877802402,  0.4543329704184054],
            [ 0.3969669479129449,  0.6321903535930828,  0.3329156356041961,  0.0163533225344433, -0.2638879466190024]
        ], dtype=float, order='F')

        # U2 matrix
        u2 = np.array([
            [-0.1795922007470742,  0.1908329820840911,  0.0868799433942070,  0.3114741142062388, -0.2579907627915167],
            [-0.2447897730222852, -0.1028403314750045, -0.1157840914576285, -0.1873268885694406,  0.1700708002861580],
            [-0.2243335325285328,  0.3180998613802520,  0.3315380214794822,  0.1977859924739963,  0.5072476567310013],
            [-0.2128397588651423, -0.2740560593051881,  0.1941418870268881, -0.3096684962457369, -0.0581576193198714],
            [-0.2002027567371932, -0.0040094115506855, -0.3979373387545264,  0.1520881534833910, -0.2010804514091372]
        ], dtype=float, order='F')

        # V1 matrix
        v1 = np.array([
            [ 0.4447147692018334, -0.6830166755147440, -0.0002576861753487,  0.5781954611783305, -0.0375091627893805],
            [ 0.5121756358795817,  0.0297197140254773,  0.4332229148788766, -0.3240527006890552,  0.5330850295256511],
            [ 0.3664711365265602,  0.3288511296455119,  0.0588396016404451,  0.1134221597062257,  0.1047567336850078],
            [ 0.4535357098437908,  0.1062866148880792, -0.3964092656837774, -0.2211800890450674,  0.0350667323996222],
            [ 0.4450432900616097,  0.2950206358263853, -0.1617837757183893, -0.0376369332204927, -0.6746752660482623]
        ], dtype=float, order='F')

        # V2 matrix
        v2 = np.array([
            [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
            [ 0.0299719306696789, -0.2322624725320701, -0.0280846899680325, -0.3044255686880000, -0.1077641482535519],
            [-0.0069083614679702,  0.3351358347080056, -0.4922707032978891,  0.4293545450291714,  0.4372821269062001],
            [ 0.0167847133528843,  0.2843629278945327,  0.5958979805231146,  0.3097336757510886, -0.2086733033047188],
            [ 0.0248567764822071, -0.2810759958040470, -0.1653113624869834, -0.3528780198620412, -0.0254898556119252]
        ], dtype=float, order='F')

        scale = np.zeros(n, dtype=float, order='F')

        # Call with WHICH='A', METH='L', STAB='B', BALANC='N', ORTBAL='B'
        m, wr, wi, us, uu, info = mb03zd(
            'A', 'L', 'B', 'N', 'B',
            n, 2*n, ilo, scale, s, t, g, u1, u2, v1, v2
        )

        assert info == 0, f"mb03zd returned info={info}"
        assert m == n, f"Expected m={n}, got m={m}"

        # Expected eigenvalues from HTML doc
        expected_wr = np.array([-3.1941, -0.1350, -0.1350, -0.0595, -0.0595])
        expected_wi = np.array([0.0000, 0.3179, -0.3179, 0.2793, -0.2793])

        # Validate eigenvalues (loose tolerance for 4-digit precision in HTML)
        np.testing.assert_allclose(wr, expected_wr, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(wi, expected_wi, rtol=1e-3, atol=1e-3)

        # Verify US and UU have correct shape
        assert us.shape == (2*n, n), f"US shape {us.shape}, expected {(2*n, n)}"
        assert uu.shape == (2*n, n), f"UU shape {uu.shape}, expected {(2*n, n)}"

        # Verify orthogonality of US: || US'*US - I ||_F should be small
        us_orth = us.T @ us - np.eye(m)
        orth_us = np.linalg.norm(us_orth, 'fro')
        assert orth_us < 1e-13, f"US orthogonality error: {orth_us}"

        # Verify orthogonality of UU
        uu_orth = uu.T @ uu - np.eye(m)
        orth_uu = np.linalg.norm(uu_orth, 'fro')
        assert orth_uu < 1e-13, f"UU orthogonality error: {orth_uu}"

        # Verify symplecticity of US: || US'*J*US ||_F should be small
        # J = [0 I; -I 0] for symplectic structure
        us1, us2 = us[:n, :], us[n:, :]
        symp_us = us1.T @ us2 - us2.T @ us1
        symp_us_norm = np.linalg.norm(symp_us, 'fro')
        assert symp_us_norm < 1e-13, f"US symplecticity error: {symp_us_norm}"

        # Verify symplecticity of UU
        uu1, uu2 = uu[:n, :], uu[n:, :]
        symp_uu = uu1.T @ uu2 - uu2.T @ uu1
        symp_uu_norm = np.linalg.norm(symp_uu, 'fro')
        assert symp_uu_norm < 1e-12, f"UU symplecticity error: {symp_uu_norm}"

    def test_stable_only(self):
        """
        Test computing only stable invariant subspace (STAB='S').

        Random seed: 42 (for reproducibility)
        """
        from ctrlsys import mb03zd

        np.random.seed(42)
        n = 3
        ilo = 1

        # Create test matrices with stable eigenvalues
        s = np.array([
            [-2.0, 0.1, 0.2],
            [ 0.0, -1.0, 0.3],
            [ 0.0,  0.0, -0.5]
        ], dtype=float, order='F')

        t = np.array([
            [1.0, 0.1, 0.2],
            [0.0, 1.0, 0.1],
            [0.0, 0.0, 1.0]
        ], dtype=float, order='F')

        g = np.random.randn(n, n).astype(float, order='F')

        # Create orthogonal symplectic matrices
        q1, _ = np.linalg.qr(np.random.randn(n, n))
        q2, _ = np.linalg.qr(np.random.randn(n, n))
        u1 = q1.astype(float, order='F')
        u2 = (0.1 * q2).astype(float, order='F')
        v1 = q1.astype(float, order='F')
        v2 = (0.1 * q2).astype(float, order='F')

        scale = np.zeros(n, dtype=float, order='F')

        # Call with STAB='S' for stable subspace only, METH='S'
        m, wr, wi, us, uu, info = mb03zd(
            'A', 'S', 'S', 'N', 'B',
            n, n, ilo, scale, s.copy(order='F'), t.copy(order='F'), g.copy(order='F'),
            u1.copy(order='F'), u2.copy(order='F'), v1.copy(order='F'), v2.copy(order='F')
        )

        # info=5 is warning (inaccurate stable subspace for METH='S'), not error
        assert info in [0, 5], f"mb03zd returned info={info}"
        assert m == n, f"Expected m={n}, got m={m}"

        # All eigenvalues should have negative real part
        assert np.all(wr < 0), f"Not all eigenvalues stable: wr={wr}"

        # US should be orthonormal
        us_orth = us.T @ us - np.eye(m)
        orth_norm = np.linalg.norm(us_orth, 'fro')
        assert orth_norm < 1e-12, f"US orthogonality error: {orth_norm}"

    def test_unstable_only(self):
        """
        Test computing only unstable invariant subspace (STAB='U').

        Random seed: 123 (for reproducibility)
        """
        from ctrlsys import mb03zd

        np.random.seed(123)
        n = 3
        ilo = 1

        # Create test matrices
        s = np.array([
            [-2.0, 0.1, 0.2],
            [ 0.0, -1.0, 0.3],
            [ 0.0,  0.0, -0.5]
        ], dtype=float, order='F')

        t = np.array([
            [1.0, 0.1, 0.2],
            [0.0, 1.0, 0.1],
            [0.0, 0.0, 1.0]
        ], dtype=float, order='F')

        g = np.random.randn(n, n).astype(float, order='F')

        q1, _ = np.linalg.qr(np.random.randn(n, n))
        q2, _ = np.linalg.qr(np.random.randn(n, n))
        u1 = q1.astype(float, order='F')
        u2 = (0.1 * q2).astype(float, order='F')
        v1 = q1.astype(float, order='F')
        v2 = (0.1 * q2).astype(float, order='F')

        scale = np.zeros(n, dtype=float, order='F')

        # Call with STAB='U' for unstable subspace only
        m, wr, wi, us, uu, info = mb03zd(
            'A', 'S', 'U', 'N', 'B',
            n, n, ilo, scale, s.copy(order='F'), t.copy(order='F'), g.copy(order='F'),
            u1.copy(order='F'), u2.copy(order='F'), v1.copy(order='F'), v2.copy(order='F')
        )

        assert info == 0, f"mb03zd returned info={info}"

        # UU should be orthonormal
        uu_orth = uu.T @ uu - np.eye(m)
        orth_norm = np.linalg.norm(uu_orth, 'fro')
        assert orth_norm < 1e-12, f"UU orthogonality error: {orth_norm}"

    def test_quick_return_n_zero(self):
        """Test quick return when n=0."""
        from ctrlsys import mb03zd

        n = 0
        mm = 0
        ilo = 1

        scale = np.array([], dtype=float, order='F')
        s = np.zeros((1, 0), dtype=float, order='F')
        t = np.zeros((1, 0), dtype=float, order='F')
        g = np.zeros((1, 0), dtype=float, order='F')
        u1 = np.zeros((1, 0), dtype=float, order='F')
        u2 = np.zeros((1, 0), dtype=float, order='F')
        v1 = np.zeros((1, 0), dtype=float, order='F')
        v2 = np.zeros((1, 0), dtype=float, order='F')

        m, wr, wi, us, uu, info = mb03zd(
            'A', 'S', 'B', 'N', 'B',
            n, mm, ilo, scale, s, t, g, u1, u2, v1, v2
        )

        assert info == 0, f"mb03zd returned info={info}"
        assert m == 0, f"Expected m=0, got m={m}"

    def test_invalid_which(self):
        """Test error handling for invalid WHICH parameter."""
        from ctrlsys import mb03zd

        n = 3
        ilo = 1

        s = np.eye(n, dtype=float, order='F')
        t = np.eye(n, dtype=float, order='F')
        g = np.eye(n, dtype=float, order='F')
        u1 = np.eye(n, dtype=float, order='F')
        u2 = np.eye(n, dtype=float, order='F')
        v1 = np.eye(n, dtype=float, order='F')
        v2 = np.eye(n, dtype=float, order='F')
        scale = np.zeros(n, dtype=float, order='F')

        m, wr, wi, us, uu, info = mb03zd(
            'X', 'S', 'B', 'N', 'B',
            n, n, ilo, scale, s, t, g, u1, u2, v1, v2
        )

        assert info == -1, f"Expected info=-1 for invalid WHICH, got info={info}"

    def test_invalid_meth(self):
        """Test error handling for invalid METH parameter."""
        from ctrlsys import mb03zd

        n = 3
        ilo = 1

        s = np.eye(n, dtype=float, order='F')
        t = np.eye(n, dtype=float, order='F')
        g = np.eye(n, dtype=float, order='F')
        u1 = np.eye(n, dtype=float, order='F')
        u2 = np.eye(n, dtype=float, order='F')
        v1 = np.eye(n, dtype=float, order='F')
        v2 = np.eye(n, dtype=float, order='F')
        scale = np.zeros(n, dtype=float, order='F')

        m, wr, wi, us, uu, info = mb03zd(
            'A', 'X', 'B', 'N', 'B',
            n, n, ilo, scale, s, t, g, u1, u2, v1, v2
        )

        assert info == -2, f"Expected info=-2 for invalid METH, got info={info}"

    def test_invalid_stab(self):
        """Test error handling for invalid STAB parameter."""
        from ctrlsys import mb03zd

        n = 3
        ilo = 1

        s = np.eye(n, dtype=float, order='F')
        t = np.eye(n, dtype=float, order='F')
        g = np.eye(n, dtype=float, order='F')
        u1 = np.eye(n, dtype=float, order='F')
        u2 = np.eye(n, dtype=float, order='F')
        v1 = np.eye(n, dtype=float, order='F')
        v2 = np.eye(n, dtype=float, order='F')
        scale = np.zeros(n, dtype=float, order='F')

        m, wr, wi, us, uu, info = mb03zd(
            'A', 'S', 'X', 'N', 'B',
            n, n, ilo, scale, s, t, g, u1, u2, v1, v2
        )

        assert info == -3, f"Expected info=-3 for invalid STAB, got info={info}"

    def test_meth_small_from_n_vectors(self):
        """
        Test METH='S' which computes basis from n vectors.

        Random seed: 456 (for reproducibility)
        """
        from ctrlsys import mb03zd

        np.random.seed(456)
        n = 4
        ilo = 1

        # Create a diagonal S matrix for predictable eigenvalues
        s = np.diag([-3.0, -2.0, -1.0, -0.5]).astype(float, order='F')
        t = np.eye(n, dtype=float, order='F')
        g = np.zeros((n, n), dtype=float, order='F')

        # Orthogonal matrices
        q, _ = np.linalg.qr(np.random.randn(n, n))
        u1 = q.astype(float, order='F')
        u2 = np.zeros((n, n), dtype=float, order='F')
        v1 = q.astype(float, order='F')
        v2 = np.zeros((n, n), dtype=float, order='F')

        scale = np.zeros(n, dtype=float, order='F')

        # METH='S' means MM should be n
        m, wr, wi, us, uu, info = mb03zd(
            'A', 'S', 'B', 'N', 'B',
            n, n, ilo, scale, s.copy(order='F'), t.copy(order='F'), g.copy(order='F'),
            u1.copy(order='F'), u2.copy(order='F'), v1.copy(order='F'), v2.copy(order='F')
        )

        # info=5 or info=6 are warnings (inaccurate subspace), still valid
        assert info in [0, 5, 6], f"mb03zd returned unexpected info={info}"
        assert m == n, f"Expected m={n}, got m={m}"

    def test_meth_large_from_2n_vectors(self):
        """
        Test METH='L' which computes basis from 2*n vectors.

        This is the method used in the HTML example and typically gives
        more accurate results.

        Random seed: 789 (for reproducibility)
        """
        from ctrlsys import mb03zd

        np.random.seed(789)
        n = 4
        ilo = 1

        # Diagonal S for predictable eigenvalues
        s = np.diag([-2.5, -1.5, -0.8, -0.3]).astype(float, order='F')
        t = np.eye(n, dtype=float, order='F')
        g = 0.1 * np.random.randn(n, n).astype(float, order='F')

        q, _ = np.linalg.qr(np.random.randn(n, n))
        u1 = q.astype(float, order='F')
        u2 = 0.1 * np.random.randn(n, n).astype(float, order='F')
        v1 = q.astype(float, order='F')
        v2 = 0.1 * np.random.randn(n, n).astype(float, order='F')

        scale = np.zeros(n, dtype=float, order='F')

        # METH='L' means MM should be 2*n
        m, wr, wi, us, uu, info = mb03zd(
            'A', 'L', 'B', 'N', 'B',
            n, 2*n, ilo, scale, s.copy(order='F'), t.copy(order='F'), g.copy(order='F'),
            u1.copy(order='F'), u2.copy(order='F'), v1.copy(order='F'), v2.copy(order='F')
        )

        assert info == 0, f"mb03zd returned info={info}"
        assert m == n, f"Expected m={n}, got m={m}"

        # Verify orthogonality
        us_orth = us.T @ us - np.eye(m)
        assert np.linalg.norm(us_orth, 'fro') < 1e-12

        uu_orth = uu.T @ uu - np.eye(m)
        assert np.linalg.norm(uu_orth, 'fro') < 1e-12
