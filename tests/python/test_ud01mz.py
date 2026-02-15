"""
Tests for UD01MZ - Print complex matrix with formatted output.

UD01MZ prints an M-by-N complex matrix row by row with 7 significant figures.
"""

import numpy as np
import pytest
from ctrlsys import ud01mz


class TestUD01MZBasic:
    """Basic functionality tests for UD01MZ."""

    def test_small_matrix_l1(self):
        """
        Test 2x2 complex matrix with L=1 (one element per line).

        Validates formatted output contains correct values.
        """
        a = np.array([
            [1.0 + 2.0j, 3.0 + 4.0j],
            [5.0 - 1.0j, -2.0 + 0.5j]
        ], order='F', dtype=np.complex128)

        text = "Test Matrix A"
        result, info = ud01mz(a, text, l=1)

        assert info == 0
        assert isinstance(result, str)
        assert "Test Matrix A" in result
        assert "(    2X    2)" in result
        assert "1.0000000" in result
        assert "2.0000000" in result

    def test_small_matrix_l2(self):
        """
        Test 2x3 complex matrix with L=2 (two elements per line).

        Validates multi-column output formatting.
        """
        a = np.array([
            [1.0 + 0.0j, 2.0 + 1.0j, 3.0 - 2.0j],
            [4.0 + 3.0j, 5.0 - 4.0j, 6.0 + 0.0j]
        ], order='F', dtype=np.complex128)

        text = "Matrix B"
        result, info = ud01mz(a, text, l=2)

        assert info == 0
        assert "Matrix B" in result
        assert "(    2X    3)" in result

    def test_small_matrix_l3(self):
        """
        Test 3x3 complex matrix with L=3 (three elements per line).

        Validates maximum elements per line.
        """
        a = np.array([
            [1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j],
            [4.0 - 1.0j, 5.0 - 2.0j, 6.0 - 3.0j],
            [7.0 + 0.0j, 8.0 + 0.0j, 9.0 + 0.0j]
        ], order='F', dtype=np.complex128)

        text = "Square Matrix"
        result, info = ud01mz(a, text, l=3)

        assert info == 0
        assert "Square Matrix" in result
        assert "(    3X    3)" in result

    def test_column_blocking(self):
        """
        Test matrix with more columns than L requires column blocking.

        When N > L, columns are printed in blocks.
        """
        a = np.array([
            [1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j, 4.0 + 0.0j, 5.0 + 0.0j],
        ], order='F', dtype=np.complex128)

        text = "Wide Matrix"
        result, info = ud01mz(a, text, l=2)

        assert info == 0
        assert "Wide Matrix" in result
        lines = result.strip().split('\n')
        assert len(lines) > 3


class TestUD01MZNumericalFormat:
    """Test numerical formatting precision."""

    def test_scientific_notation(self):
        """
        Test that large/small values use scientific notation.

        Random seed: 42 (for reproducibility)
        """
        a = np.array([
            [1.2345678e10 + 9.8765432e-5j],
            [3.1415927e-8 - 2.7182818e15j]
        ], order='F', dtype=np.complex128)

        text = "Scientific"
        result, info = ud01mz(a, text, l=1)

        assert info == 0
        assert "D+" in result or "D-" in result or "E+" in result or "E-" in result

    def test_negative_values(self):
        """
        Test negative real and imaginary parts are formatted correctly.
        """
        a = np.array([
            [-1.5 - 2.5j, -3.5 + 4.5j],
            [5.5 - 6.5j, -7.5 - 8.5j]
        ], order='F', dtype=np.complex128)

        text = "Negative Values"
        result, info = ud01mz(a, text, l=2)

        assert info == 0
        assert "-" in result

    def test_zero_matrix(self):
        """
        Test matrix of zeros.
        """
        a = np.zeros((2, 2), order='F', dtype=np.complex128)

        text = "Zero Matrix"
        result, info = ud01mz(a, text, l=2)

        assert info == 0
        assert "0.0000000" in result


class TestUD01MZEdgeCases:
    """Edge case tests."""

    def test_single_element(self):
        """
        Test 1x1 matrix (minimum valid size).
        """
        a = np.array([[42.0 + 24.0j]], order='F', dtype=np.complex128)

        text = "Scalar"
        result, info = ud01mz(a, text, l=1)

        assert info == 0
        assert "Scalar" in result
        assert "(    1X    1)" in result

    def test_single_row(self):
        """
        Test 1xN matrix (row vector).
        """
        a = np.array([[1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]], order='F', dtype=np.complex128)

        text = "Row Vector"
        result, info = ud01mz(a, text, l=3)

        assert info == 0
        assert "(    1X    3)" in result

    def test_single_column(self):
        """
        Test Mx1 matrix (column vector).
        """
        a = np.array([
            [1.0 + 1.0j],
            [2.0 + 2.0j],
            [3.0 + 3.0j]
        ], order='F', dtype=np.complex128)

        text = "Column Vector"
        result, info = ud01mz(a, text, l=1)

        assert info == 0
        assert "(    3X    1)" in result

    def test_long_text_truncated(self):
        """
        Test that text longer than 72 characters is handled.
        """
        long_text = "A" * 100
        a = np.array([[1.0 + 0.0j]], order='F', dtype=np.complex128)

        result, info = ud01mz(a, long_text, l=1)

        assert info == 0

    def test_empty_text(self):
        """
        Test with empty title text.
        """
        a = np.array([[1.0 + 0.0j]], order='F', dtype=np.complex128)

        result, info = ud01mz(a, "", l=1)

        assert info == 0


class TestUD01MZErrorHandling:
    """Error handling tests."""

    def test_invalid_l_too_small(self):
        """
        Test L < 1 returns info = -3.
        """
        a = np.array([[1.0 + 0.0j]], order='F', dtype=np.complex128)

        with pytest.raises(ValueError, match="l.*1"):
            ud01mz(a, "Test", l=0)

    def test_invalid_l_too_large(self):
        """
        Test L > 3 returns info = -3.
        """
        a = np.array([[1.0 + 0.0j]], order='F', dtype=np.complex128)

        with pytest.raises(ValueError, match="l.*3"):
            ud01mz(a, "Test", l=4)

    def test_empty_matrix_m_zero(self):
        """
        Test empty matrix (M=0) returns error.
        """
        a = np.zeros((0, 2), order='F', dtype=np.complex128)

        with pytest.raises(ValueError):
            ud01mz(a, "Test", l=1)

    def test_empty_matrix_n_zero(self):
        """
        Test empty matrix (N=0) returns error.
        """
        a = np.zeros((2, 0), order='F', dtype=np.complex128)

        with pytest.raises(ValueError):
            ud01mz(a, "Test", l=1)


class TestUD01MZRowColumnNumbers:
    """Test row and column number formatting."""

    def test_row_numbers_present(self):
        """
        Test that row numbers appear in output.
        """
        a = np.array([
            [1.0 + 0.0j, 2.0 + 0.0j],
            [3.0 + 0.0j, 4.0 + 0.0j],
            [5.0 + 0.0j, 6.0 + 0.0j]
        ], order='F', dtype=np.complex128)

        result, info = ud01mz(a, "Test", l=2)

        assert info == 0
        lines = result.split('\n')
        row_lines = [l for l in lines if l.strip().startswith(('1', '2', '3'))]
        assert len(row_lines) >= 3

    def test_column_numbers_present(self):
        """
        Test that column numbers appear in output header.
        """
        a = np.array([
            [1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]
        ], order='F', dtype=np.complex128)

        result, info = ud01mz(a, "Test", l=3)

        assert info == 0
        assert "1" in result
        assert "2" in result
        assert "3" in result
