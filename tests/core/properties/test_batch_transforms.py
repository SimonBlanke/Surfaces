# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for batch transformation utilities."""

import numpy as np
import pytest

from surfaces.test_functions._batch_transforms import (
    batch_f_pen,
    batch_lambda_alpha,
    batch_rotate,
    batch_shift,
    batch_shift_rotate,
    batch_t_asy,
    batch_t_osz,
)


class TestBatchShift:
    """Tests for batch_shift."""

    def test_basic_shift(self):
        """Shift subtracts the shift vector."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        shift = np.array([1.0, 1.0])
        result = batch_shift(X, shift)
        expected = np.array([[0.0, 1.0], [2.0, 3.0]])
        np.testing.assert_allclose(result, expected)

    def test_zero_shift(self):
        """Zero shift returns original."""
        X = np.array([[1.0, 2.0]])
        result = batch_shift(X, np.zeros(2))
        np.testing.assert_allclose(result, X)


class TestBatchRotate:
    """Tests for batch_rotate."""

    def test_identity_rotation(self):
        """Identity rotation returns original."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        R = np.eye(2)
        result = batch_rotate(X, R)
        np.testing.assert_allclose(result, X)

    def test_90_degree_rotation(self):
        """90-degree rotation works correctly."""
        X = np.array([[1.0, 0.0]])
        R = np.array([[0.0, -1.0], [1.0, 0.0]])
        result = batch_rotate(X, R)
        np.testing.assert_allclose(result, [[0.0, 1.0]], atol=1e-10)


class TestBatchShiftRotate:
    """Tests for batch_shift_rotate."""

    def test_combined(self):
        """Shift then rotate."""
        X = np.array([[2.0, 3.0]])
        shift = np.array([1.0, 1.0])
        R = np.eye(2)
        result = batch_shift_rotate(X, shift, R)
        expected = np.array([[1.0, 2.0]])
        np.testing.assert_allclose(result, expected)


class TestBatchTOsz:
    """Tests for batch_t_osz (oscillation transformation)."""

    def test_zero_input(self):
        """Zero input stays zero."""
        X = np.zeros((3, 5))
        result = batch_t_osz(X)
        np.testing.assert_allclose(result, 0.0)

    def test_sign_preserved(self):
        """Sign is preserved."""
        X = np.array([[-2.0, 3.0, -1.0, 0.5]])
        result = batch_t_osz(X)
        assert result[0, 0] < 0  # negative input -> negative output
        assert result[0, 1] > 0  # positive input -> positive output
        assert result[0, 2] < 0
        assert result[0, 3] > 0

    def test_different_coefficients_positive_negative(self):
        """Positive and negative values use different coefficients."""
        X = np.array([[5.0]])
        neg_X = np.array([[-5.0]])
        pos_result = batch_t_osz(X)
        neg_result = batch_t_osz(neg_X)
        # Results should not just be negations due to different c1/c2
        assert abs(pos_result[0, 0]) != pytest.approx(abs(neg_result[0, 0]), rel=1e-10)


class TestBatchTAsy:
    """Tests for batch_t_asy (asymmetry transformation)."""

    def test_1d(self):
        """1D case uses simplified formula (no i/(n-1) term)."""
        X = np.array([[2.0]])
        result = batch_t_asy(X, beta=0.5, n_dim=1)
        # For 1D: exp = 1 + beta * sqrt(max(x, 0))
        exp = 1 + 0.5 * np.sqrt(2.0)
        expected = 2.0**exp
        np.testing.assert_allclose(result[0, 0], expected, rtol=1e-10)

    def test_nd(self):
        """Multi-dimensional case."""
        X = np.array([[2.0, 3.0, 4.0]])
        result = batch_t_asy(X, beta=0.2, n_dim=3)
        # Check all positive values are transformed
        assert result[0, 0] > 0
        assert result[0, 1] > 0
        assert result[0, 2] > 0

    def test_negative_values_unchanged(self):
        """Negative values are not transformed."""
        X = np.array([[-1.0, -2.0, -3.0]])
        result = batch_t_asy(X, beta=0.5)
        np.testing.assert_allclose(result, X)

    def test_mixed_signs(self):
        """Mixed signs: positive transformed, negative unchanged."""
        X = np.array([[-1.0, 2.0]])
        result = batch_t_asy(X, beta=0.5)
        assert result[0, 0] == -1.0  # negative unchanged
        assert result[0, 1] > 2.0  # positive transformed (power > 1)

    def test_n_dim_inferred(self):
        """n_dim is inferred from X when not provided."""
        X = np.array([[1.0, 2.0, 3.0]])
        result = batch_t_asy(X, beta=0.2)
        assert result.shape == (1, 3)


class TestBatchFPen:
    """Tests for batch_f_pen (boundary penalty)."""

    def test_inside_bounds(self):
        """No penalty inside bounds."""
        X = np.array([[1.0, -1.0], [3.0, 4.0]])
        result = batch_f_pen(X, bound=5.0)
        np.testing.assert_allclose(result, [0.0, 0.0])

    def test_outside_bounds(self):
        """Penalty for values outside bounds."""
        X = np.array([[6.0, 0.0], [0.0, -7.0]])
        result = batch_f_pen(X, bound=5.0)
        # |6| - 5 = 1, penalty = 1^2 = 1
        assert result[0] == pytest.approx(1.0)
        # |-7| - 5 = 2, penalty = 2^2 = 4
        assert result[1] == pytest.approx(4.0)

    def test_on_boundary(self):
        """Exactly at boundary has no penalty."""
        X = np.array([[5.0, -5.0]])
        result = batch_f_pen(X, bound=5.0)
        np.testing.assert_allclose(result, [0.0])


class TestBatchLambdaAlpha:
    """Tests for batch_lambda_alpha (diagonal conditioning)."""

    def test_1d(self):
        """1D case returns ones scaling (no conditioning)."""
        X = np.array([[3.0]])
        result = batch_lambda_alpha(X, alpha=10.0, n_dim=1)
        # For 1D: diag = ones(1)
        np.testing.assert_allclose(result, [[3.0]])

    def test_nd(self):
        """Multi-dimensional case applies correct scaling."""
        X = np.ones((1, 3))
        result = batch_lambda_alpha(X, alpha=100.0, n_dim=3)
        # diag[0] = 100^0 = 1
        # diag[1] = 100^0.25
        # diag[2] = 100^0.5 = 10
        assert result[0, 0] == pytest.approx(1.0)
        assert result[0, 1] == pytest.approx(100**0.25)
        assert result[0, 2] == pytest.approx(10.0)

    def test_alpha_1_identity(self):
        """alpha=1 gives identity scaling."""
        X = np.array([[1.0, 2.0, 3.0]])
        result = batch_lambda_alpha(X, alpha=1.0)
        np.testing.assert_allclose(result, X)

    def test_n_dim_inferred(self):
        """n_dim is inferred from X when not provided."""
        X = np.array([[1.0, 2.0]])
        result = batch_lambda_alpha(X, alpha=10.0)
        assert result.shape == (1, 2)
