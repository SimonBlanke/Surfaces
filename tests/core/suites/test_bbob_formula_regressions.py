# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Formula regression tests for BBOB functions."""

import warnings

import numpy as np
import pytest

from surfaces.test_functions._batch_transforms import batch_t_osz
from surfaces.test_functions.bbob import Sphere, bbob_functions

pytestmark = pytest.mark.bbob


def _assert_no_runtime_warning(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()

    runtime_warnings = [
        warning for warning in caught if issubclass(warning.category, RuntimeWarning)
    ]
    assert runtime_warnings == []
    return result


def _reference_t_osz(x):
    x = np.asarray(x)
    x_hat = np.zeros_like(x, dtype=np.result_type(x, np.float64))
    np.log(np.abs(x), out=x_hat, where=x != 0)

    c1 = np.where(x > 0, 10.0, 5.5)
    c2 = np.where(x > 0, 7.9, 3.1)
    return np.sign(x) * np.exp(x_hat + 0.049 * (np.sin(c1 * x_hat) + np.sin(c2 * x_hat)))


def test_t_osz_is_warning_free_at_zero_and_matches_reference_values():
    func = Sphere(n_dim=12)
    values = np.array(
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            1e-300,
            -1e-300,
            np.finfo(float).tiny,
            -np.finfo(float).tiny,
            1e-12,
            -1e-12,
            10.0,
            -10.0,
        ]
    )

    result = _assert_no_runtime_warning(lambda: func.t_osz(values))
    expected = _reference_t_osz(values)

    np.testing.assert_allclose(result, expected, rtol=1e-15, atol=0.0)
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(0.0)


def test_t_osz_scalar_zero_is_warning_free():
    func = Sphere(n_dim=2)

    result = _assert_no_runtime_warning(lambda: func.t_osz(0.0))

    assert result == pytest.approx(0.0)


def test_batch_t_osz_is_warning_free_at_zero_and_matches_reference_values():
    X = np.array(
        [
            [0.0, -0.0, 1.0, -1.0, 1e-300, -1e-300],
            [np.finfo(float).tiny, -np.finfo(float).tiny, 1e-12, -1e-12, 10.0, -10.0],
        ]
    )

    result = _assert_no_runtime_warning(lambda: batch_t_osz(X))
    expected = _reference_t_osz(X)

    np.testing.assert_allclose(result, expected, rtol=1e-15, atol=0.0)


@pytest.mark.parametrize("func_class", bbob_functions, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("n_dim", [2, 5])
def test_bbob_x_opt_evaluates_to_f_opt_without_runtime_warnings(func_class, n_dim):
    func = func_class(n_dim=n_dim)
    params = {f"x{i}": func.x_opt[i] for i in range(n_dim)}

    scalar_result = _assert_no_runtime_warning(lambda: func._objective(params))
    batch_result = _assert_no_runtime_warning(
        lambda: np.asarray(func._batch_objective(func.x_opt[None, :]))
    )

    assert scalar_result == pytest.approx(func.f_opt, abs=1e-8, rel=1e-6)
    np.testing.assert_allclose(batch_result, [func.f_opt], atol=1e-8, rtol=1e-6)
