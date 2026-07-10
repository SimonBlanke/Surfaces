import numpy as np
import pytest

from surfaces import CustomTestFunction
from surfaces.test_functions.algebraic.constrained import (
    PressureVesselFunction,
    constrained_functions,
)
from tests.conftest import func_id

pytestmark = pytest.mark.engineering


def _params_from_vector(func, vector):
    return {name: float(vector[idx]) for idx, name in enumerate(func.variable_names)}


def _midpoint_vector(func):
    return np.array([(lb + ub) / 2.0 for lb, ub in func.variable_bounds], dtype=float)


def test_default_array_input_contract_remains_sorted_search_space():
    func = CustomTestFunction(
        lambda params: 10.0 * params["a"] + params["b"],
        search_space={"b": [0.0, 1.0], "a": [0.0, 1.0]},
        collect_data=False,
    )

    assert func([1.0, 2.0]) == 12.0


def test_pressure_vessel_declares_continuous_relaxation_optimum():
    func = PressureVesselFunction(collect_data=False)

    np.testing.assert_allclose(
        func.x_global,
        [0.7781686413751053, 0.3846491626279018, 40.31961872409872, 200.0],
        rtol=0,
        atol=0,
    )
    assert func.f_global == pytest.approx(5885.332773616459, rel=0, abs=0)


@pytest.mark.parametrize("func_class", constrained_functions, ids=func_id)
def test_engineering_array_input_uses_variable_names(func_class):
    func = func_class(collect_data=False)
    x = np.asarray(func.x_global, dtype=float)
    params = _params_from_vector(func, x)
    expected = func(params)

    np.testing.assert_allclose(func(x), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(func(list(x)), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(func(tuple(x)), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("func_class", constrained_functions, ids=func_id)
def test_engineering_batch_matches_dict_and_array_inputs(func_class):
    func = func_class(collect_data=False)
    x_opt = np.asarray(func.x_global, dtype=float)
    x_mid = _midpoint_vector(func)
    X = np.vstack([x_opt, x_mid])

    batch_result = func.batch(X)
    dict_result = np.array([func(_params_from_vector(func, row)) for row in X])
    array_result = np.array([func(row) for row in X])

    np.testing.assert_allclose(batch_result, dict_result, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(array_result, dict_result, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("func_class", constrained_functions, ids=func_id)
def test_declared_engineering_optimum_is_feasible_and_penalty_free(func_class):
    func = func_class(collect_data=False)
    x = np.asarray(func.x_global, dtype=float)
    params = _params_from_vector(func, x)

    assert func.is_feasible(params)
    assert func.constraint_violations(params) == [0] * len(func.constraints(params))
    assert func.penalty(params) == 0.0
    np.testing.assert_allclose(func.raw_objective(params), func.f_global, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(func(params), func.f_global, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        func.batch(x.reshape(1, -1))[0], func.f_global, rtol=1e-10, atol=1e-12
    )
