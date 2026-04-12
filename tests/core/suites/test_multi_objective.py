# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for multi-objective optimization test functions."""

import numpy as np
import pytest

from surfaces.test_functions._base_test_function import BaseTestFunction
from surfaces.test_functions.algebraic.multi_objective import (
    DTLZ1,
    DTLZ2,
    DTLZ7,
    WFG1,
    ZDT1,
    ZDT2,
    ZDT4,
    FonsecaFleming,
    Kursawe,
    multi_objective_functions,
)
from surfaces.test_functions.algebraic.multi_objective.dtlz import dtlz_functions
from surfaces.test_functions.algebraic.multi_objective.wfg import wfg_functions
from surfaces.test_functions.algebraic.multi_objective.wfg._base_wfg import BaseWFGFunction
from surfaces.test_functions.algebraic.multi_objective.zdt import zdt_functions


def _make_func(func_class):
    """Instantiate a MO function with n_objectives=2 to keep shapes uniform."""
    if issubclass(func_class, BaseWFGFunction):
        return func_class(n_objectives=2, n_dim=3)
    if func_class in dtlz_functions:
        return func_class(n_objectives=2, n_dim=3)
    return func_class(n_dim=3)


class TestMultiObjectiveEvaluation:
    """Test evaluation of multi-objective functions."""

    @pytest.mark.parametrize("func_class", multi_objective_functions)
    def test_returns_ndarray(self, func_class):
        """Each function returns an ndarray."""
        func = _make_func(func_class)
        result = func(np.zeros(func.n_dim))
        assert isinstance(result, np.ndarray)

    @pytest.mark.parametrize("func_class", multi_objective_functions)
    def test_returns_correct_shape(self, func_class):
        """Each function returns shape (n_objectives,)."""
        func = _make_func(func_class)
        result = func(np.zeros(func.n_dim))
        assert result.shape == (func.n_objectives,)

    def test_zdt1_known_values(self):
        """ZDT1 at zeros: f1=0, f2=1."""
        func = ZDT1(n_dim=30)
        result = func(np.zeros(30))
        np.testing.assert_allclose(result, [0.0, 1.0], atol=1e-10)

    def test_zdt2_known_values(self):
        """ZDT2 at zeros: f1=0, f2=1."""
        func = ZDT2(n_dim=30)
        result = func(np.zeros(30))
        np.testing.assert_allclose(result, [0.0, 1.0], atol=1e-10)

    def test_zdt4_known_values(self):
        """ZDT4 at zeros: f1=0, g large (multimodal)."""
        func = ZDT4(n_dim=10)
        result = func(np.zeros(10))
        assert result[0] == 0.0
        assert result[1] == 1.0  # g=1 when x[1:]=0

    def test_dtlz1_pareto_sum(self):
        """DTLZ1 Pareto-optimal point: sum(f) = 0.5."""
        func = DTLZ1(n_objectives=3)
        x = np.full(func.n_dim, 0.5)
        result = func(x)
        np.testing.assert_allclose(np.sum(result), 0.5, atol=1e-10)

    def test_dtlz2_pareto_sphere(self):
        """DTLZ2 Pareto-optimal point: sum(f^2) = 1."""
        func = DTLZ2(n_objectives=3)
        x = np.full(func.n_dim, 0.5)
        result = func(x)
        np.testing.assert_allclose(np.sum(result**2), 1.0, atol=1e-10)

    def test_fonseca_fleming_at_optimum(self):
        """FonsecaFleming at 1/sqrt(n) should give f1 near 0."""
        func = FonsecaFleming(n_dim=3)
        offset = 1.0 / np.sqrt(3)
        x = np.full(3, offset)
        result = func(x)
        assert result[0] < 0.01

    @pytest.mark.parametrize("func_class", multi_objective_functions)
    def test_dict_input(self, func_class):
        """Functions accept dict input."""
        func = _make_func(func_class)
        params = {f"x{i}": 0.0 for i in range(func.n_dim)}
        result = func(params)
        assert result.shape == (func.n_objectives,)

    @pytest.mark.parametrize("func_class", multi_objective_functions)
    def test_list_input(self, func_class):
        """Functions accept list input."""
        func = _make_func(func_class)
        result = func([0.0] * func.n_dim)
        assert result.shape == (func.n_objectives,)

    @pytest.mark.parametrize("func_class", multi_objective_functions)
    def test_array_input(self, func_class):
        """Functions accept numpy array input."""
        func = _make_func(func_class)
        result = func(np.zeros(func.n_dim))
        assert result.shape == (func.n_objectives,)


class TestMultiObjectiveBatch:
    """Test batch evaluation for multi-objective functions."""

    @pytest.mark.parametrize("func_class", multi_objective_functions)
    def test_batch_returns_correct_shape(self, func_class):
        """Batch returns shape (n_points, n_objectives)."""
        func = _make_func(func_class)
        X = np.zeros((10, func.n_dim))
        result = func.batch(X)
        assert result.shape == (10, func.n_objectives)

    @pytest.mark.parametrize("func_class", multi_objective_functions)
    def test_batch_matches_sequential(self, func_class):
        """Batch matches sequential evaluation."""
        rng = np.random.default_rng(42)
        func = _make_func(func_class)

        lo, hi = func.spec.get("default_bounds", (0, 1))
        X = rng.uniform(lo, hi, size=(5, func.n_dim))

        batch_results = func.batch(X)
        sequential = np.array([func(X[i]) for i in range(5)])

        np.testing.assert_allclose(batch_results, sequential, rtol=1e-10)

    def test_batch_dimension_mismatch(self):
        """Wrong n_dim in batch raises ValueError."""
        func = ZDT1(n_dim=30)
        X = np.zeros((5, 10))
        with pytest.raises(ValueError, match="Expected 30 dimensions"):
            func.batch(X)

    def test_batch_1d_input_raises(self):
        """1D input raises ValueError."""
        func = ZDT1(n_dim=30)
        X = np.zeros(30)
        with pytest.raises(ValueError, match="Expected 2D"):
            func.batch(X)


class TestParetoFront:
    """Test Pareto front generation."""

    @pytest.mark.parametrize("func_class", multi_objective_functions)
    def test_pareto_front_shape(self, func_class):
        """Pareto front has correct shape."""
        func = _make_func(func_class)
        front = func.pareto_front(n_points=20)
        assert front.shape == (20, func.n_objectives)

    def test_zdt1_pareto_front_convex(self):
        """ZDT1 Pareto front: f2 = 1 - sqrt(f1)."""
        func = ZDT1(n_dim=30)
        front = func.pareto_front(n_points=100)
        expected_f2 = 1 - np.sqrt(front[:, 0])
        np.testing.assert_allclose(front[:, 1], expected_f2, atol=1e-10)

    def test_zdt2_pareto_front_concave(self):
        """ZDT2 Pareto front: f2 = 1 - f1^2."""
        func = ZDT2(n_dim=30)
        front = func.pareto_front(n_points=100)
        expected_f2 = 1 - front[:, 0] ** 2
        np.testing.assert_allclose(front[:, 1], expected_f2, atol=1e-10)

    def test_dtlz1_pareto_front_hyperplane(self):
        """DTLZ1 Pareto front lies on sum(f) = 0.5."""
        func = DTLZ1(n_objectives=3)
        front = func.pareto_front(n_points=50)
        np.testing.assert_allclose(np.sum(front, axis=1), 0.5, atol=1e-10)

    def test_dtlz2_pareto_front_sphere(self):
        """DTLZ2 Pareto front lies on unit sphere."""
        func = DTLZ2(n_objectives=3)
        front = func.pareto_front(n_points=50)
        norms = np.sum(front**2, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=0.1)

    def test_dtlz2_3obj_pareto_front(self):
        """DTLZ2 with 3 objectives: Pareto front in first orthant."""
        func = DTLZ2(n_objectives=3)
        front = func.pareto_front(n_points=50)
        assert front.shape == (50, 3)
        assert np.all(front >= -1e-10)

    def test_dtlz7_pareto_front_nonnegative(self):
        """DTLZ7 Pareto front values are non-negative."""
        func = DTLZ7(n_objectives=2)
        front = func.pareto_front(n_points=50)
        assert np.all(front >= -1e-10)


class TestParetoSet:
    """Test Pareto set generation."""

    @pytest.mark.parametrize("func_class", multi_objective_functions)
    def test_pareto_set_shape(self, func_class):
        """Pareto set has correct shape."""
        func = _make_func(func_class)
        pset = func.pareto_set(n_points=20)
        assert pset.shape == (20, func.n_dim)

    def test_zdt1_pareto_set_structure(self):
        """ZDT1 Pareto set: x2...xn = 0."""
        func = ZDT1(n_dim=30)
        pset = func.pareto_set(n_points=100)
        np.testing.assert_allclose(pset[:, 1:], 0.0, atol=1e-10)


class TestMultiObjectiveSpec:
    """Test function specifications."""

    @pytest.mark.parametrize("func_class", multi_objective_functions)
    def test_search_space(self, func_class):
        """Functions have a search space with correct dimensions."""
        func = _make_func(func_class)
        space = func.search_space
        assert isinstance(space, dict)
        assert len(space) == func.n_dim

    @pytest.mark.parametrize("func_class", multi_objective_functions)
    def test_spec_properties(self, func_class):
        """Functions have spec with expected keys."""
        func = _make_func(func_class)
        spec = func.spec
        assert "continuous" in spec
        assert "scalable" in spec

    def test_zdt1_requires_2_dims(self):
        """ZDT1 requires n_dim >= 2."""
        with pytest.raises(ValueError, match="n_dim must be >= 2"):
            ZDT1(n_dim=1)

    def test_kursawe_requires_2_dims(self):
        """Kursawe requires n_dim >= 2."""
        with pytest.raises(ValueError, match="n_dim must be >= 2"):
            Kursawe(n_dim=1)

    def test_dtlz_n_objectives_configurable(self):
        """DTLZ functions accept configurable n_objectives."""
        for M in [2, 3, 5]:
            func = DTLZ2(n_objectives=M)
            assert func.n_objectives == M
            assert func.spec.n_objectives == M

    def test_wfg_n_objectives_configurable(self):
        """WFG functions accept configurable n_objectives."""
        for M in [2, 3, 5]:
            func = WFG1(n_objectives=M)
            assert func.n_objectives == M
            assert func.spec.n_objectives == M

    def test_wfg_n_dim_override(self):
        """WFG accepts n_dim to override computed dimensions."""
        func = WFG1(n_objectives=2, n_dim=5)
        assert func.n_dim == 5
        assert func._k == 2
        assert func._n_dist == 3

    def test_wfg_n_dim_too_small_raises(self):
        """WFG raises ValueError when n_dim <= k."""
        with pytest.raises(ValueError, match="n_dim must be > k"):
            WFG1(n_objectives=3, n_dim=3)

    @pytest.mark.parametrize("func_class", zdt_functions)
    def test_zdt_always_2_objectives(self, func_class):
        """ZDT functions always have exactly 2 objectives."""
        func = func_class(n_dim=5)
        assert func.n_objectives == 2

    def test_function_count(self):
        """multi_objective_functions contains expected count."""
        assert len(multi_objective_functions) == 23

    def test_family_counts(self):
        """Each family has the expected number of functions."""
        assert len(zdt_functions) == 5
        assert len(dtlz_functions) == 7
        assert len(wfg_functions) == 9


class TestMultiObjectiveHierarchy:
    """Test that multi-objective functions integrate into the BaseTestFunction hierarchy."""

    @pytest.mark.parametrize("func_class", multi_objective_functions)
    def test_isinstance_base_test_function(self, func_class):
        """Multi-objective functions are instances of BaseTestFunction."""
        func = _make_func(func_class)
        assert isinstance(func, BaseTestFunction)

    @pytest.mark.parametrize("func_class", multi_objective_functions)
    def test_memory_caching(self, func_class):
        """Memory caching works for multi-objective functions."""
        func = _make_func(func_class)
        func = type(func)(memory=True, **_init_kwargs(func))
        params = np.zeros(func.n_dim)

        result1 = func(params)
        result2 = func(params)

        np.testing.assert_array_equal(result1, result2)

    @pytest.mark.parametrize("func_class", multi_objective_functions)
    def test_data_collection(self, func_class):
        """Data collection works for multi-objective functions."""
        func = _make_func(func_class)
        func = type(func)(collect_data=True, **_init_kwargs(func))

        func(np.zeros(func.n_dim))
        func(np.ones(func.n_dim) * 0.5)

        assert func.data.n_evaluations == 2
        assert len(func.data.search_data) == 2
        assert func.data.total_time >= 0

    @pytest.mark.parametrize("func_class", multi_objective_functions)
    def test_callbacks(self, func_class):
        """Callbacks are invoked for multi-objective functions."""
        records = []
        func = _make_func(func_class)
        func = type(func)(callbacks=lambda r: records.append(r), **_init_kwargs(func))

        func(np.zeros(func.n_dim))
        func(np.ones(func.n_dim) * 0.5)

        assert len(records) == 2
        assert "score" in records[0]


def _init_kwargs(func):
    """Extract constructor kwargs to recreate a function with different options."""
    if isinstance(func, BaseWFGFunction):
        return {"n_objectives": func.n_objectives, "n_dim": func.n_dim}
    if type(func) in dtlz_functions:
        return {"n_objectives": func.n_objectives, "n_dim": func.n_dim}
    return {"n_dim": func.n_dim}
