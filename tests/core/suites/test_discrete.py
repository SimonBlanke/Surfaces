# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for discrete (pseudo-boolean/combinatorial) test functions."""

import numpy as np
import pytest

from surfaces.test_functions.algebraic.discrete import (
    KnapsackFunction,
    LeadingOnesFunction,
    NKLandscapeFunction,
    OneMaxFunction,
    TrapFunction,
    discrete_functions,
)
from tests.conftest import func_id, get_sample_params


def make_discrete(func_class):
    """Instantiate discrete functions with their defaults (n_dim=20)."""
    return func_class()


@pytest.mark.discrete
class TestDiscreteInstantiation:
    """Test basic instantiation and evaluation."""

    @pytest.mark.parametrize("func_class", discrete_functions, ids=func_id)
    def test_instantiates(self, func_class):
        func = make_discrete(func_class)
        assert func is not None
        assert len(func.search_space) > 0

    @pytest.mark.parametrize("func_class", discrete_functions, ids=func_id)
    def test_evaluates(self, func_class):
        func = make_discrete(func_class)
        params = get_sample_params(func)
        result = func(params)
        assert isinstance(result, (int, float, np.integer, np.floating))

    @pytest.mark.parametrize("func_class", discrete_functions, ids=func_id)
    def test_returns_finite(self, func_class):
        func = make_discrete(func_class)
        params = get_sample_params(func)
        result = func(params)
        assert np.isfinite(result)


@pytest.mark.discrete
class TestDiscreteSearchSpace:
    """Test that search spaces are binary."""

    @pytest.mark.parametrize("func_class", discrete_functions, ids=func_id)
    def test_binary_values(self, func_class):
        func = make_discrete(func_class)
        for name, values in func.search_space.items():
            assert list(values) == [0, 1], f"Expected binary [0,1] for {name}"

    @pytest.mark.parametrize("func_class", discrete_functions, ids=func_id)
    def test_n_dim_matches_search_space(self, func_class):
        func = make_discrete(func_class)
        assert len(func.search_space) == func.n_dim


@pytest.mark.discrete
class TestDiscreteSpec:
    """Test spec properties for discrete functions."""

    @pytest.mark.parametrize("func_class", discrete_functions, ids=func_id)
    def test_not_continuous(self, func_class):
        func = make_discrete(func_class)
        assert func.spec.continuous is False

    @pytest.mark.parametrize("func_class", discrete_functions, ids=func_id)
    def test_not_differentiable(self, func_class):
        func = make_discrete(func_class)
        assert func.spec.differentiable is False


@pytest.mark.discrete
class TestOneMax:
    """Test OneMax function specifics."""

    def test_all_ones_is_global_minimum(self):
        func = OneMaxFunction(n_dim=10)
        params = {f"x{i}": 1 for i in range(10)}
        assert func(params) == 0

    def test_all_zeros_is_worst(self):
        func = OneMaxFunction(n_dim=10)
        params = {f"x{i}": 0 for i in range(10)}
        assert func(params) == 10

    def test_partial(self):
        func = OneMaxFunction(n_dim=5)
        params = {"x0": 1, "x1": 0, "x2": 1, "x3": 0, "x4": 1}
        assert func(params) == 2

    def test_batch_matches_sequential(self):
        func = OneMaxFunction(n_dim=8)
        X = np.array([[1, 0, 1, 0, 1, 0, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1]])
        batch_results = func.batch(X)
        seq_results = [func.pure(row) for row in X]
        np.testing.assert_array_equal(batch_results, seq_results)


@pytest.mark.discrete
class TestLeadingOnes:
    """Test LeadingOnes function specifics."""

    def test_all_ones(self):
        func = LeadingOnesFunction(n_dim=5)
        params = {f"x{i}": 1 for i in range(5)}
        assert func(params) == 0

    def test_leading_three(self):
        func = LeadingOnesFunction(n_dim=5)
        params = {"x0": 1, "x1": 1, "x2": 1, "x3": 0, "x4": 1}
        assert func(params) == 2

    def test_no_leading_ones(self):
        func = LeadingOnesFunction(n_dim=5)
        params = {"x0": 0, "x1": 1, "x2": 1, "x3": 1, "x4": 1}
        assert func(params) == 5

    def test_batch_matches_sequential(self):
        func = LeadingOnesFunction(n_dim=6)
        X = np.array([[1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
        batch_results = func.batch(X)
        seq_results = [func.pure(row) for row in X]
        np.testing.assert_array_equal(batch_results, seq_results)


@pytest.mark.discrete
class TestNKLandscape:
    """Test NK Landscape function specifics."""

    def test_k_zero_is_separable(self):
        """With K=0, each variable contributes independently."""
        func = NKLandscapeFunction(n_dim=10, k=0, seed=42)
        assert func.k == 0

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError, match="k must satisfy"):
            NKLandscapeFunction(n_dim=5, k=5)

    def test_result_bounded(self):
        """Loss should be in [0, 1] since fitness contributions are in [0, 1]."""
        func = NKLandscapeFunction(n_dim=10, k=3, seed=42)
        for _ in range(20):
            bits = np.random.randint(0, 2, size=10)
            params = {f"x{i}": int(bits[i]) for i in range(10)}
            result = func(params)
            assert 0.0 <= result <= 1.0

    def test_reproducible_with_same_seed(self):
        func1 = NKLandscapeFunction(n_dim=10, k=2, seed=123)
        func2 = NKLandscapeFunction(n_dim=10, k=2, seed=123)
        params = {f"x{i}": i % 2 for i in range(10)}
        assert func1(params) == func2(params)

    def test_different_seeds_differ(self):
        func1 = NKLandscapeFunction(n_dim=10, k=2, seed=1)
        func2 = NKLandscapeFunction(n_dim=10, k=2, seed=2)
        params = {f"x{i}": 1 for i in range(10)}
        assert func1(params) != func2(params)

    def test_batch_matches_sequential(self):
        func = NKLandscapeFunction(n_dim=8, k=2, seed=42)
        X = np.array([[1, 0, 1, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
        batch_results = func.batch(X)
        seq_results = [func.pure(row) for row in X]
        np.testing.assert_allclose(batch_results, seq_results)


@pytest.mark.discrete
class TestTrap:
    """Test Trap function specifics."""

    def test_all_ones_is_global_minimum(self):
        func = TrapFunction(n_dim=10, trap_size=5)
        params = {f"x{i}": 1 for i in range(10)}
        assert func(params) == 0

    def test_all_zeros_deceptive_attractor(self):
        """All zeros should score well (deceptive), but not optimal."""
        func = TrapFunction(n_dim=10, trap_size=5)
        params = {f"x{i}": 0 for i in range(10)}
        result = func(params)
        assert result > 0
        # Each block: trap(0, 5) = 4, so total trap = 8, loss = 10 - 8 = 2
        assert result == 2

    def test_near_optimum_is_worst(self):
        """4 out of 5 ones in a block is the worst trap value (0)."""
        func = TrapFunction(n_dim=5, trap_size=5)
        params = {"x0": 1, "x1": 1, "x2": 1, "x3": 1, "x4": 0}
        # u=4, trap(4,5) = 5-1-4 = 0, loss = 5 - 0 = 5
        assert func(params) == 5

    def test_invalid_n_dim_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            TrapFunction(n_dim=7, trap_size=5)

    def test_batch_matches_sequential(self):
        func = TrapFunction(n_dim=10, trap_size=5)
        X = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
            ]
        )
        batch_results = func.batch(X)
        seq_results = [func.pure(row) for row in X]
        np.testing.assert_array_equal(batch_results, seq_results)

    def test_batch_uses_input_array_namespace_for_where(self, monkeypatch):
        import surfaces.test_functions.algebraic.discrete.trap as trap_module

        class NamespaceSpy:
            def __init__(self):
                self.where_called = False

            def zeros(self, *args, **kwargs):
                return np.zeros(*args, **kwargs)

            def sum(self, *args, **kwargs):
                return np.sum(*args, **kwargs)

            def where(self, *args, **kwargs):
                self.where_called = True
                return np.where(*args, **kwargs)

        spy = NamespaceSpy()
        monkeypatch.setattr(trap_module, "get_array_namespace", lambda X: spy)

        func = TrapFunction(n_dim=5, trap_size=5)
        result = func.batch(np.array([[1, 1, 1, 1, 1]]))

        assert spy.where_called is True
        np.testing.assert_array_equal(result, np.array([0]))


@pytest.mark.discrete
class TestKnapsack:
    """Test Knapsack function specifics."""

    def test_empty_selection(self):
        """Selecting nothing should give value 0 (loss = 0) with no penalty."""
        func = KnapsackFunction(n_items=5, seed=42)
        params = {f"x{i}": 0 for i in range(5)}
        assert func(params) == 0.0

    def test_penalty_for_overweight(self):
        """Selecting everything on a tight capacity should incur a penalty."""
        weights = [10, 10, 10, 10, 10]
        values = [5, 5, 5, 5, 5]
        func = KnapsackFunction(n_items=5, weights=weights, values=values, capacity=20.0)
        params = {f"x{i}": 1 for i in range(5)}
        result = func(params)
        # value = 25, overweight = 50 - 20 = 30, penalty = 10 * 900 = 9000
        assert result == -25 + 9000

    def test_feasible_no_penalty(self):
        """A feasible selection should have loss = -value."""
        weights = [10, 10, 10, 10, 10]
        values = [20, 30, 40, 50, 60]
        func = KnapsackFunction(n_items=5, weights=weights, values=values, capacity=25.0)
        # Select items 0 and 1: weight = 20 <= 25
        params = {"x0": 1, "x1": 1, "x2": 0, "x3": 0, "x4": 0}
        assert func(params) == -50.0

    def test_reproducible_with_same_seed(self):
        func1 = KnapsackFunction(n_items=10, seed=99)
        func2 = KnapsackFunction(n_items=10, seed=99)
        params = {f"x{i}": i % 2 for i in range(10)}
        assert func1(params) == func2(params)

    def test_batch_matches_sequential(self):
        func = KnapsackFunction(n_items=8, seed=42)
        X = np.array([[1, 0, 1, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
        batch_results = func.batch(X)
        seq_results = [func.pure(row) for row in X]
        np.testing.assert_allclose(batch_results, seq_results)


@pytest.mark.discrete
class TestDiscreteInputFormats:
    """Test that discrete functions accept various input formats."""

    @pytest.mark.parametrize("func_class", discrete_functions, ids=func_id)
    def test_dict_input(self, func_class):
        func = make_discrete(func_class)
        params = get_sample_params(func)
        result = func(params)
        assert np.isfinite(result)

    @pytest.mark.parametrize("func_class", discrete_functions, ids=func_id)
    def test_array_input(self, func_class):
        func = make_discrete(func_class)
        arr = np.zeros(func.n_dim)
        result = func(arr)
        assert np.isfinite(result)

    @pytest.mark.parametrize("func_class", discrete_functions, ids=func_id)
    def test_list_input(self, func_class):
        func = make_discrete(func_class)
        lst = [0] * func.n_dim
        result = func(lst)
        assert np.isfinite(result)
