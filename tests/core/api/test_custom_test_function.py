# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for CustomTestFunction (private API)."""

import numpy as np
import pytest

from surfaces.test_functions._custom_test_function import CustomTestFunction
from surfaces.modifiers import GaussianNoise


class TestCustomTestFunctionBasic:
    """Basic functionality tests."""

    def test_simple_function(self):
        """Test with a simple sphere-like function."""

        def sphere(params):
            return sum(v**2 for v in params.values())

        func = CustomTestFunction(
            objective_fn=sphere,
            search_space={"x": (-5, 5), "y": (-5, 5)},
        )

        result = func(x=0, y=0)
        assert result == 0

        result = func(x=1, y=2)
        assert result == 5

    def test_dict_input(self):
        """Test dict input format."""

        def sphere(params):
            return params["x"] ** 2 + params["y"] ** 2

        func = CustomTestFunction(
            objective_fn=sphere,
            search_space={"x": (-5, 5), "y": (-5, 5)},
        )

        result = func({"x": 3, "y": 4})
        assert result == 25

    def test_array_input(self):
        """Test array input format."""

        def sphere(params):
            return params["x"] ** 2 + params["y"] ** 2

        func = CustomTestFunction(
            objective_fn=sphere,
            search_space={"x": (-5, 5), "y": (-5, 5)},
        )

        result = func(np.array([3, 4]))
        assert result == 25

    def test_list_input(self):
        """Test list input format."""

        def sphere(params):
            return params["x"] ** 2 + params["y"] ** 2

        func = CustomTestFunction(
            objective_fn=sphere,
            search_space={"x": (-5, 5), "y": (-5, 5)},
        )

        result = func([3, 4])
        assert result == 25


class TestSearchSpace:
    """Tests for search space handling."""

    def test_bounds_tuple(self):
        """Test search space with bounds tuples."""
        func = CustomTestFunction(
            objective_fn=lambda p: 0,
            search_space={"x": (-5, 5), "y": (-10, 10)},
            resolution=50,
        )

        assert len(func.search_space["x"]) == 50
        assert len(func.search_space["y"]) == 50
        assert func.search_space["x"][0] == -5
        assert func.search_space["x"][-1] == pytest.approx(5, rel=0.1)
        assert func.search_space["y"][0] == -10
        assert func.search_space["y"][-1] == pytest.approx(10, rel=0.1)

    def test_explicit_arrays(self):
        """Test search space with explicit arrays."""
        x_vals = np.array([1, 2, 3, 4, 5])
        y_vals = np.linspace(0, 100, 1000)

        func = CustomTestFunction(
            objective_fn=lambda p: 0,
            search_space={"x": x_vals, "y": y_vals},
        )

        np.testing.assert_array_equal(func.search_space["x"], x_vals)
        np.testing.assert_array_equal(func.search_space["y"], y_vals)

    def test_mixed_search_space(self):
        """Test search space with mixed definitions."""
        func = CustomTestFunction(
            objective_fn=lambda p: 0,
            search_space={
                "x": (-5, 5),  # bounds
                "y": np.array([1, 2, 3]),  # explicit array
                "z": [10, 20, 30],  # list
            },
            resolution=100,
        )

        assert len(func.search_space["x"]) == 100
        np.testing.assert_array_equal(func.search_space["y"], [1, 2, 3])
        np.testing.assert_array_equal(func.search_space["z"], [10, 20, 30])

    def test_n_dim(self):
        """Test n_dim property."""
        func = CustomTestFunction(
            objective_fn=lambda p: 0,
            search_space={"x": (-5, 5), "y": (-5, 5), "z": (-5, 5)},
        )
        assert func.n_dim == 3

    def test_param_names(self):
        """Test param_names property."""
        func = CustomTestFunction(
            objective_fn=lambda p: 0,
            search_space={"b": (-5, 5), "a": (-5, 5), "c": (-5, 5)},
        )
        assert func.param_names == ["a", "b", "c"]


class TestGlobalOptimum:
    """Tests for global optimum handling."""

    def test_global_optimum_provided(self):
        """Test with global optimum provided."""
        func = CustomTestFunction(
            objective_fn=lambda p: p["x"] ** 2,
            search_space={"x": (-5, 5)},
            global_optimum={"position": {"x": 0}, "score": 0},
        )

        assert func.f_global == 0
        assert func.x_global == {"x": 0}

    def test_no_global_optimum(self):
        """Test without global optimum."""
        func = CustomTestFunction(
            objective_fn=lambda p: p["x"] ** 2,
            search_space={"x": (-5, 5)},
        )

        assert func.f_global is None
        assert func.x_global is None


class TestModifiers:
    """Tests for modifier integration."""

    def test_with_gaussian_noise(self):
        """Test with Gaussian noise modifier."""

        def constant(params):
            return 10.0

        func = CustomTestFunction(
            objective_fn=constant,
            search_space={"x": (-5, 5)},
            modifiers=[GaussianNoise(sigma=1.0, seed=42)],
        )

        results = [func(x=0) for _ in range(100)]
        mean = np.mean(results)
        std = np.std(results)

        assert mean == pytest.approx(10.0, abs=0.5)
        assert std == pytest.approx(1.0, abs=0.3)

    def test_true_value(self):
        """Test true_value bypasses modifiers."""

        def constant(params):
            return 10.0

        func = CustomTestFunction(
            objective_fn=constant,
            search_space={"x": (-5, 5)},
            modifiers=[GaussianNoise(sigma=1.0, seed=42)],
        )

        true = func.true_value(x=0)
        assert true == 10.0


class TestDataCollection:
    """Tests for data collection features."""

    def test_n_evaluations(self):
        """Test evaluation counting."""
        func = CustomTestFunction(
            objective_fn=lambda p: p["x"] ** 2,
            search_space={"x": (-5, 5)},
        )

        assert func.n_evaluations == 0
        func(x=1)
        assert func.n_evaluations == 1
        func(x=2)
        func(x=3)
        assert func.n_evaluations == 3

    def test_search_data(self):
        """Test search data collection."""
        func = CustomTestFunction(
            objective_fn=lambda p: p["x"] ** 2,
            search_space={"x": (-5, 5)},
        )

        func(x=2)
        func(x=3)

        assert len(func.search_data) == 2
        assert func.search_data[0] == {"x": 2, "score": 4}
        assert func.search_data[1] == {"x": 3, "score": 9}

    def test_best_score_minimize(self):
        """Test best score tracking for minimization."""
        func = CustomTestFunction(
            objective_fn=lambda p: p["x"] ** 2,
            search_space={"x": (-5, 5)},
            objective="minimize",
        )

        func(x=3)
        assert func.best_score == 9
        func(x=1)
        assert func.best_score == 1
        func(x=5)
        assert func.best_score == 1  # Still 1

    def test_best_score_maximize(self):
        """Test best score tracking for maximization."""
        func = CustomTestFunction(
            objective_fn=lambda p: p["x"] ** 2,
            search_space={"x": (-5, 5)},
            objective="maximize",
        )

        func(x=1)  # raw=1, returned=-1
        func(x=3)  # raw=9, returned=-9
        func(x=2)  # raw=4, returned=-4

        # For maximize, best_score tracks highest returned value
        # (scores are negated, so -1 > -9)
        assert func.best_score == -1

    def test_reset(self):
        """Test reset clears all data."""
        func = CustomTestFunction(
            objective_fn=lambda p: p["x"] ** 2,
            search_space={"x": (-5, 5)},
        )

        func(x=1)
        func(x=2)
        func.reset()

        assert func.n_evaluations == 0
        assert func.search_data == []
        assert func.best_score is None


class TestMemory:
    """Tests for memory caching."""

    def test_memory_caching(self):
        """Test memory caching avoids recomputation."""
        call_count = [0]

        def counting_sphere(params):
            call_count[0] += 1
            return params["x"] ** 2

        func = CustomTestFunction(
            objective_fn=counting_sphere,
            search_space={"x": (-5, 5)},
            memory=True,
        )

        func(x=2)
        assert call_count[0] == 1

        func(x=2)  # Should use cache
        assert call_count[0] == 1

        func(x=3)  # New point
        assert call_count[0] == 2


class TestCallbacks:
    """Tests for callback functionality."""

    def test_callback_invoked(self):
        """Test callbacks are invoked after evaluation."""
        records = []

        func = CustomTestFunction(
            objective_fn=lambda p: p["x"] ** 2,
            search_space={"x": (-5, 5)},
            callbacks=lambda r: records.append(r),
        )

        func(x=3)
        func(x=4)

        assert len(records) == 2
        assert records[0] == {"x": 3, "score": 9}
        assert records[1] == {"x": 4, "score": 16}


class TestCatchErrors:
    """Tests for error handling."""

    def test_catch_specific_error(self):
        """Test catching specific exception types."""

        def risky(params):
            if params["x"] == 0:
                raise ZeroDivisionError()
            return 1 / params["x"]

        func = CustomTestFunction(
            objective_fn=risky,
            search_space={"x": (-5, 5)},
            catch_errors={ZeroDivisionError: float("inf")},
        )

        assert func(x=0) == float("inf")
        assert func(x=2) == 0.5

    def test_catch_all_errors(self):
        """Test catch-all with ellipsis."""

        def risky(params):
            raise ValueError("oops")

        func = CustomTestFunction(
            objective_fn=risky,
            search_space={"x": (-5, 5)},
            catch_errors={...: -999},
        )

        assert func(x=0) == -999


class TestRepr:
    """Tests for string representation."""

    def test_repr(self):
        """Test __repr__ output."""
        func = CustomTestFunction(
            objective_fn=lambda p: 0,
            search_space={"x": (-5, 5), "y": (-10, 10)},
        )

        repr_str = repr(func)
        assert "CustomTestFunction" in repr_str
        assert "n_dim=2" in repr_str
        assert "minimize" in repr_str
