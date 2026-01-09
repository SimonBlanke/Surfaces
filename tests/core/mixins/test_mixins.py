# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for BaseTestFunction mixins.

These tests verify that the extracted mixins work correctly when used
through a concrete test function (SphereFunction).
"""

import numpy as np
import pytest

from surfaces.test_functions.algebraic import SphereFunction
from surfaces.modifiers import GaussianNoise


class TestCallbackMixin:
    """Tests for CallbackMixin functionality."""

    def test_init_with_no_callbacks(self):
        """Test initialization without callbacks."""
        func = SphereFunction(n_dim=2)
        assert func.callbacks == []
        assert func._callbacks == []

    def test_init_with_single_callback(self):
        """Test initialization with a single callback function."""
        records = []
        func = SphereFunction(n_dim=2, callbacks=lambda r: records.append(r))

        func([1.0, 2.0])

        assert len(records) == 1
        assert "score" in records[0]

    def test_init_with_callback_list(self):
        """Test initialization with a list of callbacks."""
        records1 = []
        records2 = []
        func = SphereFunction(
            n_dim=2,
            callbacks=[
                lambda r: records1.append(r),
                lambda r: records2.append(r),
            ]
        )

        func([1.0, 2.0])

        assert len(records1) == 1
        assert len(records2) == 1

    def test_add_callback(self):
        """Test adding a callback after initialization."""
        func = SphereFunction(n_dim=2)
        records = []

        func.add_callback(lambda r: records.append(r))
        func([1.0, 2.0])

        assert len(records) == 1

    def test_remove_callback(self):
        """Test removing a callback."""
        func = SphereFunction(n_dim=2)
        records = []
        callback = lambda r: records.append(r)

        func.add_callback(callback)
        func([1.0, 2.0])
        assert len(records) == 1

        func.remove_callback(callback)
        func([3.0, 4.0])
        assert len(records) == 1  # No new records

    def test_remove_callback_not_found(self):
        """Test removing a callback that doesn't exist raises ValueError."""
        func = SphereFunction(n_dim=2)

        with pytest.raises(ValueError):
            func.remove_callback(lambda r: None)

    def test_clear_callbacks(self):
        """Test clearing all callbacks."""
        func = SphereFunction(n_dim=2)
        func.add_callback(lambda r: None)
        func.add_callback(lambda r: None)

        assert len(func.callbacks) == 2

        func.clear_callbacks()

        assert func.callbacks == []

    def test_callbacks_property_returns_copy(self):
        """Test that callbacks property returns a copy, not the internal list."""
        func = SphereFunction(n_dim=2)
        func.add_callback(lambda r: None)

        callbacks_copy = func.callbacks
        callbacks_copy.append(lambda r: None)

        assert len(func.callbacks) == 1  # Original unchanged


class TestDataCollectionMixin:
    """Tests for DataCollectionMixin functionality."""

    def test_init_data_collection(self):
        """Test that data collection attributes are initialized."""
        func = SphereFunction(n_dim=2)

        assert func.n_evaluations == 0
        assert func.search_data == []
        assert func.best_score is None
        assert func.best_params is None
        assert func.total_time == 0.0

    def test_record_evaluation_updates_counters(self):
        """Test that evaluation updates all counters."""
        func = SphereFunction(n_dim=2)

        func([1.0, 2.0])

        assert func.n_evaluations == 1
        assert len(func.search_data) == 1
        assert func.best_score is not None
        assert func.best_params is not None
        assert func.total_time > 0.0

    def test_search_data_contains_params_and_score(self):
        """Test that search_data contains parameters and score."""
        func = SphereFunction(n_dim=2)

        func([1.0, 2.0])

        record = func.search_data[0]
        assert "x0" in record
        assert "x1" in record
        assert "score" in record
        assert record["x0"] == 1.0
        assert record["x1"] == 2.0

    def test_best_score_minimize(self):
        """Test that best_score tracks minimum for minimize objective."""
        func = SphereFunction(n_dim=2, objective="minimize")

        func([3.0, 4.0])  # score = 25
        func([1.0, 1.0])  # score = 2
        func([2.0, 2.0])  # score = 8

        assert func.best_score == 2.0
        assert func.best_params == {"x0": 1.0, "x1": 1.0}

    def test_best_score_maximize(self):
        """Test that best_score tracks maximum for maximize objective."""
        func = SphereFunction(n_dim=2, objective="maximize")

        func([1.0, 1.0])  # score = -2
        func([3.0, 4.0])  # score = -25
        func([2.0, 2.0])  # score = -8

        assert func.best_score == -2.0
        assert func.best_params == {"x0": 1.0, "x1": 1.0}

    def test_reset_data_clears_all(self):
        """Test that reset_data clears all collected data."""
        func = SphereFunction(n_dim=2)

        func([1.0, 2.0])
        func([3.0, 4.0])

        assert func.n_evaluations == 2

        func.reset_data()

        assert func.n_evaluations == 0
        assert func.search_data == []
        assert func.best_score is None
        assert func.best_params is None
        assert func.total_time == 0.0

    def test_collect_data_false_disables_tracking(self):
        """Test that collect_data=False disables data tracking."""
        func = SphereFunction(n_dim=2, collect_data=False)

        func([1.0, 2.0])
        func([3.0, 4.0])

        assert func.n_evaluations == 0
        assert func.search_data == []
        assert func.best_score is None

    def test_callbacks_still_work_with_collect_data_false(self):
        """Test that callbacks work even when collect_data is False."""
        records = []
        func = SphereFunction(
            n_dim=2,
            collect_data=False,
            callbacks=lambda r: records.append(r)
        )

        func([1.0, 2.0])

        assert len(records) == 1
        assert func.n_evaluations == 0  # Data collection disabled


class TestModifierMixin:
    """Tests for ModifierMixin functionality."""

    def test_init_modifiers_empty(self):
        """Test initialization without modifiers."""
        func = SphereFunction(n_dim=2)

        assert func.modifiers == []

    def test_init_modifiers_with_list(self):
        """Test initialization with modifier list."""
        modifier = GaussianNoise(sigma=0.1, seed=42)
        func = SphereFunction(n_dim=2, modifiers=[modifier])

        assert len(func.modifiers) == 1
        assert func.modifiers[0] is modifier

    def test_modifiers_affect_evaluation(self):
        """Test that modifiers affect the evaluation result."""
        func_clean = SphereFunction(n_dim=2)
        func_noisy = SphereFunction(
            n_dim=2,
            modifiers=[GaussianNoise(sigma=1.0, seed=42)]
        )

        clean_result = func_clean([1.0, 2.0])
        noisy_result = func_noisy([1.0, 2.0])

        # Results should differ due to noise
        assert clean_result != noisy_result

    def test_true_value_bypasses_modifiers(self):
        """Test that true_value returns result without modifiers."""
        func = SphereFunction(
            n_dim=2,
            modifiers=[GaussianNoise(sigma=1.0, seed=42)]
        )

        true_result = func.true_value([1.0, 2.0])
        expected = 1.0**2 + 2.0**2  # Sphere function: sum of squares

        assert true_result == expected

    def test_true_value_does_not_update_search_data(self):
        """Test that true_value doesn't affect search_data."""
        func = SphereFunction(n_dim=2)

        func.true_value([1.0, 2.0])

        assert func.n_evaluations == 0
        assert func.search_data == []

    def test_reset_modifiers(self):
        """Test that reset_modifiers resets modifier state."""
        modifier = GaussianNoise(sigma=0.5, seed=42)
        func = SphereFunction(n_dim=2, modifiers=[modifier])

        # Evaluate to advance modifier state
        func([1.0, 2.0])
        func([1.0, 2.0])

        # Reset modifiers
        func.reset_modifiers()

        # Modifier should be reset (evaluation count back to 0)
        assert modifier._evaluation_count == 0


class TestVisualizationMixin:
    """Tests for VisualizationMixin functionality."""

    def test_dimensions_property(self):
        """Test that dimensions returns sorted parameter names."""
        func = SphereFunction(n_dim=3)

        assert func.dimensions == ["x0", "x1", "x2"]

    def test_default_plot_dims_returns_first_two(self):
        """Test that default_plot_dims returns first two dimensions."""
        func = SphereFunction(n_dim=5)

        assert func.default_plot_dims == ["x0", "x1"]

    def test_default_plot_dims_with_one_dimension(self):
        """Test default_plot_dims with 1D function."""
        func = SphereFunction(n_dim=1)

        assert func.default_plot_dims == ["x0"]

    def test_default_bounds_per_dim(self):
        """Test that default_bounds_per_dim returns bounds for each dimension."""
        func = SphereFunction(n_dim=2)

        bounds = func.default_bounds_per_dim

        assert "x0" in bounds
        assert "x1" in bounds
        assert isinstance(bounds["x0"], tuple)
        assert len(bounds["x0"]) == 2

    def test_default_fixed(self):
        """Test that default_fixed returns default values for each dimension."""
        func = SphereFunction(n_dim=3)

        fixed = func.default_fixed

        assert "x0" in fixed
        assert "x1" in fixed
        assert "x2" in fixed

    def test_default_step(self):
        """Test that default_step returns step sizes for each dimension."""
        func = SphereFunction(n_dim=2)

        step = func.default_step

        assert "x0" in step
        assert "x1" in step

    def test_infer_bounds_numeric_values(self):
        """Test bounds inference from numeric values."""
        func = SphereFunction(n_dim=2)

        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        bounds = func._infer_bounds_for_dimension(values)

        assert bounds == (1.0, 5.0)

    def test_infer_bounds_empty_values(self):
        """Test bounds inference with empty values."""
        func = SphereFunction(n_dim=2)

        bounds = func._infer_bounds_for_dimension([])

        assert bounds == (0.0, 1.0)

    def test_infer_bounds_categorical_values(self):
        """Test bounds inference from categorical (non-numeric) values."""
        func = SphereFunction(n_dim=2)

        values = ["a", "b", "c", "d"]
        bounds = func._infer_bounds_for_dimension(values)

        assert bounds == (0.0, 3.0)  # Index range

    def test_infer_fixed_middle_value(self):
        """Test fixed value inference uses middle value."""
        func = SphereFunction(n_dim=2)

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        fixed = func._infer_fixed_for_dimension(values)

        assert fixed == 3.0  # Middle value (index 2)

    def test_infer_fixed_empty_values(self):
        """Test fixed value inference with empty values."""
        func = SphereFunction(n_dim=2)

        fixed = func._infer_fixed_for_dimension([])

        assert fixed == 0.0

    def test_infer_step_continuous(self):
        """Test step inference for continuous (dense) arrays."""
        func = SphereFunction(n_dim=2)

        # Dense array with >50 values
        values = np.linspace(0, 10, 100)
        step = func._infer_step_for_dimension(values)

        assert step == pytest.approx(0.1, rel=0.01)  # (10-0)/100

    def test_infer_step_discrete(self):
        """Test step inference for discrete numeric values."""
        func = SphereFunction(n_dim=2)

        values = [1, 3, 5, 7, 9]
        step = func._infer_step_for_dimension(values)

        assert step == 2.0  # Minimum difference

    def test_infer_step_categorical(self):
        """Test step inference for categorical values returns None."""
        func = SphereFunction(n_dim=2)

        values = ["a", "b", "c"]
        step = func._infer_step_for_dimension(values)

        assert step is None

    def test_infer_step_single_value(self):
        """Test step inference with single value returns None."""
        func = SphereFunction(n_dim=2)

        step = func._infer_step_for_dimension([1.0])

        assert step is None

    def test_plot_property_returns_accessor(self):
        """Test that plot property returns PlotAccessor."""
        func = SphereFunction(n_dim=2)

        accessor = func.plot

        assert hasattr(accessor, "surface")
        assert hasattr(accessor, "contour")
        assert hasattr(accessor, "available")


class TestMixinIntegration:
    """Tests for mixin integration and interactions."""

    def test_reset_clears_data_and_memory(self):
        """Test that reset() clears both data and memory cache."""
        func = SphereFunction(n_dim=2, memory=True)

        func([1.0, 2.0])
        func([1.0, 2.0])  # Should be cached

        assert func.n_evaluations == 2
        assert len(func._memory_cache) == 1

        func.reset()

        assert func.n_evaluations == 0
        assert func._memory_cache == {}

    def test_mro_order(self):
        """Test that MRO includes all mixins in expected order."""
        mro_names = [cls.__name__ for cls in SphereFunction.__mro__]

        assert "CallbackMixin" in mro_names
        assert "DataCollectionMixin" in mro_names
        assert "ModifierMixin" in mro_names
        assert "VisualizationMixin" in mro_names

        # Verify order: mixins should come after BaseTestFunction
        base_idx = mro_names.index("BaseTestFunction")
        callback_idx = mro_names.index("CallbackMixin")

        assert callback_idx > base_idx

    def test_all_mixin_methods_accessible(self):
        """Test that all mixin methods are accessible on the function."""
        func = SphereFunction(n_dim=2)

        # CallbackMixin
        assert hasattr(func, "add_callback")
        assert hasattr(func, "remove_callback")
        assert hasattr(func, "clear_callbacks")
        assert hasattr(func, "callbacks")

        # DataCollectionMixin
        assert hasattr(func, "n_evaluations")
        assert hasattr(func, "search_data")
        assert hasattr(func, "best_score")
        assert hasattr(func, "reset_data")

        # ModifierMixin
        assert hasattr(func, "modifiers")
        assert hasattr(func, "true_value")
        assert hasattr(func, "reset_modifiers")

        # VisualizationMixin
        assert hasattr(func, "plot")
        assert hasattr(func, "dimensions")
        assert hasattr(func, "default_plot_dims")
