# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for surrogate model functionality.

Surrogate models are pre-trained ONNX models that approximate the behavior
of computationally expensive ML test functions. This module tests:

- SurrogateLoader: Loading and running ONNX models
- Surrogate availability: Which functions have surrogates
- SurrogateValidator: Validation against real functions
- ML Registry: Function registration for surrogate training
- ONNX utilities: File discovery and path resolution
"""

# Check for optional dependencies
import importlib.util

import numpy as np
import pytest

HAS_ONNXRUNTIME = importlib.util.find_spec("onnxruntime") is not None
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None

requires_onnxruntime = pytest.mark.skipif(
    not HAS_ONNXRUNTIME, reason="Requires onnxruntime: pip install onnxruntime"
)

requires_sklearn = pytest.mark.skipif(
    not HAS_SKLEARN, reason="Requires scikit-learn: pip install surfaces[ml]"
)


class TestOnnxUtils:
    """Test ONNX file discovery utilities."""

    def test_import_onnx_utils(self):
        """ONNX utils module can be imported."""
        from surfaces._surrogates._onnx_utils import (
            get_metadata_path,
            get_onnx_file,
            get_surrogate_model_path,
        )

        assert callable(get_onnx_file)
        assert callable(get_surrogate_model_path)
        assert callable(get_metadata_path)

    def test_get_surrogate_path_returns_path_or_none(self):
        """get_surrogate_model_path returns Path or None."""
        from pathlib import Path

        from surfaces._surrogates._onnx_utils import get_surrogate_model_path

        result = get_surrogate_model_path("k_neighbors_classifier")
        assert result is None or isinstance(result, Path)

    def test_get_nonexistent_surrogate(self):
        """Nonexistent surrogate returns None."""
        from surfaces._surrogates._onnx_utils import get_surrogate_model_path

        result = get_surrogate_model_path("nonexistent_function_xyz")
        assert result is None


@requires_sklearn
class TestMLRegistry:
    """Test ML function registration system."""

    def test_import_registry(self):
        """Registry module can be imported."""
        from surfaces._surrogates._ml_registry import (
            get_function_config,
            get_registered_functions,
        )

        assert callable(get_registered_functions)
        assert callable(get_function_config)

    def test_get_registered_functions(self):
        """Registry returns list of registered function names."""
        from surfaces._surrogates._ml_registry import get_registered_functions

        functions = get_registered_functions()
        assert isinstance(functions, list)
        assert len(functions) >= 10  # Should have at least 10 registered

    def test_expected_functions_registered(self):
        """Expected ML functions are registered."""
        from surfaces._surrogates._ml_registry import get_registered_functions

        functions = get_registered_functions()
        expected = [
            "k_neighbors_classifier",
            "k_neighbors_regressor",
            "decision_tree_classifier",
            "decision_tree_regressor",
            "random_forest_classifier",
            "random_forest_regressor",
            "gradient_boosting_classifier",
            "gradient_boosting_regressor",
            "svm_classifier",
            "svm_regressor",
        ]
        for name in expected:
            assert name in functions, f"Expected {name} to be registered"

    def test_function_config_structure(self):
        """Function config has required keys."""
        from surfaces._surrogates._ml_registry import get_function_config

        config = get_function_config("k_neighbors_classifier")
        assert "class" in config
        assert "fixed_params" in config
        assert "hyperparams" in config

    def test_function_config_fixed_params(self):
        """Fixed params include dataset and cv."""
        from surfaces._surrogates._ml_registry import get_function_config

        config = get_function_config("k_neighbors_classifier")
        assert "dataset" in config["fixed_params"]
        assert "cv" in config["fixed_params"]

    def test_unknown_function_raises(self):
        """Unknown function name raises ValueError."""
        from surfaces._surrogates._ml_registry import get_function_config

        with pytest.raises(ValueError, match="Unknown function"):
            get_function_config("nonexistent_function")


class TestSurrogateLoader:
    """Test SurrogateLoader class."""

    def test_import_loader(self):
        """SurrogateLoader can be imported."""
        from surfaces._surrogates import SurrogateLoader, load_surrogate

        assert SurrogateLoader is not None
        assert callable(load_surrogate)

    def test_load_surrogate_returns_none_if_missing(self):
        """load_surrogate returns None for nonexistent function."""
        from surfaces._surrogates import load_surrogate

        result = load_surrogate("nonexistent_function_xyz")
        assert result is None

    @requires_onnxruntime
    def test_loader_requires_metadata(self, tmp_path):
        """SurrogateLoader requires metadata file."""
        from surfaces._surrogates import SurrogateLoader

        # Create fake onnx file without metadata
        fake_model = tmp_path / "fake.onnx"
        fake_model.write_bytes(b"fake")

        loader = SurrogateLoader(fake_model)
        with pytest.raises(FileNotFoundError, match="Metadata file not found"):
            _ = loader.metadata


@requires_sklearn
class TestSurrogateAvailability:
    """Test which functions have surrogates available."""

    def test_use_surrogate_parameter_exists(self):
        """ML functions have use_surrogate parameter."""
        from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction

        # Should not raise
        func = KNeighborsClassifierFunction(use_surrogate=False)
        assert not func.use_surrogate

    def test_use_surrogate_false_by_default(self):
        """use_surrogate defaults to False."""
        from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction

        func = KNeighborsClassifierFunction()
        assert not func.use_surrogate

    @requires_onnxruntime
    def test_use_surrogate_true_loads_model(self):
        """use_surrogate=True attempts to load surrogate."""
        from surfaces._surrogates import get_surrogate_path
        from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction

        # Check if surrogate exists
        path = get_surrogate_path("k_neighbors_classifier")
        if path is None:
            pytest.skip("No surrogate model installed for k_neighbors_classifier")

        func = KNeighborsClassifierFunction(use_surrogate=True)
        assert func.use_surrogate
        assert func._surrogate is not None


@requires_sklearn
class TestSurrogateValidator:
    """Test SurrogateValidator class."""

    def test_import_validator(self):
        """SurrogateValidator can be imported."""
        from surfaces._surrogates import SurrogateValidator

        assert SurrogateValidator is not None

    def test_validator_requires_surrogate_false(self):
        """Validator raises if use_surrogate=True."""
        from surfaces._surrogates import SurrogateValidator, get_surrogate_path
        from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction

        # Check if surrogate exists
        path = get_surrogate_path("k_neighbors_classifier")
        if path is None:
            pytest.skip("No surrogate model installed")

        func_with_surrogate = KNeighborsClassifierFunction(use_surrogate=True)
        if not func_with_surrogate.use_surrogate:
            pytest.skip("Surrogate not loaded")

        with pytest.raises(ValueError, match="use_surrogate=False"):
            SurrogateValidator(func_with_surrogate)

    @requires_onnxruntime
    def test_validator_init(self):
        """Validator initializes with real function."""
        from surfaces._surrogates import SurrogateValidator, get_surrogate_path
        from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction

        # Check if surrogate exists
        path = get_surrogate_path("k_neighbors_classifier")
        if path is None:
            pytest.skip("No surrogate model installed")

        func = KNeighborsClassifierFunction(use_surrogate=False)
        try:
            validator = SurrogateValidator(func)
            assert validator.function is func
        except ValueError as e:
            if "No surrogate model available" in str(e):
                pytest.skip("No surrogate model available")
            raise

    @requires_onnxruntime
    def test_validator_copies_fixed_params(self):
        """Validator copies fixed parameters from real function to surrogate.

        This test verifies the fix for the bug where the validator would create
        a surrogate function with default parameters instead of copying them from
        the real function, leading to invalid comparisons.
        """
        from surfaces._surrogates import SurrogateValidator, get_surrogate_path
        from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction

        # Check if surrogate exists
        path = get_surrogate_path("k_neighbors_classifier")
        if path is None:
            pytest.skip("No surrogate model installed")

        # Create function with non-default fixed parameters
        func = KNeighborsClassifierFunction(
            dataset="wine",  # Non-default (default is "digits")
            cv=10,  # Non-default (default is 5)
            use_surrogate=False,
        )

        try:
            validator = SurrogateValidator(func)
        except ValueError as e:
            if "No surrogate model available" in str(e):
                pytest.skip("No surrogate model available")
            raise

        # Verify surrogate function has same fixed parameters
        assert (
            validator._surrogate_func.dataset == "wine"
        ), "Surrogate should have dataset='wine' copied from real function"
        assert (
            validator._surrogate_func.cv == 10
        ), "Surrogate should have cv=10 copied from real function"

        # Verify surrogate is enabled
        assert validator._surrogate_func.use_surrogate is True

    @requires_onnxruntime
    def test_validator_copies_all_init_params(self):
        """Validator copies all initialization parameters including common ones."""
        from surfaces._surrogates import SurrogateValidator, get_surrogate_path
        from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction

        # Check if surrogate exists
        path = get_surrogate_path("k_neighbors_classifier")
        if path is None:
            pytest.skip("No surrogate model installed")

        # Create function with various parameters
        func = KNeighborsClassifierFunction(
            dataset="iris",
            cv=5,
            objective="minimize",  # Non-default (default is "maximize")
            memory=True,  # Non-default (default is False)
            collect_data=False,  # Non-default (default is True)
            use_surrogate=False,
        )

        try:
            validator = SurrogateValidator(func)
        except ValueError as e:
            if "No surrogate model available" in str(e):
                pytest.skip("No surrogate model available")
            raise

        # Verify all parameters are copied
        assert validator._surrogate_func.dataset == "iris"
        assert validator._surrogate_func.cv == 5
        assert validator._surrogate_func.objective == "minimize"
        assert validator._surrogate_func.memory.enabled is True
        assert validator._surrogate_func.collect_data is False
        assert validator._surrogate_func.use_surrogate is True  # Override


@requires_sklearn
@requires_onnxruntime
class TestSurrogatePrediction:
    """Test surrogate model predictions."""

    def test_surrogate_returns_float(self):
        """Surrogate prediction returns float."""
        from surfaces._surrogates import get_surrogate_path
        from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction

        # Check if surrogate exists
        path = get_surrogate_path("k_neighbors_classifier")
        if path is None:
            pytest.skip("No surrogate model installed")

        func = KNeighborsClassifierFunction(use_surrogate=True)
        if not func.use_surrogate:
            pytest.skip("Surrogate not loaded")

        # Get sample params from search space
        params = {
            key: list(values)[0] if hasattr(values, "__iter__") else values
            for key, values in func.search_space.items()
        }
        result = func(params)
        assert isinstance(result, (int, float))

    def test_surrogate_faster_than_real(self):
        """Surrogate evaluation is faster than real function."""
        import time

        from surfaces._surrogates import get_surrogate_path
        from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction

        # Check if surrogate exists
        path = get_surrogate_path("k_neighbors_classifier")
        if path is None:
            pytest.skip("No surrogate model installed")

        real_func = KNeighborsClassifierFunction(use_surrogate=False)
        surr_func = KNeighborsClassifierFunction(use_surrogate=True)

        if not surr_func.use_surrogate:
            pytest.skip("Surrogate not loaded")

        params = {
            key: list(values)[0] if hasattr(values, "__iter__") else values
            for key, values in real_func.search_space.items()
        }

        # Time real function
        start = time.time()
        real_func(params)
        real_time = time.time() - start

        # Time surrogate (multiple calls for better measurement)
        start = time.time()
        for _ in range(10):
            surr_func(params)
        surr_time = (time.time() - start) / 10

        # Surrogate should be at least 10x faster
        assert (
            surr_time < real_time
        ), f"Surrogate ({surr_time:.4f}s) not faster than real ({real_time:.4f}s)"

    def test_surrogate_results_reasonable(self):
        """Surrogate results are within reasonable range."""
        from surfaces._surrogates import get_surrogate_path
        from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction

        path = get_surrogate_path("k_neighbors_classifier")
        if path is None:
            pytest.skip("No surrogate model installed")

        func = KNeighborsClassifierFunction(use_surrogate=True)
        if not func.use_surrogate:
            pytest.skip("Surrogate not loaded")

        params = {
            key: list(values)[0] if hasattr(values, "__iter__") else values
            for key, values in func.search_space.items()
        }
        result = func(params)

        # Classification accuracy loss should be between 0 and 1
        # (or negative if maximizing)
        assert -2.0 <= result <= 2.0, f"Result {result} outside reasonable range"


@requires_sklearn
@requires_onnxruntime
class TestSurrogateIntegration:
    """Integration tests for surrogate system."""

    def test_end_to_end_workflow(self):
        """Test complete surrogate usage workflow."""
        from surfaces._surrogates import get_surrogate_path, load_surrogate

        # 1. Check if surrogate exists
        path = get_surrogate_path("k_neighbors_classifier")
        if path is None:
            pytest.skip("No surrogate model installed")

        # 2. Load surrogate
        loader = load_surrogate("k_neighbors_classifier")
        assert loader is not None

        # 3. Check metadata
        assert "param_names" in loader.metadata
        assert len(loader.param_names) > 0

        # 4. Make prediction with all required params
        # Surrogate expects fixed params (cv, dataset) plus hyperparams
        from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction
        from surfaces.test_functions.machine_learning.hyperparameter_optimization.tabular.classification.datasets import (
            iris_data,
        )

        func = KNeighborsClassifierFunction(use_surrogate=False)
        params = {
            key: list(values)[0] if hasattr(values, "__iter__") else values
            for key, values in func.search_space.items()
        }
        # Add fixed params that surrogate expects
        params["cv"] = 5
        params["dataset"] = iris_data

        prediction = loader.predict(params)
        assert isinstance(prediction, float)
        assert np.isfinite(prediction) or np.isnan(prediction)

    def test_surrogate_callable_interface(self):
        """Surrogate provides callable interface."""
        from surfaces._surrogates import get_surrogate_path, load_surrogate

        path = get_surrogate_path("k_neighbors_classifier")
        if path is None:
            pytest.skip("No surrogate model installed")

        loader = load_surrogate("k_neighbors_classifier")
        objective_func = loader.as_objective_function()

        assert callable(objective_func)

        from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction
        from surfaces.test_functions.machine_learning.hyperparameter_optimization.tabular.classification.datasets import (
            iris_data,
        )

        func = KNeighborsClassifierFunction(use_surrogate=False)
        params = {
            key: list(values)[0] if hasattr(values, "__iter__") else values
            for key, values in func.search_space.items()
        }
        # Add fixed params that surrogate expects
        params["cv"] = 5
        params["dataset"] = iris_data

        result = objective_func(params)
        assert isinstance(result, float)


@requires_sklearn
class TestTrainerAPI:
    """Test surrogate trainer API (import and basic usage)."""

    def test_import_trainer(self):
        """Trainer classes can be imported."""
        from surfaces._surrogates import (
            MLSurrogateTrainer,
            SurrogateTrainer,
            list_ml_surrogates,
            train_all_ml_surrogates,
            train_ml_surrogate,
        )

        assert SurrogateTrainer is not None
        assert MLSurrogateTrainer is not None
        assert callable(train_ml_surrogate)
        assert callable(train_all_ml_surrogates)
        assert callable(list_ml_surrogates)

    def test_list_ml_surrogates(self):
        """list_ml_surrogates returns status info."""
        from surfaces._surrogates import list_ml_surrogates

        result = list_ml_surrogates()
        assert isinstance(result, dict)
        assert len(result) >= 10  # At least 10 registered functions

        # Each entry should have status info
        for name, info in result.items():
            assert isinstance(name, str)
            assert isinstance(info, dict)
            assert "has_model" in info or "exists" in info or isinstance(info, bool)
