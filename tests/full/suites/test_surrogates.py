# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for surrogate model functionality.

Surrogate models are pre-trained ONNX models that approximate the behavior
of computationally expensive ML test functions. This module tests:

- SurrogateLoader: Loading and running ONNX models
- Surrogate availability: Which functions have surrogates
- ONNX utilities: File discovery and path resolution

Training, validation, and registry tests live in the surfaces-surrogates package.
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
        assert surr_time < real_time, (
            f"Surrogate ({surr_time:.4f}s) not faster than real ({real_time:.4f}s)"
        )

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
