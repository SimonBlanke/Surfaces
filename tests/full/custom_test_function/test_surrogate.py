# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for CustomTestFunction surrogate namespace (requires scikit-learn)."""

import numpy as np
import pytest

from surfaces.custom_test_function import CustomTestFunction


@pytest.fixture
def sphere_func():
    """Simple sphere function for testing."""

    def sphere(params):
        return sum(v**2 for v in params.values())

    return CustomTestFunction(
        objective_fn=sphere,
        search_space={"x": (-5, 5), "y": (-5, 5)},
    )


@pytest.fixture
def sphere_func_with_data(sphere_func):
    """Sphere function with evaluation data."""
    np.random.seed(42)
    for _ in range(50):
        x, y = np.random.uniform(-5, 5, 2)
        sphere_func({"x": x, "y": y})
    return sphere_func


class TestSurrogateNamespace:
    """Test surrogate namespace methods."""

    def test_fit_random_forest(self, sphere_func_with_data):
        """Test Random Forest surrogate fitting."""
        sphere_func_with_data.surrogate.fit(method="random_forest")

        assert sphere_func_with_data.surrogate.is_fitted
        assert sphere_func_with_data.surrogate.method == "random_forest"

    def test_predict(self, sphere_func_with_data):
        """Test surrogate prediction."""
        sphere_func_with_data.surrogate.fit(method="random_forest")

        # Predict at origin (should be near 0)
        pred = sphere_func_with_data.surrogate.predict({"x": 0, "y": 0})
        assert pred < 5  # Should be small

    def test_predict_array(self, sphere_func_with_data):
        """Test surrogate prediction with array input."""
        sphere_func_with_data.surrogate.fit(method="random_forest")

        X = np.array([[0, 0], [1, 1], [2, 2]])
        preds = sphere_func_with_data.surrogate.predict(X)

        assert len(preds) == 3

    def test_suggest_next(self, sphere_func_with_data):
        """Test next point suggestion."""
        sphere_func_with_data.surrogate.fit(method="random_forest")

        suggestions = sphere_func_with_data.surrogate.suggest_next(n_suggestions=3)

        assert len(suggestions) == 3
        assert all("x" in s and "y" in s for s in suggestions)

    def test_score(self, sphere_func_with_data):
        """Test surrogate R^2 score."""
        sphere_func_with_data.surrogate.fit(method="random_forest")

        score = sphere_func_with_data.surrogate.score()
        assert 0 < score <= 1  # R^2 should be positive for good fit

    def test_not_fitted_error(self, sphere_func_with_data):
        """Test error when predicting without fitting."""
        with pytest.raises(RuntimeError, match="No surrogate model fitted"):
            sphere_func_with_data.surrogate.predict({"x": 0, "y": 0})
