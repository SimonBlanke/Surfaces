"""Tests for fANOVA-based parameter importance (requires sklearn)."""

import numpy as np
import pytest

from surfaces import CustomTestFunction


@pytest.fixture
def sphere_func_with_data():
    """Sphere function with 50 evaluation data points."""
    func = CustomTestFunction(
        objective_fn=lambda p: sum(v**2 for v in p.values()),
        search_space={"x": (-5, 5), "y": (-5, 5)},
    )
    np.random.seed(42)
    for _ in range(50):
        x, y = np.random.uniform(-5, 5, 2)
        func({"x": x, "y": y})
    return func


@pytest.fixture
def asymmetric_func_with_data():
    """Function where x dominates: f = x^2 + 0.01*y. 100 evaluations."""
    func = CustomTestFunction(
        objective_fn=lambda p: p["x"] ** 2 + 0.01 * p["y"],
        search_space={"x": (-5, 5), "y": (-5, 5)},
    )
    rng = np.random.default_rng(42)
    for _ in range(100):
        func({"x": float(rng.uniform(-5, 5)), "y": float(rng.uniform(-5, 5))})
    return func


class TestFanovaImportance:
    """Test fANOVA-based parameter importance analysis."""

    def test_returns_correct_structure(self, sphere_func_with_data):
        """Returns dict with all param names, values >= 0, summing to 1."""
        importance = sphere_func_with_data.analysis.parameter_importance(method="fanova")
        assert set(importance.keys()) == {"x", "y"}
        assert all(v >= 0 for v in importance.values())
        assert sum(importance.values()) == pytest.approx(1.0, abs=1e-10)

    def test_identifies_dominant_parameter(self, asymmetric_func_with_data):
        """For f = x^2 + 0.01*y, fANOVA should assign x > 80% importance."""
        importance = asymmetric_func_with_data.analysis.parameter_importance(method="fanova")
        assert importance["x"] > 0.8

    def test_symmetric_function_equal_importance(self):
        """For f = x^2 + y^2, both parameters should get roughly equal weight."""
        func = CustomTestFunction(
            objective_fn=lambda p: p["x"] ** 2 + p["y"] ** 2,
            search_space={"x": (-5, 5), "y": (-5, 5)},
        )
        rng = np.random.default_rng(42)
        for _ in range(100):
            func({"x": float(rng.uniform(-5, 5)), "y": float(rng.uniform(-5, 5))})

        importance = func.analysis.parameter_importance(method="fanova")
        assert abs(importance["x"] - importance["y"]) < 0.25

    def test_requires_minimum_30_evaluations(self):
        """fANOVA should raise ValueError with fewer than 30 data points."""
        func = CustomTestFunction(
            objective_fn=lambda p: p["x"] ** 2,
            search_space={"x": (-5, 5)},
        )
        for i in range(20):
            func({"x": float(i) - 10.0})

        with pytest.raises(ValueError, match="requires at least 30"):
            func.analysis.parameter_importance(method="fanova")

    def test_outperforms_variance_on_nonlinear(self, asymmetric_func_with_data):
        """fANOVA should capture x^2 better than linear correlation does."""
        var_imp = asymmetric_func_with_data.analysis.parameter_importance(method="variance")
        fanova_imp = asymmetric_func_with_data.analysis.parameter_importance(method="fanova")
        assert fanova_imp["x"] > var_imp["x"]
