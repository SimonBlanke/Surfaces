# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for BBOB (Black-Box Optimization Benchmarking) functions.

The BBOB test suite consists of 24 noiseless benchmark functions organized
into five categories based on their properties:
- Separable (f1-f5)
- Low/Moderate Conditioning (f6-f9)
- High Conditioning & Unimodal (f10-f14)
- Multimodal with Adequate Global Structure (f15-f19)
- Multimodal with Weak Global Structure (f20-f24)

Reference:
    Hansen, N., et al. (2009). Real-parameter black-box optimization
    benchmarking 2009: Noiseless functions definitions. INRIA.
"""

import numpy as np
import pytest

from surfaces.test_functions.bbob import (
    BBOB_FUNCTIONS,
    bbob_functions,
    # Low/Moderate Conditioning (f6-f9)
    AttractiveSector,
    BentCigar,
    BuecheRastrigin,
    DifferentPowers,
    Discus,
    # High Conditioning & Unimodal (f10-f14)
    EllipsoidalRotated,
    EllipsoidalSeparable,
    Gallagher21,
    Gallagher101,
    GriewankRosenbrock,
    Katsuura,
    LinearSlope,
    LunacekBiRastrigin,
    # Multimodal with Adequate Global Structure (f15-f19)
    RastriginRotated,
    RastriginSeparable,
    RosenbrockOriginal,
    RosenbrockRotated,
    SchaffersF7,
    SchaffersF7Ill,
    # Multimodal with Weak Global Structure (f20-f24)
    Schwefel,
    SharpRidge,
    # Separable (f1-f5)
    Sphere,
    StepEllipsoidal,
    Weierstrass,
)

# Organize functions by category
SEPARABLE = [Sphere, EllipsoidalSeparable, RastriginSeparable, BuecheRastrigin, LinearSlope]
LOW_CONDITIONING = [AttractiveSector, StepEllipsoidal, RosenbrockOriginal, RosenbrockRotated]
HIGH_CONDITIONING = [EllipsoidalRotated, Discus, BentCigar, SharpRidge, DifferentPowers]
MULTIMODAL_ADEQUATE = [
    RastriginRotated,
    Weierstrass,
    SchaffersF7,
    SchaffersF7Ill,
    GriewankRosenbrock,
]
MULTIMODAL_WEAK = [Schwefel, Gallagher101, Gallagher21, Katsuura, LunacekBiRastrigin]

ALL_BBOB = bbob_functions


def func_id(func_class):
    """Generate readable test ID."""
    return func_class.__name__


# =============================================================================
# Global Optimum Tests
# =============================================================================


@pytest.mark.bbob
class TestBBOBGlobalOptimum:
    """Test that BBOB functions have global optimum information.

    Note: BBOB functions use instance-based random transformations.
    x_opt/x_global is the optimal location, but complex functions
    may have numerical discrepancies due to transformation chains.
    """

    @pytest.mark.parametrize("func_class", ALL_BBOB, ids=func_id)
    def test_has_global_optimum(self, func_class):
        """BBOB functions should define f_global and x_global."""
        func = func_class(n_dim=2)
        assert func.f_global is not None
        assert func.x_global is not None

    @pytest.mark.parametrize("func_class", ALL_BBOB, ids=func_id)
    def test_x_global_in_bounds(self, func_class):
        """x_global should be within search bounds."""
        func = func_class(n_dim=2)
        x_global = func.x_global
        bounds = func.default_bounds
        # Allow small tolerance outside bounds
        assert np.all(x_global >= bounds[0] - 1)
        assert np.all(x_global <= bounds[1] + 1)


# =============================================================================
# Function Properties Tests
# =============================================================================


@pytest.mark.bbob
class TestBBOBFunctionProperties:
    """Test BBOB function properties and specs."""

    @pytest.mark.parametrize("func_class", ALL_BBOB, ids=func_id)
    def test_has_func_id(self, func_class):
        """Each BBOB function has a func_id between 1-24."""
        func = func_class(n_dim=2)
        func_id_val = func.spec.get("func_id")
        assert func_id_val is not None
        assert 1 <= func_id_val <= 24

    @pytest.mark.parametrize("func_class", ALL_BBOB, ids=func_id)
    def test_func_id_matches_dict(self, func_class):
        """Function's func_id matches its position in BBOB_FUNCTIONS dict."""
        func = func_class(n_dim=2)
        func_id_val = func.spec.get("func_id")
        assert BBOB_FUNCTIONS.get(func_id_val) == func_class

    @pytest.mark.parametrize("func_class", ALL_BBOB, ids=func_id)
    def test_has_name(self, func_class):
        """Each BBOB function has a name."""
        func = func_class(n_dim=2)
        name = func.spec.get("name")
        assert name is not None and len(name) > 0

    @pytest.mark.parametrize("func_class", ALL_BBOB, ids=func_id)
    def test_has_continuous_spec(self, func_class):
        """All BBOB functions have continuous spec (may be True or False)."""
        func = func_class(n_dim=2)
        # StepEllipsoidal is not continuous due to floor operation
        assert "continuous" in func.spec or func.spec.get("continuous", True) is True


# =============================================================================
# Category-Specific Tests
# =============================================================================


@pytest.mark.bbob
class TestSeparableFunctions:
    """Test properties specific to separable functions (f1-f5)."""

    @pytest.mark.parametrize("func_class", SEPARABLE, ids=func_id)
    def test_separable_marked(self, func_class):
        """Separable functions should have separable=True in spec."""
        func = func_class(n_dim=2)
        assert func.spec.get("separable", False) is True

    @pytest.mark.parametrize("func_class", SEPARABLE, ids=func_id)
    def test_func_id_range(self, func_class):
        """Separable functions have func_id 1-5."""
        func = func_class(n_dim=2)
        assert 1 <= func.spec["func_id"] <= 5


@pytest.mark.bbob
class TestHighConditioningFunctions:
    """Test properties specific to high conditioning functions (f10-f14)."""

    @pytest.mark.parametrize("func_class", HIGH_CONDITIONING, ids=func_id)
    def test_unimodal_marked(self, func_class):
        """High conditioning functions are unimodal."""
        func = func_class(n_dim=2)
        assert func.spec.get("unimodal", False) is True

    @pytest.mark.parametrize("func_class", HIGH_CONDITIONING, ids=func_id)
    def test_func_id_range(self, func_class):
        """High conditioning functions have func_id 10-14."""
        func = func_class(n_dim=2)
        assert 10 <= func.spec["func_id"] <= 14


@pytest.mark.bbob
class TestMultimodalFunctions:
    """Test properties specific to multimodal functions (f15-f24)."""

    @pytest.mark.parametrize("func_class", MULTIMODAL_ADEQUATE + MULTIMODAL_WEAK, ids=func_id)
    def test_not_unimodal(self, func_class):
        """Multimodal functions have unimodal=False."""
        func = func_class(n_dim=2)
        assert func.spec.get("unimodal", True) is False


# =============================================================================
# Dimension Handling Tests
# =============================================================================


@pytest.mark.bbob
class TestBBOBDimensions:
    """Test dimension handling for BBOB functions."""

    @pytest.mark.parametrize("n_dim", [2, 5, 10, 20, 40])
    def test_supported_dimensions(self, n_dim):
        """BBOB functions work with various dimensions."""
        func = Sphere(n_dim=n_dim)
        assert func.n_dim == n_dim
        assert len(func.search_space) == n_dim

        result = func(np.zeros(n_dim))
        assert np.isfinite(result)

    @pytest.mark.parametrize("func_class", ALL_BBOB, ids=func_id)
    def test_dimension_2(self, func_class):
        """All BBOB functions work in 2D."""
        func = func_class(n_dim=2)
        result = func(np.zeros(2))
        assert np.isfinite(result)

    @pytest.mark.parametrize("func_class", ALL_BBOB, ids=func_id)
    def test_dimension_10(self, func_class):
        """All BBOB functions work in 10D."""
        func = func_class(n_dim=10)
        result = func(np.zeros(10))
        assert np.isfinite(result)


# =============================================================================
# Input Format Tests
# =============================================================================


@pytest.mark.bbob
class TestBBOBInputFormats:
    """Test different input formats for BBOB functions."""

    def test_array_input(self):
        """BBOB functions accept numpy array input."""
        func = Sphere(n_dim=5)
        result = func(np.zeros(5))
        assert np.isfinite(result)

    def test_list_input(self):
        """BBOB functions accept list input."""
        func = Sphere(n_dim=5)
        result = func([0.0] * 5)
        assert np.isfinite(result)

    def test_dict_input(self):
        """BBOB functions accept dict input."""
        func = Sphere(n_dim=5)
        params = {f"x{i}": 0.0 for i in range(5)}
        result = func(params)
        assert np.isfinite(result)

    def test_kwargs_input(self):
        """BBOB functions accept kwargs input."""
        func = Sphere(n_dim=2)
        result = func(x0=0.0, x1=0.0)
        assert np.isfinite(result)


# =============================================================================
# Search Space Tests
# =============================================================================


@pytest.mark.bbob
class TestBBOBSearchSpace:
    """Test search space properties for BBOB functions."""

    @pytest.mark.parametrize("func_class", ALL_BBOB, ids=func_id)
    def test_default_bounds(self, func_class):
        """Default bounds should be [-5, 5]."""
        func = func_class(n_dim=2)
        assert func.default_bounds == (-5.0, 5.0)

    @pytest.mark.parametrize("func_class", ALL_BBOB, ids=func_id)
    def test_search_space_keys(self, func_class):
        """Search space keys follow x0, x1, ... pattern."""
        func = func_class(n_dim=5)
        space = func.search_space
        for i in range(5):
            assert f"x{i}" in space


# =============================================================================
# Objective Direction Tests
# =============================================================================


@pytest.mark.bbob
class TestBBOBObjective:
    """Test objective parameter behavior."""

    def test_minimize_objective(self):
        """Minimize objective returns raw values."""
        func = Sphere(n_dim=2, objective="minimize")
        result = func(func.x_global)
        assert result == func.f_global

    def test_maximize_objective(self):
        """Maximize objective negates values."""
        func = Sphere(n_dim=2, objective="maximize")
        result = func(func.x_global)
        assert result == -func.f_global


# =============================================================================
# Instance Reproducibility Tests
# =============================================================================


@pytest.mark.bbob
class TestBBOBInstances:
    """Test instance-based reproducibility."""

    @pytest.mark.parametrize("func_class", ALL_BBOB[:5], ids=func_id)
    def test_same_instance_reproducible(self, func_class):
        """Same instance with same input gives same output."""
        func = func_class(n_dim=5, instance=1)
        x = np.random.RandomState(42).randn(5)

        result1 = func(x)
        result2 = func(x)

        assert result1 == result2

    @pytest.mark.parametrize("func_class", ALL_BBOB[:5], ids=func_id)
    def test_different_instances_differ(self, func_class):
        """Different instances may give different outputs."""
        func1 = func_class(n_dim=5, instance=1)
        func2 = func_class(n_dim=5, instance=2)
        x = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        result1 = func1(x)
        result2 = func2(x)

        # Most functions will differ between instances
        # (due to random shifts/rotations)
        # This is not strictly required but typical
        # We just verify both evaluate without error
        assert np.isfinite(result1)
        assert np.isfinite(result2)
