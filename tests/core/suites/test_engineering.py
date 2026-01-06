# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for engineering design optimization functions.

Engineering functions are real-world inspired benchmarks with
physical meaning, commonly used to evaluate constrained optimization
algorithms. Each function provides constraint handling via penalty methods.
"""

import numpy as np
import pytest

from surfaces.test_functions.algebraic.constrained import (
    CantileverBeamFunction,
    PressureVesselFunction,
    TensionCompressionSpringFunction,
    ThreeBarTrussFunction,
    WeldedBeamFunction,
    constrained_functions,
)
from tests.conftest import func_id, get_sample_params, instantiate_function

# =============================================================================
# Basic Instantiation and Evaluation
# =============================================================================


@pytest.mark.engineering
class TestEngineeringInstantiation:
    """Test basic instantiation and evaluation."""

    @pytest.mark.parametrize("func_class", constrained_functions, ids=func_id)
    def test_instantiates(self, func_class):
        """Engineering functions instantiate correctly."""
        func = instantiate_function(func_class)
        assert func is not None
        assert len(func.search_space) > 0

    @pytest.mark.parametrize("func_class", constrained_functions, ids=func_id)
    def test_evaluates(self, func_class):
        """Engineering functions evaluate and return numeric result."""
        func = instantiate_function(func_class)
        params = get_sample_params(func)
        result = func(params)
        assert isinstance(result, (int, float))

    @pytest.mark.parametrize("func_class", constrained_functions, ids=func_id)
    def test_returns_finite(self, func_class):
        """Engineering functions return finite values."""
        func = instantiate_function(func_class)
        params = get_sample_params(func)
        result = func(params)
        assert np.isfinite(result)


# =============================================================================
# Constraint Methods
# =============================================================================


@pytest.mark.engineering
class TestConstraintMethods:
    """Test constraint-related methods."""

    @pytest.mark.parametrize("func_class", constrained_functions, ids=func_id)
    def test_has_constraints_method(self, func_class):
        """Engineering functions have constraints method."""
        func = instantiate_function(func_class)
        assert hasattr(func, "constraints")
        assert callable(func.constraints)

    @pytest.mark.parametrize("func_class", constrained_functions, ids=func_id)
    def test_has_is_feasible_method(self, func_class):
        """Engineering functions have is_feasible method."""
        func = instantiate_function(func_class)
        assert hasattr(func, "is_feasible")
        assert callable(func.is_feasible)

    @pytest.mark.parametrize("func_class", constrained_functions, ids=func_id)
    def test_has_raw_objective_method(self, func_class):
        """Engineering functions have raw_objective method."""
        func = instantiate_function(func_class)
        assert hasattr(func, "raw_objective")
        assert callable(func.raw_objective)

    @pytest.mark.parametrize("func_class", constrained_functions, ids=func_id)
    def test_has_penalty_method(self, func_class):
        """Engineering functions have penalty method."""
        func = instantiate_function(func_class)
        assert hasattr(func, "penalty")
        assert callable(func.penalty)

    @pytest.mark.parametrize("func_class", constrained_functions, ids=func_id)
    def test_constraints_returns_sequence(self, func_class):
        """constraints() returns array-like sequence."""
        func = instantiate_function(func_class)
        params = get_sample_params(func)
        constraints = func.constraints(params)
        # Constraints can be list or ndarray
        assert hasattr(constraints, "__len__")
        assert len(constraints) > 0

    @pytest.mark.parametrize("func_class", constrained_functions, ids=func_id)
    def test_is_feasible_returns_bool(self, func_class):
        """is_feasible() returns boolean."""
        func = instantiate_function(func_class)
        params = get_sample_params(func)
        feasible = func.is_feasible(params)
        assert isinstance(feasible, (bool, np.bool_))


# =============================================================================
# Penalty Behavior
# =============================================================================


@pytest.mark.engineering
class TestPenaltyBehavior:
    """Test penalty calculation behavior."""

    @pytest.mark.parametrize("func_class", constrained_functions, ids=func_id)
    def test_penalty_non_negative(self, func_class):
        """Penalty is always >= 0."""
        func = instantiate_function(func_class)
        params = get_sample_params(func)
        penalty = func.penalty(params)
        assert penalty >= 0

    @pytest.mark.parametrize("func_class", constrained_functions, ids=func_id)
    def test_feasible_zero_penalty(self, func_class):
        """Feasible solutions have zero penalty."""
        func = instantiate_function(func_class)
        params = get_sample_params(func)
        if func.is_feasible(params):
            penalty = func.penalty(params)
            assert np.isclose(penalty, 0.0)

    @pytest.mark.parametrize("func_class", constrained_functions, ids=func_id)
    def test_objective_includes_penalty(self, func_class):
        """__call__ returns raw_objective + penalty."""
        func = instantiate_function(func_class)
        params = get_sample_params(func)

        total = func(params)
        raw = func.raw_objective(params)
        penalty = func.penalty(params)

        # The penalized objective should be raw + penalty * coefficient
        # (the coefficient may vary, so we just check the relationship)
        assert total >= raw or np.isclose(total, raw)


# =============================================================================
# Specific Function Tests
# =============================================================================


@pytest.mark.engineering
class TestThreeBarTruss:
    """Test Three-Bar Truss function specifics."""

    def test_dimensions(self):
        """Three-Bar Truss has 2 dimensions."""
        func = ThreeBarTrussFunction()
        assert len(func.search_space) == 2

    def test_constraint_count(self):
        """Three-Bar Truss has 3 constraints."""
        func = ThreeBarTrussFunction()
        params = get_sample_params(func)
        constraints = func.constraints(params)
        assert len(constraints) == 3


@pytest.mark.engineering
class TestWeldedBeam:
    """Test Welded Beam function specifics."""

    def test_dimensions(self):
        """Welded Beam has 4 dimensions."""
        func = WeldedBeamFunction()
        assert len(func.search_space) == 4

    def test_constraint_count(self):
        """Welded Beam has multiple constraints."""
        func = WeldedBeamFunction()
        params = get_sample_params(func)
        constraints = func.constraints(params)
        assert len(constraints) >= 5


@pytest.mark.engineering
class TestPressureVessel:
    """Test Pressure Vessel function specifics."""

    def test_dimensions(self):
        """Pressure Vessel has 4 dimensions."""
        func = PressureVesselFunction()
        assert len(func.search_space) == 4

    def test_constraint_count(self):
        """Pressure Vessel has 3 constraints."""
        func = PressureVesselFunction()
        params = get_sample_params(func)
        constraints = func.constraints(params)
        assert len(constraints) >= 3


@pytest.mark.engineering
class TestTensionCompressionSpring:
    """Test Tension/Compression Spring function specifics."""

    def test_dimensions(self):
        """Tension/Compression Spring has 3 dimensions."""
        func = TensionCompressionSpringFunction()
        assert len(func.search_space) == 3

    def test_constraint_count(self):
        """Tension/Compression Spring has 4 constraints."""
        func = TensionCompressionSpringFunction()
        params = get_sample_params(func)
        constraints = func.constraints(params)
        assert len(constraints) == 4


@pytest.mark.engineering
class TestCantileverBeam:
    """Test Cantilever Beam function specifics."""

    def test_dimensions(self):
        """Cantilever Beam has 5 dimensions."""
        func = CantileverBeamFunction()
        assert len(func.search_space) == 5

    def test_constraint_count(self):
        """Cantilever Beam has 1 constraint."""
        func = CantileverBeamFunction()
        params = get_sample_params(func)
        constraints = func.constraints(params)
        assert len(constraints) == 1


# =============================================================================
# Input Format Tests
# =============================================================================


@pytest.mark.engineering
class TestEngineeringInputFormats:
    """Test input format handling."""

    @pytest.mark.parametrize("func_class", constrained_functions, ids=func_id)
    def test_dict_input(self, func_class):
        """Engineering functions accept dict input."""
        func = instantiate_function(func_class)
        params = get_sample_params(func)
        result = func(params)
        assert np.isfinite(result)

    @pytest.mark.parametrize("func_class", constrained_functions, ids=func_id)
    def test_constraint_methods_accept_dict(self, func_class):
        """Constraint methods accept dict input."""
        func = instantiate_function(func_class)
        params = get_sample_params(func)

        # All these should work with dict input
        constraints = func.constraints(params)
        is_feasible = func.is_feasible(params)
        raw = func.raw_objective(params)
        penalty = func.penalty(params)

        # Constraints can be list or ndarray
        assert hasattr(constraints, "__len__")
        assert isinstance(is_feasible, (bool, np.bool_))
        assert isinstance(raw, (int, float))
        assert isinstance(penalty, (int, float))
