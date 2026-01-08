# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for CEC 2006 constrained benchmark functions."""

import numpy as np
import pytest

from surfaces.test_functions.benchmark.cec.cec2006 import (
    CEC2006Function,
    G01, G02, G03, G04, G05, G06, G07, G08, G09, G10,
    G11, G12, G13, G14, G15, G16, G17, G18, G19, G20,
    G21, G22, G23, G24,
    CEC2006_ALL,
)

# Functions with known optimal solutions
CEC2006_WITH_KNOWN_OPTIMUM = [
    G01, G03, G04, G05, G06, G07, G08, G09, G10,
    G11, G12, G13, G14, G15, G17, G18, G21, G23, G24
]

# Functions with equality constraints (need higher tolerance)
CEC2006_WITH_EQUALITY = [
    G03, G05, G11, G13, G14, G15, G17, G20, G21, G22, G23
]


class TestCEC2006Instantiation:
    """Test that all functions can be instantiated."""

    @pytest.mark.parametrize("func_class", CEC2006_ALL)
    def test_instantiation(self, func_class):
        """All functions should instantiate without error."""
        func = func_class()
        assert func is not None
        assert func.n_dim > 0

    @pytest.mark.parametrize("func_class", CEC2006_ALL)
    def test_func_id(self, func_class):
        """Each function should have a valid func_id."""
        func = func_class()
        assert func.func_id is not None
        assert 1 <= func.func_id <= 24


class TestCEC2006Properties:
    """Test function properties and metadata."""

    @pytest.mark.parametrize("func_class", CEC2006_ALL)
    def test_has_bounds(self, func_class):
        """Each function should have variable bounds defined."""
        func = func_class()
        assert len(func.variable_bounds) == func.n_dim
        for lb, ub in func.variable_bounds:
            assert lb < ub

    @pytest.mark.parametrize("func_class", CEC2006_ALL)
    def test_has_constraints(self, func_class):
        """Each function should have constraint count defined."""
        func = func_class()
        n_total = func.n_linear_ineq + func.n_nonlinear_eq + func.n_nonlinear_ineq
        assert n_total == func.n_constraints
        assert n_total > 0  # All CEC2006 functions are constrained

    @pytest.mark.parametrize("func_class", CEC2006_ALL)
    def test_spec_constrained(self, func_class):
        """All functions should be marked as constrained."""
        func = func_class()
        assert func.spec.get("constrained") is True

    @pytest.mark.parametrize("func_class", CEC2006_ALL)
    def test_spec_not_scalable(self, func_class):
        """All functions should be marked as not scalable."""
        func = func_class()
        assert func.spec.get("scalable") is False


class TestCEC2006Evaluation:
    """Test function evaluation."""

    @pytest.mark.parametrize("func_class", CEC2006_ALL)
    def test_evaluation_returns_finite(self, func_class):
        """Function evaluation should return finite values."""
        func = func_class()
        # Test with random point within bounds
        np.random.seed(42)
        x = np.array([
            np.random.uniform(lb, ub)
            for lb, ub in func.variable_bounds
        ])
        result = func.raw_objective(x)
        assert np.isfinite(result)

    @pytest.mark.parametrize("func_class", CEC2006_ALL)
    def test_dict_input(self, func_class):
        """Function should accept dict input."""
        func = func_class()
        np.random.seed(42)
        params = {}
        for i, (lb, ub) in enumerate(func.variable_bounds):
            params[f"x{i}"] = np.random.uniform(lb, ub)
        result = func(params)
        assert np.isfinite(result)

    @pytest.mark.parametrize("func_class", CEC2006_WITH_KNOWN_OPTIMUM)
    def test_optimal_value(self, func_class):
        """Test that f(x*) is close to f_global."""
        func = func_class()
        if func.x_global is None:
            pytest.skip("No known optimal solution")

        f_at_opt = func.raw_objective(func.x_global)
        # Allow some tolerance due to numerical precision
        assert np.isclose(f_at_opt, func.f_global, rtol=1e-3, atol=1e-3), \
            f"{func_class.__name__}: f(x*)={f_at_opt}, expected {func.f_global}"


class TestCEC2006Constraints:
    """Test constraint handling."""

    @pytest.mark.parametrize("func_class", CEC2006_ALL)
    def test_constraints_return_correct_count(self, func_class):
        """Constraint methods should return correct number of constraints."""
        func = func_class()
        np.random.seed(42)
        x = np.array([
            np.random.uniform(lb, ub)
            for lb, ub in func.variable_bounds
        ])
        ineq = func.inequality_constraints(x)
        eq = func.equality_constraints(x)
        assert len(ineq) == func.n_linear_ineq + func.n_nonlinear_ineq
        assert len(eq) == func.n_nonlinear_eq

    @pytest.mark.parametrize("func_class", CEC2006_WITH_KNOWN_OPTIMUM)
    def test_optimal_feasibility(self, func_class):
        """Known optimal solutions should be feasible (within tolerance)."""
        func = func_class(equality_tolerance=1e-3)
        if func.x_global is None:
            pytest.skip("No known optimal solution")

        ineq_viol, eq_viol = func.constraint_violations(func.x_global)
        # Allow small violations due to numerical precision
        max_ineq_viol = max(ineq_viol) if ineq_viol else 0
        max_eq_viol = max(eq_viol) if eq_viol else 0

        assert max_ineq_viol < 1e-3, \
            f"{func_class.__name__}: max inequality violation = {max_ineq_viol}"
        assert max_eq_viol < 1e-3, \
            f"{func_class.__name__}: max equality violation = {max_eq_viol}"

    @pytest.mark.parametrize("func_class", CEC2006_ALL)
    def test_penalty_nonnegative(self, func_class):
        """Penalty should always be non-negative."""
        func = func_class()
        np.random.seed(42)
        for _ in range(5):
            x = np.array([
                np.random.uniform(lb, ub)
                for lb, ub in func.variable_bounds
            ])
            penalty = func.penalty(x)
            assert penalty >= 0


class TestCEC2006SearchSpace:
    """Test search space generation."""

    @pytest.mark.parametrize("func_class", CEC2006_ALL)
    def test_search_space_keys(self, func_class):
        """Search space should have correct keys."""
        func = func_class()
        search_space = func.search_space
        expected_keys = [f"x{i}" for i in range(func.n_dim)]
        assert list(search_space.keys()) == expected_keys

    @pytest.mark.parametrize("func_class", CEC2006_ALL)
    def test_search_space_bounds(self, func_class):
        """Search space values should be within bounds."""
        func = func_class()
        search_space = func.search_space
        for i, (lb, ub) in enumerate(func.variable_bounds):
            values = search_space[f"x{i}"]
            assert values.min() >= lb - 1e-10
            assert values.max() <= ub + 1e-10


class TestCEC2006BatchEvaluation:
    """Test batch evaluation functionality."""

    @pytest.mark.parametrize("func_class", [G01, G03, G06, G11, G24])
    def test_batch_matches_sequential(self, func_class):
        """Batch evaluation should match sequential evaluation."""
        func = func_class()
        np.random.seed(42)

        # Generate random test points within bounds
        n_points = 5
        X = np.array([
            [np.random.uniform(lb, ub) for lb, ub in func.variable_bounds]
            for _ in range(n_points)
        ])

        # Sequential evaluation
        sequential = np.array([func.raw_objective(x) for x in X])

        # Batch evaluation
        batch = func._batch_raw_objective(X)

        assert np.allclose(sequential, batch, rtol=1e-10)

    @pytest.mark.parametrize("func_class", [G01, G03, G06, G11, G24])
    def test_batch_penalty_matches_sequential(self, func_class):
        """Batch penalty should match sequential penalty."""
        func = func_class()
        np.random.seed(42)

        # Generate random test points within bounds
        n_points = 5
        X = np.array([
            [np.random.uniform(lb, ub) for lb, ub in func.variable_bounds]
            for _ in range(n_points)
        ])

        # Sequential evaluation
        sequential = np.array([func.penalty(x) for x in X])

        # Batch evaluation
        batch = func._batch_penalty(X)

        assert np.allclose(sequential, batch, rtol=1e-10)


class TestCEC2006SpecificFunctions:
    """Test specific function implementations."""

    def test_g01_dimensions(self):
        """G01 should have 13 dimensions and 9 linear inequality constraints."""
        func = G01()
        assert func.n_dim == 13
        assert func.n_linear_ineq == 9
        assert func.n_nonlinear_eq == 0
        assert func.n_nonlinear_ineq == 0

    def test_g03_dimensions(self):
        """G03 should have 10 dimensions and 1 equality constraint."""
        func = G03()
        assert func.n_dim == 10
        assert func.n_nonlinear_eq == 1

    def test_g12_disjoint_regions(self):
        """G12 has 729 disjoint feasible regions (spheres)."""
        func = G12()
        # Test a point at center of one sphere (p=5, q=5, r=5)
        x_feasible = np.array([5.0, 5.0, 5.0])
        assert func.is_feasible(x_feasible)

        # Test a point outside all spheres
        x_infeasible = np.array([1.5, 1.5, 1.5])  # Between spheres
        assert not func.is_feasible(x_infeasible)

    def test_g20_unknown_feasible(self):
        """G20 has unknown optimal (may be infeasible)."""
        func = G20()
        assert func.x_global is None
        assert func.n_dim == 24
        assert func.n_nonlinear_eq == 20
        assert func.n_nonlinear_ineq == 6


class TestCEC2006Objective:
    """Test objective parameter."""

    def test_minimize_objective(self):
        """Minimize objective should return penalized values."""
        func = G01(objective="minimize")
        result = func(func.x_global)
        # At optimal, penalty should be near zero
        assert np.isclose(result, func.f_global, rtol=1e-3)

    def test_maximize_objective(self):
        """Maximize objective should negate values."""
        func = G01(objective="maximize")
        result = func(func.x_global)
        # At optimal, penalty should be near zero, but value negated
        assert np.isclose(result, -func.f_global, rtol=1e-3)
