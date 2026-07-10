"""Tests for the Compute Units (CU) cost module and eval_cost spec property."""

import dataclasses

import pytest

from surfaces import _cost as cost
from surfaces.test_functions.algebraic import algebraic_functions
from surfaces.test_functions.bbob import bbob_functions
from tests.conftest import func_id, instantiate_function


class TestCostModule:
    """Tests for surfaces._cost internal API."""

    def setup_method(self):
        cost.reset()

    def test_calibrate_returns_positive_float(self):
        ref_time = cost.calibrate()
        assert isinstance(ref_time, float)
        assert ref_time > 0

    def test_calibrate_is_cached(self):
        first = cost.calibrate()
        second = cost.calibrate()
        assert first == second

    def test_to_cu_one_ref_op_equals_one(self):
        ref_time = cost.calibrate()
        cu = cost.to_cu(ref_time)
        assert abs(cu - 1.0) < 1e-10

    def test_to_cu_zero(self):
        assert cost.to_cu(0.0) == 0.0

    def test_to_cu_scales_linearly(self):
        ref_time = cost.calibrate()
        assert abs(cost.to_cu(ref_time * 10) - 10.0) < 1e-8
        assert abs(cost.to_cu(ref_time * 100) - 100.0) < 1e-6

    def test_reset_clears_cache(self):
        cost.calibrate()
        cost.reset()
        assert cost._ref_time is None

    def test_calibrate_min_duration(self):
        ref_short = cost.calibrate(min_duration=0.1)
        assert ref_short > 0
        cost.reset()
        ref_long = cost.calibrate(min_duration=0.5)
        assert ref_long > 0
        # Both should give similar results (within 50%)
        ratio = ref_short / ref_long
        assert 0.5 < ratio < 2.0


class TestEvalCostSpec:
    """Tests for the eval_cost field on the resolved FunctionSpec."""

    @pytest.mark.parametrize("func_class", algebraic_functions, ids=func_id)
    def test_eval_cost_accessible(self, func_class):
        """eval_cost is accessible via spec."""
        func = instantiate_function(func_class)
        val = func.spec.eval_cost
        assert val is None or isinstance(val, (int, float))

    @pytest.mark.parametrize("func_class", algebraic_functions, ids=func_id)
    def test_eval_cost_in_spec_dict(self, func_class):
        """eval_cost appears in the merged spec dict."""
        func = instantiate_function(func_class)
        assert "eval_cost" in dataclasses.asdict(func.spec)

    @pytest.mark.parametrize("func_class", bbob_functions, ids=func_id)
    def test_bbob_eval_cost_accessible(self, func_class):
        """BBOB functions have eval_cost in spec."""
        func = instantiate_function(func_class, n_dim=2)
        val = func.spec.eval_cost
        assert val is None or isinstance(val, (int, float))
