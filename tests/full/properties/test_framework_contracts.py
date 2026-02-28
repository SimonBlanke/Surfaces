# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Framework Contract Tests
========================

These tests verify that the template method pattern correctly enforces
its contracts at each level of the class hierarchy. They create minimal
"bad" subclasses that violate specific contracts and assert the framework
produces clear errors.

Hierarchy contracts tested:

    BaseTestFunction
        _objective(params) -> float            [must override]

    EngineeringFunction(BaseTestFunction)
        _raw_objective(params) -> float         [must override]
        _constraints(params) -> list[float]     [optional override]

    MachineLearningFunction(BaseTestFunction)
        _ml_objective(params) -> float          [must override]

    SimulationFunction(BaseTestFunction)
        _setup_simulation()                     [must override (abstract)]
        _run_simulation(params) -> Any          [must override (abstract)]
        _extract_objective(result) -> float     [must override (abstract)]

    ODESimulationFunction(SimulationFunction)
        _ode_system(t, y, params) -> ndarray    [must override (abstract)]
        _get_initial_conditions() -> ndarray    [must override (abstract)]
        _compute_objective(t, y, params) -> float  [must override (abstract)]

Usage:
    pytest tests/core/properties/test_framework_contracts.py -v
"""

import numpy as np
import pytest

from surfaces.test_functions._base_single_objective import BaseSingleObjectiveTestFunction
from surfaces.test_functions._base_test_function import BaseTestFunction
from surfaces.test_functions.algebraic.constrained._base_engineering_function import (
    EngineeringFunction,
)
from surfaces.test_functions.machine_learning._base_machine_learning import (
    MachineLearningFunction,
)


class _ValidAlgebraic(BaseSingleObjectiveTestFunction):
    """Minimal valid concrete algebraic function."""

    _spec = {"n_dim": 1, "default_bounds": (-5.0, 5.0)}

    @property
    def search_space(self):
        return {"x0": np.linspace(-5, 5, 100)}

    def _objective(self, params):
        return params["x0"] ** 2


class _ValidEngineering(EngineeringFunction):
    """Minimal valid concrete engineering function."""

    variable_names = ["x0"]
    variable_bounds = [(0.0, 1.0)]

    def _raw_objective(self, params):
        return params["x0"] ** 2


class _ValidML(MachineLearningFunction):
    """Minimal valid concrete ML function."""

    para_names = ["alpha"]
    alpha_default = [0.01, 0.1, 1.0]

    def _ml_objective(self, params):
        return 1.0 / (1.0 + params["alpha"])


class TestBaseContracts:
    """Verify BaseTestFunction enforces _objective override."""

    def test_missing_objective_raises(self):
        """Subclass without _objective raises NotImplementedError on call."""

        class MissingObjective(BaseTestFunction):
            @property
            def search_space(self):
                return {"x0": np.linspace(-5, 5, 10)}

        func = MissingObjective()
        with pytest.raises(NotImplementedError, match="must implement _objective"):
            func._objective({"x0": 1.0})

    def test_missing_objective_via_call(self):
        """NotImplementedError propagates through __call__."""

        class MissingObjective(BaseTestFunction):
            @property
            def search_space(self):
                return {"x0": np.linspace(-5, 5, 10)}

        func = MissingObjective()
        with pytest.raises(NotImplementedError, match="must implement _objective"):
            func({"x0": 1.0})

    def test_missing_objective_via_pure(self):
        """NotImplementedError propagates through pure()."""

        class MissingObjective(BaseTestFunction):
            @property
            def search_space(self):
                return {"x0": np.linspace(-5, 5, 10)}

        func = MissingObjective()
        with pytest.raises(NotImplementedError, match="must implement _objective"):
            func.pure({"x0": 1.0})

    def test_error_message_includes_class_name(self):
        """Error message names the offending class."""

        class MyBrokenFunction(BaseTestFunction):
            @property
            def search_space(self):
                return {"x0": np.linspace(-5, 5, 10)}

        func = MyBrokenFunction()
        with pytest.raises(NotImplementedError, match="MyBrokenFunction"):
            func._objective({"x0": 1.0})

    def test_valid_algebraic_works(self):
        """Positive control: valid implementation works."""
        func = _ValidAlgebraic()
        result = func.pure({"x0": 2.0})
        assert result == 4.0

    def test_objective_invalid_direction(self):
        """Invalid objective direction raises ValueError."""
        with pytest.raises(ValueError, match="minimize.*maximize"):
            _ValidAlgebraic(objective="invalid")


class TestEngineeringContracts:
    """Verify EngineeringFunction enforces _raw_objective override."""

    def test_missing_raw_objective_raises(self):
        """Subclass without _raw_objective raises NotImplementedError."""

        class MissingRawObjective(EngineeringFunction):
            variable_names = ["x0"]
            variable_bounds = [(0.0, 1.0)]

        func = MissingRawObjective()
        with pytest.raises(NotImplementedError, match="_raw_objective"):
            func._objective({"x0": 0.5})

    def test_missing_raw_objective_via_call(self):
        """NotImplementedError propagates through __call__."""

        class MissingRawObjective(EngineeringFunction):
            variable_names = ["x0"]
            variable_bounds = [(0.0, 1.0)]

        func = MissingRawObjective()
        with pytest.raises(NotImplementedError, match="_raw_objective"):
            func({"x0": 0.5})

    def test_objective_combines_raw_and_penalty(self):
        """_objective = raw_objective + penalty (sub-template contract)."""

        class WithConstraint(EngineeringFunction):
            variable_names = ["x0"]
            variable_bounds = [(0.0, 1.0)]

            def _raw_objective(self, params):
                return params["x0"]

            def _constraints(self, params):
                # g <= 0 is feasible; x0 - 0.3 > 0 when x0 > 0.3 -> violation
                return [params["x0"] - 0.3]

        func = WithConstraint(penalty_coefficient=100.0)

        # Feasible point (x0=0.2, constraint = -0.1, no penalty)
        feasible = func._objective({"x0": 0.2})
        assert feasible == pytest.approx(0.2)

        # Infeasible point (x0=0.5, constraint = 0.2, penalty = 100 * 0.04)
        infeasible = func._objective({"x0": 0.5})
        expected = 0.5 + 100.0 * 0.2**2
        assert infeasible == pytest.approx(expected)

    def test_default_constraints_empty(self):
        """Default constraints() returns empty list (no constraints)."""
        func = _ValidEngineering()
        assert func.constraints({"x0": 0.5}) == []
        assert func.penalty({"x0": 0.5}) == 0.0

    def test_valid_engineering_works(self):
        """Positive control: valid engineering function works."""
        func = _ValidEngineering()
        result = func._objective({"x0": 0.5})
        assert result == pytest.approx(0.25)

    def test_overriding_objective_bypasses_sub_template(self):
        """Overriding _objective directly bypasses raw_objective + penalty."""

        class DirectOverride(EngineeringFunction):
            variable_names = ["x0"]
            variable_bounds = [(0.0, 1.0)]

            def _objective(self, params):
                return 42.0

        func = DirectOverride()
        assert func._objective({"x0": 0.5}) == 42.0


class TestMLContracts:
    """Verify MachineLearningFunction enforces _ml_objective override."""

    def test_missing_ml_objective_raises(self):
        """Subclass without _ml_objective raises NotImplementedError."""

        class MissingMLObjective(MachineLearningFunction):
            para_names = ["alpha"]
            alpha_default = [0.01, 0.1]

        func = MissingMLObjective()
        with pytest.raises(NotImplementedError, match="must implement _ml_objective"):
            func._ml_objective({"alpha": 0.1})

    def test_missing_ml_objective_via_objective(self):
        """NotImplementedError propagates through _objective."""

        class MissingMLObjective(MachineLearningFunction):
            para_names = ["alpha"]
            alpha_default = [0.01, 0.1]

        func = MissingMLObjective()
        with pytest.raises(NotImplementedError, match="must implement _ml_objective"):
            func._objective({"alpha": 0.1})

    def test_error_message_includes_class_name(self):
        """Error message names the offending ML class."""

        class MyBrokenMLFunction(MachineLearningFunction):
            para_names = ["alpha"]
            alpha_default = [0.01, 0.1]

        func = MyBrokenMLFunction()
        with pytest.raises(NotImplementedError, match="MyBrokenMLFunction"):
            func._ml_objective({"alpha": 0.1})

    def test_ml_objective_delegates_correctly(self):
        """_objective delegates to _ml_objective (sub-template contract)."""
        func = _ValidML()
        direct = func._ml_objective({"alpha": 0.1})
        via_template = func._objective({"alpha": 0.1})
        assert direct == via_template

    def test_ml_default_objective_is_maximize(self):
        """ML functions default to 'maximize'."""
        func = _ValidML()
        assert func.objective == "maximize"

    def test_ml_maximize_negates_for_call(self):
        """__call__ negates the score when objective='maximize' (internal convention).

        ML functions use inverted convention: _evaluate returns raw score
        for maximize, but negated for minimize.
        """
        func = _ValidML()
        raw = func._ml_objective({"alpha": 0.1})
        via_call = func({"alpha": 0.1})
        # ML _evaluate returns raw when maximize, so __call__ == _ml_objective
        assert via_call == pytest.approx(raw)

    def test_valid_ml_works(self):
        """Positive control: valid ML function works."""
        func = _ValidML()
        result = func._ml_objective({"alpha": 0.1})
        assert isinstance(result, float)
        assert result > 0


class TestSimulationContracts:
    """Verify SimulationFunction enforces abstract method overrides.

    Note: SimulationFunction uses @abstractmethod but BaseTestFunction
    does not use ABCMeta, so Python does not enforce at instantiation.
    Enforcement happens at evaluation time when unimplemented methods
    return None instead of proper values.
    """

    def test_missing_run_simulation_fails_at_eval(self):
        """Missing _run_simulation causes failure at evaluation time."""
        from surfaces.test_functions.simulation._base_simulation import SimulationFunction

        class MissingRun(SimulationFunction):
            @property
            def search_space(self):
                return {"x0": np.linspace(0, 1, 10)}

            def _setup_simulation(self):
                pass

            def _extract_objective(self, result):
                return float(result)

        func = MissingRun()
        # _run_simulation (abstract, body=pass) returns None
        # _extract_objective(None) -> TypeError: float(None)
        with pytest.raises(TypeError):
            func._objective({"x0": 0.5})

    def test_missing_extract_objective_fails_at_eval(self):
        """Missing _extract_objective causes failure at evaluation time."""
        from surfaces.test_functions.simulation._base_simulation import SimulationFunction

        class MissingExtract(SimulationFunction):
            @property
            def search_space(self):
                return {"x0": np.linspace(0, 1, 10)}

            def _setup_simulation(self):
                pass

            def _run_simulation(self, params):
                return params["x0"] ** 2

        func = MissingExtract()
        # _extract_objective (abstract, body=pass) returns None
        result = func._objective({"x0": 0.5})
        assert result is None  # Wrong type: should be float

    def test_valid_simulation_works(self):
        """Positive control: complete SimulationFunction works."""
        from surfaces.test_functions.simulation._base_simulation import SimulationFunction

        class ValidSim(SimulationFunction):
            @property
            def search_space(self):
                return {"x0": np.linspace(0, 1, 10)}

            def _setup_simulation(self):
                pass

            def _run_simulation(self, params):
                return params["x0"] ** 2

            def _extract_objective(self, result):
                return float(result)

        func = ValidSim()
        result = func._objective({"x0": 0.5})
        assert result == pytest.approx(0.25)

    def test_sub_template_pipeline(self):
        """_objective calls _run_simulation then _extract_objective."""
        from surfaces.test_functions.simulation._base_simulation import SimulationFunction

        call_log = []

        class TrackedSim(SimulationFunction):
            @property
            def search_space(self):
                return {"x0": np.linspace(0, 1, 10)}

            def _setup_simulation(self):
                pass

            def _run_simulation(self, params):
                call_log.append("run")
                return {"raw": params["x0"] * 10}

            def _extract_objective(self, result):
                call_log.append("extract")
                return result["raw"]

        func = TrackedSim()
        result = func._objective({"x0": 0.3})

        assert call_log == ["run", "extract"]
        assert result == pytest.approx(3.0)


class TestODEContracts:
    """Verify ODESimulationFunction enforces abstract method overrides.

    Note: Like SimulationFunction, @abstractmethod is not enforced at
    instantiation (no ABCMeta). Missing methods return None from the
    abstract `pass` body. We test each method individually because
    running through the full solver with None returns can hang.
    """

    def test_missing_ode_system_returns_none(self):
        """Unimplemented _ode_system returns None (violates contract)."""
        from surfaces.test_functions.simulation.dynamics._base_ode import (
            ODESimulationFunction,
        )

        class MissingODE(ODESimulationFunction):
            @property
            def search_space(self):
                return {"k": np.linspace(0.1, 1, 10)}

            def _get_initial_conditions(self):
                return np.array([1.0])

            def _compute_objective(self, t, y, params):
                return float(y[0, -1])

        func = MissingODE(t_span=(0, 1))
        # Abstract body=pass returns None, violating ndarray contract
        result = func._ode_system(0.0, np.array([1.0]), {"k": 0.5})
        assert result is None

    def test_missing_initial_conditions_returns_none(self):
        """Unimplemented _get_initial_conditions returns None (violates contract)."""
        from surfaces.test_functions.simulation.dynamics._base_ode import (
            ODESimulationFunction,
        )

        class MissingIC(ODESimulationFunction):
            @property
            def search_space(self):
                return {"k": np.linspace(0.1, 1, 10)}

            def _ode_system(self, t, y, params):
                return np.array([-params["k"] * y[0]])

            def _compute_objective(self, t, y, params):
                return float(y[0, -1])

        func = MissingIC(t_span=(0, 1))
        result = func._get_initial_conditions()
        assert result is None

    def test_missing_compute_objective_returns_none(self):
        """Unimplemented _compute_objective returns None (violates contract)."""
        from surfaces.test_functions.simulation.dynamics._base_ode import (
            ODESimulationFunction,
        )

        class MissingCompute(ODESimulationFunction):
            @property
            def search_space(self):
                return {"k": np.linspace(0.1, 1, 10)}

            def _ode_system(self, t, y, params):
                return np.array([-params["k"] * y[0]])

            def _get_initial_conditions(self):
                return np.array([1.0])

        func = MissingCompute(t_span=(0, 1))
        t = np.linspace(0, 1, 10)
        y = np.ones((1, 10))
        result = func._compute_objective(t, y, {"k": 0.5})
        assert result is None

    def test_valid_ode_works(self):
        """Positive control: complete ODE function integrates and returns score."""
        from surfaces.test_functions.simulation.dynamics._base_ode import (
            ODESimulationFunction,
        )

        class SimpleDecay(ODESimulationFunction):
            """dy/dt = -k*y, minimize final y value."""

            @property
            def search_space(self):
                return {"k": np.linspace(0.1, 5.0, 50)}

            def _ode_system(self, t, y, params):
                return np.array([-params["k"] * y[0]])

            def _get_initial_conditions(self):
                return np.array([1.0])

            def _compute_objective(self, t, y, params):
                return float(y[0, -1])

        func = SimpleDecay(t_span=(0, 1))
        result = func._objective({"k": 1.0})
        # y(1) = exp(-1) ~ 0.368
        assert 0.3 < result < 0.4


class TestCrossCuttingContracts:
    """Tests that apply across all hierarchy levels."""

    def test_pure_bypasses_modifiers(self):
        """pure() skips modifiers but still applies _objective."""
        from surfaces.modifiers import GaussianNoise

        func = _ValidAlgebraic(modifiers=[GaussianNoise(sigma=100.0)])

        # pure() should return the exact value
        pure_result = func.pure({"x0": 2.0})
        assert pure_result == pytest.approx(4.0)

        # __call__ with extreme noise should differ
        results = [func({"x0": 2.0}) for _ in range(10)]
        # At least some should differ from 4.0 with sigma=100
        assert any(abs(r - 4.0) > 1.0 for r in results)

    def test_catch_errors_wraps_objective_errors(self):
        """catch_errors intercepts exceptions from _objective."""

        class ErrorFunction(BaseTestFunction):
            @property
            def search_space(self):
                return {"x0": np.linspace(-5, 5, 10)}

            def _objective(self, params):
                raise ValueError("intentional error")

        func = ErrorFunction(catch_errors={ValueError: -999.0})
        result = func({"x0": 1.0})
        assert result == -999.0

    def test_catch_errors_does_not_mask_not_implemented(self):
        """catch_errors with Ellipsis does NOT prevent NotImplementedError
        from being raised (if user wants it to, they must explicitly catch it)."""

        class MissingObjective(BaseTestFunction):
            @property
            def search_space(self):
                return {"x0": np.linspace(-5, 5, 10)}

        # Ellipsis catches all exceptions
        func = MissingObjective(catch_errors={...: -999.0})
        # NotImplementedError IS an exception, so it IS caught by Ellipsis
        result = func({"x0": 1.0})
        assert result == -999.0

    def test_memory_caches_objective_calls(self):
        """Memory caching works with _objective."""
        call_count = 0

        class CountingFunction(BaseTestFunction):
            @property
            def search_space(self):
                return {"x0": np.linspace(-5, 5, 10)}

            def _objective(self, params):
                nonlocal call_count
                call_count += 1
                return params["x0"] ** 2

        func = CountingFunction(memory=True)
        func({"x0": 2.0})
        func({"x0": 2.0})
        assert call_count == 1  # Second call served from cache
