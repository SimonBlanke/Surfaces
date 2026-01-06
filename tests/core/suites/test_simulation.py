# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for simulation-based test functions."""

import numpy as np


class TestSimulationImports:
    """Test that simulation functions can be imported."""

    def test_import_base_classes(self):
        from surfaces.test_functions.simulation import (
            ODESimulationFunction,
            SimulationFunction,
        )

        assert SimulationFunction is not None
        assert ODESimulationFunction is not None

    def test_import_dynamics_functions(self):
        from surfaces.test_functions.simulation import (
            DampedOscillatorFunction,
            LotkaVolterraFunction,
        )

        assert LotkaVolterraFunction is not None
        assert DampedOscillatorFunction is not None

    def test_import_chemical_functions(self):
        from surfaces.test_functions.simulation import ConsecutiveReactionFunction

        assert ConsecutiveReactionFunction is not None

    def test_import_electromagnetic_functions(self):
        from surfaces.test_functions.simulation import (
            RCFilterFunction,
            RLCCircuitFunction,
        )

        assert RLCCircuitFunction is not None
        assert RCFilterFunction is not None

    def test_simulation_functions_list(self):
        from surfaces.test_functions.simulation import simulation_functions

        assert len(simulation_functions) == 5


class TestLotkaVolterra:
    """Test Lotka-Volterra predator-prey function."""

    def test_instantiation(self):
        from surfaces.test_functions.simulation import LotkaVolterraFunction

        func = LotkaVolterraFunction()
        assert func is not None

    def test_search_space(self):
        from surfaces.test_functions.simulation import LotkaVolterraFunction

        func = LotkaVolterraFunction()
        space = func.search_space
        assert "alpha" in space
        assert "beta" in space
        assert "gamma" in space
        assert "delta" in space
        assert len(space) == 4

    def test_evaluation(self):
        from surfaces.test_functions.simulation import LotkaVolterraFunction

        func = LotkaVolterraFunction()
        result = func({"alpha": 1.0, "beta": 0.1, "gamma": 1.5, "delta": 0.075})
        assert isinstance(result, float)
        assert not np.isnan(result)

    def test_different_objectives(self):
        from surfaces.test_functions.simulation import LotkaVolterraFunction

        params = {"alpha": 1.0, "beta": 0.1, "gamma": 1.5, "delta": 0.075}

        func_var = LotkaVolterraFunction(objective_type="variance")
        func_prey = LotkaVolterraFunction(objective_type="prey_survival")
        func_bal = LotkaVolterraFunction(objective_type="balance")

        r1 = func_var(params)
        r2 = func_prey(params)
        r3 = func_bal(params)

        assert isinstance(r1, float)
        assert isinstance(r2, float)
        assert isinstance(r3, float)


class TestDampedOscillator:
    """Test damped oscillator function."""

    def test_instantiation(self):
        from surfaces.test_functions.simulation import DampedOscillatorFunction

        func = DampedOscillatorFunction()
        assert func is not None

    def test_search_space(self):
        from surfaces.test_functions.simulation import DampedOscillatorFunction

        func = DampedOscillatorFunction()
        space = func.search_space
        assert "damping" in space
        assert "stiffness" in space
        assert len(space) == 2

    def test_evaluation(self):
        from surfaces.test_functions.simulation import DampedOscillatorFunction

        func = DampedOscillatorFunction()
        result = func({"damping": 4.0, "stiffness": 4.0})
        assert isinstance(result, float)
        assert not np.isnan(result)

    def test_critical_damping_objective(self):
        from surfaces.test_functions.simulation import DampedOscillatorFunction

        func = DampedOscillatorFunction(objective_type="critical_damping")
        # For stiffness=4 and mass=1, critical damping is c=4
        result = func({"damping": 4.0, "stiffness": 4.0})
        assert result < 0.1  # Should be close to 0


class TestConsecutiveReaction:
    """Test consecutive reaction function."""

    def test_instantiation(self):
        from surfaces.test_functions.simulation import ConsecutiveReactionFunction

        func = ConsecutiveReactionFunction()
        assert func is not None

    def test_search_space(self):
        from surfaces.test_functions.simulation import ConsecutiveReactionFunction

        func = ConsecutiveReactionFunction()
        space = func.search_space
        assert "k1" in space
        assert "k2" in space
        assert len(space) == 2

    def test_evaluation(self):
        from surfaces.test_functions.simulation import ConsecutiveReactionFunction

        func = ConsecutiveReactionFunction(target_time=2.0)
        result = func({"k1": 1.0, "k2": 0.5})
        assert isinstance(result, float)
        assert not np.isnan(result)
        # Result should be negative (maximizing B)
        assert result < 0

    def test_different_objectives(self):
        from surfaces.test_functions.simulation import ConsecutiveReactionFunction

        params = {"k1": 1.0, "k2": 0.5}

        func_max_b = ConsecutiveReactionFunction(objective_type="max_B")
        func_int = ConsecutiveReactionFunction(objective_type="max_B_integral")
        func_sel = ConsecutiveReactionFunction(objective_type="selectivity")

        r1 = func_max_b(params)
        r2 = func_int(params)
        r3 = func_sel(params)

        assert isinstance(r1, float)
        assert isinstance(r2, float)
        assert isinstance(r3, float)


class TestRLCCircuit:
    """Test RLC circuit function."""

    def test_instantiation(self):
        from surfaces.test_functions.simulation import RLCCircuitFunction

        func = RLCCircuitFunction()
        assert func is not None

    def test_search_space(self):
        from surfaces.test_functions.simulation import RLCCircuitFunction

        func = RLCCircuitFunction()
        space = func.search_space
        assert "R" in space
        assert "L" in space
        assert "C" in space
        assert len(space) == 3

    def test_evaluation(self):
        from surfaces.test_functions.simulation import RLCCircuitFunction

        func = RLCCircuitFunction(target_frequency=100.0)
        result = func({"R": 10.0, "L": 0.01, "C": 2.5e-5})
        assert isinstance(result, float)
        assert not np.isnan(result)

    def test_resonance_match(self):
        from surfaces.test_functions.simulation import RLCCircuitFunction

        # f_0 = 1/(2*pi*sqrt(L*C))
        # For f_0=100Hz: L*C = 1/(4*pi^2*100^2) = 2.533e-6
        # L=0.01, C=2.533e-4
        func = RLCCircuitFunction(target_frequency=100.0, objective_type="resonance")
        L = 0.01
        C = 1.0 / (4 * np.pi**2 * 100**2 * L)
        result = func({"R": 10.0, "L": L, "C": C})
        assert result < 1.0  # Should be close to target


class TestRCFilter:
    """Test RC filter function."""

    def test_instantiation(self):
        from surfaces.test_functions.simulation import RCFilterFunction

        func = RCFilterFunction()
        assert func is not None

    def test_search_space(self):
        from surfaces.test_functions.simulation import RCFilterFunction

        func = RCFilterFunction()
        space = func.search_space
        assert "R" in space
        assert "C" in space
        assert len(space) == 2

    def test_evaluation(self):
        from surfaces.test_functions.simulation import RCFilterFunction

        func = RCFilterFunction(target_cutoff=1000.0)
        result = func({"R": 1000.0, "C": 1.59e-7})
        assert isinstance(result, float)
        assert not np.isnan(result)

    def test_cutoff_match(self):
        from surfaces.test_functions.simulation import RCFilterFunction

        # f_c = 1/(2*pi*R*C)
        # For f_c=1000Hz, R=1000: C = 1/(2*pi*1000*1000) = 1.59e-7
        func = RCFilterFunction(target_cutoff=1000.0, objective_type="cutoff")
        R = 1000.0
        C = 1.0 / (2 * np.pi * 1000 * R)
        result = func({"R": R, "C": C})
        assert result < 0.01  # Should be very close to target


class TestSimulationFunctionProperties:
    """Test common properties of simulation functions."""

    def test_all_have_search_space(self):
        from surfaces.test_functions.simulation import simulation_functions

        for func_cls in simulation_functions:
            func = func_cls()
            space = func.search_space
            assert isinstance(space, dict)
            assert len(space) >= 2

    def test_all_are_callable(self):
        from surfaces.test_functions.simulation import simulation_functions

        for func_cls in simulation_functions:
            func = func_cls()
            # Get a sample from search space
            params = {k: v[len(v) // 2] for k, v in func.search_space.items()}
            result = func(params)
            assert isinstance(result, float)

    def test_spec_attributes(self):
        from surfaces.test_functions.simulation import simulation_functions

        for func_cls in simulation_functions:
            assert hasattr(func_cls, "_spec")
            spec = func_cls._spec
            assert spec.get("simulation_based") is True
            assert spec.get("ode_based") is True
