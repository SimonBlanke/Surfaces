"""Integration tests for SMAC.

Tests that Surfaces functions work correctly with SMAC.
"""

import pytest
from ConfigSpace import ConfigurationSpace, Float
from smac import HyperparameterOptimizationFacade, Scenario

from surfaces.test_functions.algebraic import SphereFunction

pytestmark = pytest.mark.slow


def test_smac():
    """Test that Surfaces functions work with SMAC."""
    func = SphereFunction(n_dim=2)

    def objective(config, seed=0):
        return func({"x0": config["x0"], "x1": config["x1"]})

    configspace = ConfigurationSpace(seed=42)
    configspace.add(Float("x0", (-5.0, 5.0)))
    configspace.add(Float("x1", (-5.0, 5.0)))

    scenario = Scenario(
        configspace,
        deterministic=True,
        n_trials=50,
    )

    smac = HyperparameterOptimizationFacade(
        scenario,
        objective,
        overwrite=True,
    )
    incumbent = smac.optimize()

    result = func({"x0": incumbent["x0"], "x1": incumbent["x1"]})
    assert result < 1.0
