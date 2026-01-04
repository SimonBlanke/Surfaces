"""Integration tests for bayesian-optimization.

Tests that Surfaces functions work correctly with BayesianOptimization.
"""

import pytest
from bayes_opt import BayesianOptimization

from surfaces.test_functions.algebraic import SphereFunction

pytestmark = pytest.mark.slow


def test_bayesian_optimization():
    """Test that Surfaces functions work with bayesian-optimization."""
    func = SphereFunction(n_dim=2)

    # bayesian-optimization maximizes, so we negate for minimization
    def objective(x0, x1):
        return -func({"x0": x0, "x1": x1})

    optimizer = BayesianOptimization(
        f=objective,
        pbounds={"x0": (-5, 5), "x1": (-5, 5)},
        random_state=42,
        verbose=0,
    )
    optimizer.maximize(init_points=5, n_iter=25)

    assert -optimizer.max["target"] < 0.5
    assert abs(optimizer.max["params"]["x0"]) < 1.0
    assert abs(optimizer.max["params"]["x1"]) < 1.0
