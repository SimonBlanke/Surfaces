"""Integration tests for Nevergrad.

Tests that Surfaces functions work correctly with Nevergrad optimization.
"""

import pytest
import nevergrad as ng

from surfaces.test_functions.algebraic import SphereFunction

pytestmark = pytest.mark.slow


def test_nevergrad_array_input():
    """Test Nevergrad with array-based parametrization."""
    func = SphereFunction(n_dim=2)

    def objective(x):
        return func(x)

    parametrization = ng.p.Array(shape=(2,), lower=-5, upper=5)
    optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=50)
    recommendation = optimizer.minimize(objective)

    result = recommendation.value
    assert func(result) < 0.5
    assert abs(result[0]) < 1.0
    assert abs(result[1]) < 1.0


def test_nevergrad_dict_input():
    """Test Nevergrad with dict-based instrumentation."""
    func = SphereFunction(n_dim=2)

    parametrization = ng.p.Instrumentation(
        x0=ng.p.Scalar(lower=-5, upper=5),
        x1=ng.p.Scalar(lower=-5, upper=5),
    )

    def objective(x0, x1):
        return func({"x0": x0, "x1": x1})

    optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=50)
    recommendation = optimizer.minimize(objective)

    result = func({"x0": recommendation.kwargs["x0"], "x1": recommendation.kwargs["x1"]})
    assert result < 0.5
