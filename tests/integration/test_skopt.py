"""Integration tests for scikit-optimize.

Tests that Surfaces functions work correctly with skopt.
"""

import pytest
from skopt import gp_minimize

from surfaces.test_functions.algebraic import SphereFunction

pytestmark = pytest.mark.slow


def test_skopt_gp_minimize():
    """Test that Surfaces functions work with scikit-optimize."""
    func = SphereFunction(n_dim=2)

    def objective(x):
        return func(x)

    result = gp_minimize(
        objective,
        [(-5.0, 5.0), (-5.0, 5.0)],
        n_calls=30,
        random_state=42,
    )

    assert result.fun < 0.5
    assert abs(result.x[0]) < 1.0
    assert abs(result.x[1]) < 1.0
