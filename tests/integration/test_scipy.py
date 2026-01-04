"""Integration tests for scipy.optimize.

Tests that Surfaces functions work correctly with scipy optimization.
"""

import numpy as np
import pytest
from scipy.optimize import minimize

from surfaces.test_functions.algebraic import SphereFunction

pytestmark = pytest.mark.slow


def test_scipy_minimize():
    """Test that Surfaces functions work directly with scipy.optimize.minimize."""
    func = SphereFunction(n_dim=2)

    result = minimize(
        func,
        x0=[1.0, 1.0],
        bounds=[(-5, 5), (-5, 5)],
        method="L-BFGS-B",
    )

    assert result.success
    assert abs(result.fun) < 0.01
    assert np.allclose(result.x, [0, 0], atol=0.1)
