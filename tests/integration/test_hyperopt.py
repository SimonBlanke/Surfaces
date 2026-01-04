"""Integration tests for Hyperopt.

Tests that Surfaces functions work correctly with Hyperopt.
"""

import numpy as np
import pytest
from hyperopt import Trials, fmin, hp, tpe

from surfaces.test_functions.algebraic import SphereFunction

pytestmark = pytest.mark.slow


def test_hyperopt():
    """Test that Surfaces functions work with Hyperopt."""
    func = SphereFunction(n_dim=2)

    def objective(params):
        return func(params)

    space = {
        "x0": hp.uniform("x0", -5, 5),
        "x1": hp.uniform("x1", -5, 5),
    }

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials,
        rstate=np.random.default_rng(42),
        show_progressbar=False,
    )

    assert func(best) < 0.5
    assert abs(best["x0"]) < 1.0
    assert abs(best["x1"]) < 1.0
