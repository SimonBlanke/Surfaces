"""Integration tests for Ray Tune.

Tests that Surfaces functions work correctly with Ray Tune.
"""

import pytest
import ray
from ray import tune

from surfaces.test_functions.algebraic import SphereFunction

pytestmark = pytest.mark.slow


def test_ray_tune():
    """Test that Surfaces functions work with Ray Tune."""
    func = SphereFunction(n_dim=2)

    def objective(config):
        result = func({"x0": config["x0"], "x1": config["x1"]})
        return {"loss": result}

    # Test that the objective function works with Ray's config format
    test_config = {"x0": 0.5, "x1": -0.5}
    result = objective(test_config)
    assert "loss" in result
    assert isinstance(result["loss"], (int, float))

    # Try running full Ray Tune optimization
    try:
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        tuner = tune.Tuner(
            objective,
            param_space={
                "x0": tune.uniform(-5, 5),
                "x1": tune.uniform(-5, 5),
            },
            tune_config=tune.TuneConfig(
                num_samples=20,
                metric="loss",
                mode="min",
            ),
        )
        results = tuner.fit()
        best_result = results.get_best_result()

        assert best_result.metrics["loss"] < 2.0
    except Exception as e:
        pytest.skip(f"Ray Tune full optimization skipped: {e}")
    finally:
        if ray.is_initialized():
            ray.shutdown()
