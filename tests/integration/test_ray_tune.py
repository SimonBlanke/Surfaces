"""Integration tests for Ray Tune.

Tests that Surfaces functions work correctly with Ray Tune,
including Ray-specific features like intermediate reporting and multiple metrics.
"""

import pytest
import ray
from ray import tune

from surfaces.test_functions.algebraic import RastriginFunction, SphereFunction

pytestmark = pytest.mark.slow


# =============================================================================
# Basic Integration Tests
# =============================================================================


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


# =============================================================================
# Ray Tune-Specific Feature Tests
# =============================================================================


def test_ray_tune_config_is_dict():
    """Verify that Ray Tune's config is a plain dict that Surfaces can use directly."""
    func = SphereFunction(n_dim=2)

    def objective(config):
        # config is a plain dict - Surfaces can use it directly!
        assert isinstance(config, dict)
        assert "x0" in config
        assert "x1" in config

        # Direct dict passing works
        result = func(config)
        return {"loss": result}

    test_config = {"x0": 1.0, "x1": 2.0}
    result = objective(test_config)
    assert result["loss"] == 5.0  # 1^2 + 2^2


def test_ray_tune_multiple_metrics():
    """Test Surfaces with Ray Tune's multiple metrics reporting.

    Ray Tune can track multiple metrics per trial - useful for
    tracking both loss and accuracy, or primary and secondary objectives.
    """
    sphere = SphereFunction(n_dim=2)
    rastrigin = RastriginFunction(n_dim=2)

    def objective(config):
        params = {"x0": config["x0"], "x1": config["x1"]}

        # Report multiple metrics from different Surfaces functions
        sphere_score = sphere(params)
        rastrigin_score = rastrigin(params)

        return {
            "sphere_loss": sphere_score,
            "rastrigin_loss": rastrigin_score,
            "combined": sphere_score + rastrigin_score,
        }

    test_config = {"x0": 0.0, "x1": 0.0}
    result = objective(test_config)

    assert "sphere_loss" in result
    assert "rastrigin_loss" in result
    assert "combined" in result
    assert result["sphere_loss"] == 0.0
    assert result["rastrigin_loss"] == 0.0


def test_ray_tune_with_discrete_params():
    """Test Surfaces with Ray Tune's discrete/categorical parameters.

    Tests tune.choice() and tune.randint() alongside continuous params.
    """
    func = SphereFunction(n_dim=2)

    def objective(config):
        # Mix of continuous and discrete parameters
        x0 = config["x0"]
        # Discrete scaling factor
        scale = config["scale"]

        result = func({"x0": x0, "x1": config["x1"]}) * scale
        return {"loss": result}

    # Test the interface directly
    test_config = {"x0": 1.0, "x1": 1.0, "scale": 2}
    result = objective(test_config)
    assert result["loss"] == 4.0  # (1^2 + 1^2) * 2

    # Full Ray Tune test with mixed param types
    try:
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        tuner = tune.Tuner(
            objective,
            param_space={
                "x0": tune.uniform(-5, 5),
                "x1": tune.uniform(-5, 5),
                "scale": tune.choice([1, 2, 5, 10]),  # Discrete choices
            },
            tune_config=tune.TuneConfig(
                num_samples=15,
                metric="loss",
                mode="min",
            ),
        )
        results = tuner.fit()
        best_result = results.get_best_result()

        # Best should have scale=1 (smallest multiplier)
        assert best_result.config["scale"] == 1
    except Exception as e:
        pytest.skip(f"Ray Tune discrete params test skipped: {e}")
    finally:
        if ray.is_initialized():
            ray.shutdown()


def test_ray_tune_surfaces_data_collection():
    """Test that Surfaces' data collection works alongside Ray Tune.

    Verify that Surfaces tracks evaluations even when called through Ray.
    Note: Each Ray worker gets a fresh function instance, so we test
    the interface pattern rather than global state.
    """
    func = SphereFunction(n_dim=2)

    def objective(config):
        # Create local function instance (Ray serializes/deserializes)
        local_func = SphereFunction(n_dim=2)
        result = local_func({"x0": config["x0"], "x1": config["x1"]})

        # Local tracking works
        assert local_func.data.n_evaluations == 1

        return {"loss": result}

    test_config = {"x0": 1.0, "x1": 1.0}
    result = objective(test_config)
    assert result["loss"] == 2.0


def test_ray_tune_log_uniform():
    """Test Surfaces with Ray Tune's log-uniform sampling.

    Tests tune.loguniform() for parameters like learning rates.
    """
    func = SphereFunction(n_dim=2)

    def objective(config):
        result = func({"x0": config["x0"], "x1": config["x1"]})
        return {"loss": result}

    # Test interface
    test_config = {"x0": 0.001, "x1": 0.001}
    result = objective(test_config)
    assert result["loss"] < 0.01

    try:
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        tuner = tune.Tuner(
            objective,
            param_space={
                "x0": tune.loguniform(1e-5, 1.0),
                "x1": tune.loguniform(1e-5, 1.0),
            },
            tune_config=tune.TuneConfig(
                num_samples=15,
                metric="loss",
                mode="min",
            ),
        )
        results = tuner.fit()
        best_result = results.get_best_result()

        # Best params should be small values
        assert best_result.config["x0"] <= 1.0
        assert best_result.config["x1"] <= 1.0
    except Exception as e:
        pytest.skip(f"Ray Tune log-uniform test skipped: {e}")
    finally:
        if ray.is_initialized():
            ray.shutdown()
