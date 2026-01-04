"""Ray Tune Integration - Optimize a Surfaces test function."""

from ray import tune
from ray.tune.search.basic_variant import BasicVariantGenerator

from surfaces.test_functions import SphereFunction

# Create a 3-dimensional Sphere function
sphere = SphereFunction(n_dim=3)


def objective(config):
    """Objective function for Ray Tune."""
    params = {"x0": config["x0"], "x1": config["x1"], "x2": config["x2"]}
    loss = sphere(params)
    return {"loss": loss}


# Define search space
search_space = {
    "x0": tune.uniform(-5.0, 5.0),
    "x1": tune.uniform(-5.0, 5.0),
    "x2": tune.uniform(-5.0, 5.0),
}

# Run optimization
tuner = tune.Tuner(
    objective,
    param_space=search_space,
    tune_config=tune.TuneConfig(
        num_samples=50,
        search_alg=BasicVariantGenerator(),
        metric="loss",
        mode="min",
    ),
)

results = tuner.fit()

# Results
best_result = results.get_best_result(metric="loss", mode="min")
print(f"Best loss: {best_result.metrics['loss']:.6f}")
print(f"Best config: {best_result.config}")
