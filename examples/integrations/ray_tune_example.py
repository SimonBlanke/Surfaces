"""Ray Tune Integration - Optimize a Surfaces test function."""

import os

# Suppress Ray logs
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_AIR_NEW_OUTPUT"] = "0"

from ray import train, tune

from surfaces.test_functions.algebraic import SphereFunction

# Create a 3-dimensional Sphere function
sphere = SphereFunction(n_dim=3)


def objective(config):
    """Objective function for Ray Tune."""
    params = {"x0": config["x0"], "x1": config["x1"], "x2": config["x2"]}
    loss = sphere(params)
    train.report({"loss": loss})


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
        num_samples=10,
        metric="loss",
        mode="min",
    ),
    run_config=train.RunConfig(),
)

results = tuner.fit()

# Results
best = results.get_best_result(metric="loss", mode="min")
print(f"Best loss: {best.metrics['loss']:.6f}")
print(
    f"Best config: x0={best.config['x0']:.4f}, x1={best.config['x1']:.4f}, x2={best.config['x2']:.4f}"
)
