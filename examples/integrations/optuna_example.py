"""Optuna Integration - Optimize a Surfaces test function."""

import optuna

from surfaces.test_functions.algebraic import AckleyFunction

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Create a 2-dimensional Ackley function
ackley = AckleyFunction()


def objective(trial):
    # Sample parameters from search space
    x0 = trial.suggest_float("x0", -5.0, 5.0)
    x1 = trial.suggest_float("x1", -5.0, 5.0)

    # Evaluate the function
    return ackley({"x0": x0, "x1": x1})


# Run optimization (Optuna minimizes by default)
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

# Results
print(f"Best value: {study.best_value:.6f}")
print(f"Best params: {study.best_params}")
