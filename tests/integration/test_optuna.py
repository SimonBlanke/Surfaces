"""Integration tests for Optuna.

Tests that Surfaces functions work correctly with Optuna hyperparameter optimization.
"""

import optuna
import pytest

from surfaces.test_functions.algebraic import RastriginFunction, SphereFunction

pytestmark = pytest.mark.slow

optuna.logging.set_verbosity(optuna.logging.WARNING)


def test_optuna_sphere():
    """Test Optuna optimization with Sphere function."""
    func = SphereFunction(n_dim=2)

    def objective(trial):
        x0 = trial.suggest_float("x0", -5, 5)
        x1 = trial.suggest_float("x1", -5, 5)
        return func({"x0": x0, "x1": x1})

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, show_progress_bar=False)

    assert study.best_value < 0.9
    assert abs(study.best_params["x0"]) < 1.0
    assert abs(study.best_params["x1"]) < 1.0


def test_optuna_rastrigin():
    """Test Optuna with a more complex function."""
    func = RastriginFunction(n_dim=2)

    def objective(trial):
        params = {f"x{i}": trial.suggest_float(f"x{i}", -5, 5) for i in range(2)}
        return func(params)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, show_progress_bar=False)

    # Rastrigin has global minimum at origin with value 0
    assert study.best_value < 10.0
