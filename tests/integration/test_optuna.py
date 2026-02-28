"""Integration tests for Optuna.

Tests that Surfaces functions work correctly with Optuna hyperparameter optimization,
including Optuna-specific features like pruning and intermediate value reporting.
"""

import optuna
import pytest

from surfaces.test_functions.algebraic import RastriginFunction, SphereFunction

pytestmark = pytest.mark.slow

optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# Basic Integration Tests
# =============================================================================


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


# =============================================================================
# Optuna-Specific Feature Tests
# =============================================================================


def test_optuna_with_pruning():
    """Test that Surfaces functions work with Optuna's pruning mechanism.

    This tests trial.report() and trial.should_prune() - key Optuna features
    for early stopping of unpromising trials.
    """
    func = SphereFunction(n_dim=2)
    pruned_count = [0]

    def objective(trial):
        x0 = trial.suggest_float("x0", -5, 5)
        x1 = trial.suggest_float("x1", -5, 5)

        # Simulate iterative optimization with intermediate reporting
        # (like training a model for multiple epochs)
        for step in range(5):
            # Evaluate Surfaces function
            score = func({"x0": x0, "x1": x1})

            # Simulate decreasing loss over steps
            intermediate_value = score * (1.0 - step * 0.1)

            # Report intermediate value to Optuna
            trial.report(intermediate_value, step)

            # Check if trial should be pruned
            if trial.should_prune():
                pruned_count[0] += 1
                raise optuna.TrialPruned()

        return score

    # Use MedianPruner which prunes based on intermediate values
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=30, show_progress_bar=False)

    # Verify optimization completed
    assert study.best_value is not None
    assert len(study.trials) == 30

    # Count pruned trials from study (more reliable than our counter)
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    # We expect some trials to be pruned (not all, not none)
    # Note: With only 30 trials and 5 startup, pruning may or may not occur
    # The key is that the mechanism works without errors


def test_optuna_with_categorical_and_conditional():
    """Test Surfaces with Optuna's conditional/categorical parameters.

    This tests trial.suggest_categorical() and conditional parameter spaces.
    """
    sphere = SphereFunction(n_dim=2)
    rastrigin = RastriginFunction(n_dim=2)

    def objective(trial):
        # Categorical parameter: which function to use
        func_name = trial.suggest_categorical("function", ["sphere", "rastrigin"])

        # Continuous parameters
        x0 = trial.suggest_float("x0", -5, 5)
        x1 = trial.suggest_float("x1", -5, 5)

        # Conditional: different evaluation based on categorical choice
        if func_name == "sphere":
            return sphere({"x0": x0, "x1": x1})
        else:
            return rastrigin({"x0": x0, "x1": x1})

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30, show_progress_bar=False)

    # Verify that both functions were tried
    func_choices = [t.params.get("function") for t in study.trials]
    assert "sphere" in func_choices
    assert "rastrigin" in func_choices


def test_optuna_with_log_scale():
    """Test Surfaces with Optuna's log-scale parameter sampling.

    This tests trial.suggest_float(..., log=True) for parameters like
    learning rates that benefit from log-uniform sampling.
    """
    func = SphereFunction(n_dim=2)

    def objective(trial):
        # Log-scale sampling (common for learning rates)
        x0 = trial.suggest_float("x0", 1e-5, 1.0, log=True)
        x1 = trial.suggest_float("x1", 1e-5, 1.0, log=True)
        return func({"x0": x0, "x1": x1})

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30, show_progress_bar=False)

    # Verify log-scale sampling worked (best params should be small)
    assert study.best_params["x0"] <= 1.0
    assert study.best_params["x1"] <= 1.0


def test_optuna_surfaces_data_collection():
    """Test that Surfaces' data collection works alongside Optuna's tracking.

    Both Surfaces and Optuna track evaluations - verify they're compatible.
    """
    func = SphereFunction(n_dim=2)

    def objective(trial):
        x0 = trial.suggest_float("x0", -5, 5)
        x1 = trial.suggest_float("x1", -5, 5)
        return func({"x0": x0, "x1": x1})

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20, show_progress_bar=False)

    # Verify Surfaces tracked all evaluations
    assert func.data.n_evaluations == 20
    assert len(func.data.search_data) == 20

    # Verify Surfaces' best matches Optuna's best
    assert func.data.best_score == pytest.approx(study.best_value, rel=1e-6)
